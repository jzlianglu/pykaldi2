"""
Copyright (c) 2019 Microsoft Corporation. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import yaml
import argparse
import numpy as np
import os
import sys
import time
import json
import pickle

import torch as th
import torch.nn as nn

from reader.preprocess import GlobalMeanVarianceNormalization
from data import SpeechDataset, ChunkDataloader, SeqDataloader
from utils import utils
from models import transformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir")
    parser.add_argument("-dataPath", default='', type=str, help="path of data files")
    parser.add_argument("-train_config")
    parser.add_argument("-data_config")
    parser.add_argument("-lr", default=0.0001, type=float, help="Override the LR in the config")
    parser.add_argument("-batch_size", default=32, type=int, help="Override the batch size in the config")
    parser.add_argument("-data_loader_threads", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("-max_grad_norm", default=5, type=float, help="max_grad_norm for gradient clipping")
    parser.add_argument("-sweep_size", default=200, type=float, help="process n hours of data per sweep (default:200)")
    parser.add_argument("-num_epochs", default=1, type=int, help="number of training epochs (default:1)")
    parser.add_argument("-global_mvn", default=False, type=bool, help="if apply global mean and variance normalization")
    parser.add_argument("-resume_from_model", type=str, help="the model from which you want to resume training")
    parser.add_argument("-dropout", default=0, type=float, help="set the dropout ratio")
    parser.add_argument("-warmup_step", default=4000, type=int, help="the number of warmup steps for the lr schedule")
    parser.add_argument("-nheads", default=4, type=int, help="the number of attention heads") 
    parser.add_argument("-dim_model", default=512, type=int, help="the model dimension") 
    parser.add_argument("-ff_size", default=2048, type=int, help="the size of feed-forward layer")
    parser.add_argument("-nlayers", default=6, type=int, help="the number of layers") 
    parser.add_argument("-look_ahead", default=-1, type=int, help="the number of frames to look ahead")
    parser.add_argument('-print_freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('-hvd', default=True, type=bool, help="whether to use horovod for training")

    args = parser.parse_args()

    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    config["sweep_size"] = args.sweep_size
    with open(args.data_config) as f:
        data = yaml.safe_load(f)
        config["source_paths"] = [j for i, j in data['clean_source'].items()]
        if 'dir_noise' in data:
            config["dir_noise_paths"] = [j for i, j in data['dir_noise'].items()]
        if 'rir' in data:
            config["rir_paths"] = [j for i, j in data['rir'].items()]

    config['data_path'] = args.dataPath

    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))

     # Initialize Horovod
    if args.hvd:
        import horovod.torch as hvd
        hvd.init()
        th.cuda.set_device(hvd.local_rank())
        print("Run experiments with world size {}".format(hvd.size()))

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    trainset = SpeechDataset(config)
    train_dataloader = SeqDataloader(trainset,
                                    batch_size=args.batch_size,
                                    num_workers = args.data_loader_threads,
                                    distributed=True,
                                    test_only=False)

    if args.global_mvn:
        transform = GlobalMeanVarianceNormalization()
        print("Estimating global mean and variance of feature vectors...")
        transform.learn_mean_and_variance_from_train_loader(trainset,
                                                        trainset.stream_idx_for_transform,
                                                        n_sample_to_use=2000)
        trainset.transform = transform
        print("Global mean and variance transform trained successfully!")

        with open(args.exp_dir+"/transform.pkl", 'wb') as f:
            pickle.dump(transform, f, pickle.HIGHEST_PROTOCOL)

    print("Data loader set up successfully!")
    print("Number of minibatches: {}".format(len(train_dataloader)))

    # ceate model
    model_config = config["model_config"]
    model = transformer.TransformerAM(model_config["feat_dim"], args.dim_model, args.nheads, args.ff_size, args.nlayers, args.dropout, model_config["label_size"])

    # Start training
    th.backends.cudnn.enabled = True
    if th.cuda.is_available():
        model.cuda()

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    if args.hvd:
        # Broadcast parameters and opterimizer state from rank 0 to all other processes.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Add Horovod Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # criterion
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    start_epoch = 0
    if args.resume_from_model:

        assert os.path.isfile(args.resume_from_model), "ERROR: model file {} does not exit!".format(args.resume_from_model)

        checkpoint = th.load(args.resume_from_model)
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' ".format(args.resume_from_model))

    model.train()
    for epoch in range(start_epoch, args.num_epochs):

        run_train_epoch(model, optimizer, criterion, train_dataloader, epoch, args)

        # save model
        if not args.hvd or hvd.rank()== 0:
            checkpoint={}
            checkpoint['model']=model.state_dict()
            checkpoint['optimizer']=optimizer.state_dict()
            checkpoint['epoch']=epoch
            output_file=args.exp_dir + '/model.'+ str(epoch) +'.tar'
            th.save(checkpoint, output_file)

def run_train_epoch(model, optimizer, criterion, train_dataloader, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    grad_norm = utils.AverageMeter('grad_norm', ':.4e')
    progress = utils.ProgressMeter(len(train_dataloader), batch_time, losses, grad_norm,
                             prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    # trainloader is an iterator. This line extract one minibatch at one time
    for i, data in enumerate(train_dataloader, 0):
        feat = data["x"]
        label = data["y"]
        num_frs = data["num_frs"]
        utt_ids = data["utt_ids"]

        x = feat.to(th.float32)
        y = label.squeeze(2).long()

        if th.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        x = x.transpose(0, 1)
        key_padding_mask = th.ones((x.size(1), x.size(0)))
         
        for utt in range(len(num_frs)):
            key_padding_mask[utt, :num_frs[utt]] = 0

        src_mask = None
        if(args.look_ahead > -1):
            src_mask = th.tril(th.ones(x.size(0), x.size(0)), diagonal=args.look_ahead)
            src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
            src_mask = src_mask.cuda()

        key_padding_mask = key_padding_mask.bool().cuda()
        prediction = model(x, src_mask, key_padding_mask)
        prediction = prediction.transpose(0, 1).contiguous()
        loss = criterion(prediction.view(-1, prediction.size(2)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
  
        #update lr
        step = len(train_dataloader) * epoch + i + 1
        lr = utils.noam_decay(step, args.warmup_step, args.lr)
        for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.step()

        grad_norm.update(norm)

        # update loss
        losses.update(loss.item(), x.size(1))

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
    #        if not args.hvd or hvd.rank() == 0:
            progress.print(i)

if __name__ == '__main__':
    main()
