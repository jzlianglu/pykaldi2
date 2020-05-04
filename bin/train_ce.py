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

import feature
from data import SpeechDataset, ChunkDataloader, SeqDataloader
from models import LSTMStack, NnetAM
from utils import utils

def main():
    parser = argparse.ArgumentParser()                                                                                 
    parser.add_argument("-config")                                                                                     
    parser.add_argument("-exp_dir")    
    parser.add_argument("-data")                                                                                
    parser.add_argument("-lr", default=1, type=float, help="Override the LR in the config")                                       
    parser.add_argument("-batch_size", default=32, type=int, help="Override the batch size in the config")                         
    parser.add_argument("-data_loader_threads", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("-max_grad_norm", default=5, type=float, help="max_grad_norm for gradient clipping")                     
    parser.add_argument("-sweep_size", default=200, type=float, help="process n hours of data per sweep (default:200)")
    parser.add_argument("-num_epochs", default=1, type=int, help="number of training epochs (default:1)") 
    parser.add_argument("-global_mvn", default=False, type=bool, help="if apply global mean and variance normalization")
    parser.add_argument("-resume_from_model", type=str, help="the model from which you want to resume training")         
    parser.add_argument("-dropout", type=float, help="set the dropout ratio")                                        
    parser.add_argument("-aneal_lr_epoch", default=2, type=int, help="start to aneal the learning rate from this epoch") 
    parser.add_argument("-aneal_lr_ratio", default=0.5, type=float, help="the ratio to aneal the learning rate")
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["sweep_size"] = args.sweep_size
    with open(args.data) as f:
        data = yaml.safe_load(f)
        config["source_paths"] = [j for i, j in data['clean_source'].items()]

    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    trainset = SpeechDataset(config)
    train_dataloader = ChunkDataloader(trainset, 
                                       batch_size=args.batch_size,
                                       num_workers=args.data_loader_threads,
                                       global_mvn=args.global_mvn)

    if args.global_mvn:
        transform = feature.preprocess.GlobalMeanVarianceNormalization()
        print("Estimating global mean and variance of feature vectors...")
        transform.learn_mean_and_variance_from_train_loader(train_dataloader,
                                                        train_dataloader.stream_keys_for_transform,
                                                        n_sample_to_use=2000)
        train_dataloader.transform = transform
        print("Global mean and variance transform trained successfully!")

        with open(args.exp_dir+"/transform.pkl", 'wb') as f:
            pickle.dump(transform, f, pickle.HIGHEST_PROTOCOL)

    print("Data loader set up successfully!")
    print("Number of minibatches: {}".format(len(train_dataloader)))

    # ceate model
    model_config = config["model_config"]
    lstm = LSTMStack(model_config["feat_dim"], model_config["hidden_size"], model_config["num_layers"], model_config["dropout"], True)
    model = NnetAM(lstm, model_config["hidden_size"]*2, model_config["label_size"])

    # Start training
    th.backends.cudnn.enabled = True
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.cuda()
    model = nn.DataParallel(model, dim=0)

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

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

         # aneal learning rate
        if epoch > args.aneal_lr_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.aneal_lr_ratio

        run_train_epoch(model, optimizer, criterion, train_dataloader, epoch, args)

        # save model
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

        x = feat.to(th.float32)
        y = label.squeeze(2).long()
        
        x = x.cuda()
        y = y.cuda()

        prediction = model(x)
        loss = criterion(prediction.view(-1, prediction.shape[2]), y.view(-1))

        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        grad_norm.update(norm)

        # update loss
        losses.update(loss.item(), x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.print(i)

if __name__ == '__main__':
    main()
