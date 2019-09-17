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

import torch as th
import torch.nn as nn
import horovod.torch as hvd

#import pykaldi related modules
import kaldi.fstext as kaldi_fst
import kaldi.hmm as kaldi_hmm
import kaldi.matrix as kaldi_matrix
import kaldi.lat as kaldi_lat
import kaldi.alignment as kaldi_align
import kaldi.util as kaldi_util
import kaldi.chain as kaldi_chain
import kaldi.tree as kaldi_tree

from data import SpeechDataset, SeqDataloader
from models import LSTMStack, NnetAM
from ops import ops
from utils import utils

def main():
    parser = argparse.ArgumentParser()                                                                                 
    parser.add_argument("-config")       
    parser.add_argument("-data", help="data yaml file")
    parser.add_argument("-dataPath", default='', type=str, help="path of data files") 
    parser.add_argument("-seed_model", default='', help="the seed nerual network model") 
    parser.add_argument("-exp_dir", help="the directory to save the outputs") 
    parser.add_argument("-transform", help="feature transformation matrix or mvn statistics")
    parser.add_argument("-ali_dir", help="the directory to load trans_model and tree used for alignments") 
    parser.add_argument("-lang_dir", help="the lexicon directory to load L.fst")
    parser.add_argument("-chain_dir", help="the directory to load trans_model, tree and den.fst for chain model")
    parser.add_argument("-lr", type=float, help="set the learning rate")
    parser.add_argument("-momentum", default=0, type=float, help="set the momentum") 
    parser.add_argument("-weight_decay", default=1e-4, type=float, help="set the L2 regularization weight") 
    parser.add_argument("-batch_size", default=32, type=int, help="Override the batch size in the config")                         
    parser.add_argument("-data_loader_threads", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("-max_grad_norm", default=5, type=float, help="max_grad_norm for gradient clipping")                     
    parser.add_argument("-sweep_size", default=100, type=float, help="process n hours of data per sweep (default:100)")
    parser.add_argument("-num_epochs", default=1, type=int, help="number of training epochs (default:1)")
    parser.add_argument("-anneal_lr_epoch", default=2, type=int, help="start to anneal the learning rate from this epoch")  
    parser.add_argument("-anneal_lr_ratio", default=0.5, type=float, help="the ratio to anneal the learning rate ratio")
    parser.add_argument('-print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-save_freq', default=1000, type=int, metavar='N', help='save model frequency (default: 1000)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["sweep_size"] = args.sweep_size

    print("pytorch version:{}".format(th.__version__))

    with open(args.data) as f:
        data = yaml.safe_load(f)
        config["source_paths"] = [j for i, j in data['clean_source'].items()]
        if 'dir_noise' in data:
            config["dir_noise_paths"] = [j for i, j in data['dir_noise'].items()]
        if 'rir' in data:
            config["rir_paths"] = [j for i, j in data['rir'].items()]
    config['data_path'] = args.dataPath

    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))

    # Initialize Horovod
    hvd.init()

    th.cuda.set_device(hvd.local_rank())

    print("Run experiments with world size {}".format(hvd.size()))

    dataset = SpeechDataset(config)
    transform=None
    if args.transform is not None and os.path.isfile(args.transform):
        with open(args.transform, 'rb') as f:
            transform = pickle.load(f)
            dataset.transform = transform

    train_dataloader = SeqDataloader(dataset,
                                    batch_size=args.batch_size,
                                    num_workers = args.data_loader_threads,
                                    distributed=True,
                                    test_only=False)

    print("Data loader set up successfully!")
    print("Number of minibatches: {}".format(len(train_dataloader)))

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    # ceate model
    model_config = config["model_config"]
    lstm = LSTMStack(model_config["feat_dim"], model_config["hidden_size"], model_config["num_layers"], model_config["dropout"], True)
    model = NnetAM(lstm, model_config["hidden_size"]*2, model_config["label_size"])

    model.cuda()

    # setup the optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    # Broadcast parameters and opterimizer state from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    if os.path.isfile(args.seed_model):
        checkpoint = th.load(args.seed_model)                                            
        state_dict = checkpoint['model']
        from collections import OrderedDict                  
        new_state_dict = OrderedDict()                       
        for k, v in state_dict.items():                      
            header = k[:7]                                   
            name = k[7:] # remove 'module.' of dataparallel  
            new_state_dict[name]=v                           
        if header == "module.":                              
            model.load_state_dict(new_state_dict)            
        else:                                                
            model.load_state_dict(state_dict)                
        print("=> loaded checkpoint '{}' ".format(args.seed_model))                      

    ali_model = args.ali_dir + "/final.mdl"
    ali_tree = args.ali_dir + "/tree"
    L_fst = args.lang_dir + "/L.fst"
    disambig = args.lang_dir + "/phones/disambig.int"

    den_fst = kaldi_fst.StdVectorFst.read(args.chain_dir + "/den.fst")
    chain_model_path = args.chain_dir + "/0.trans_mdl"
    chain_tree_path = args.chain_dir + "/tree"

    if os.path.isfile(chain_model_path):
       chain_trans_model = kaldi_hmm.TransitionModel()
       with kaldi_util.io.xopen(chain_model_path) as ki:
           chain_trans_model.read(ki.stream(), ki.binary)
    else:
       sys.stderr.write('ERROR: The trans_model %s does not exist!\n'%(trans_model))
       sys.exit(0)
  
    chain_tree = kaldi_tree.ContextDependency()
    with kaldi_util.io.xopen(chain_tree_path) as ki:
        chain_tree.read(ki.stream(), ki.binary)
 
     # chain supervision options
    supervision_opts = kaldi_chain.SupervisionOptions()
    supervision_opts.convert_to_pdfs = True
    supervision_opts.frame_subsampling_factor = 3
    supervision_opts.left_tolerance = 5
    supervision_opts.right_tolerance = 5

    # chain training options
    chain_opts = kaldi_chain.ChainTrainingOptions()
    chain_opts.leaky_hmm_coefficient = 1e-4
    chain_opts.xent_regularize = 1e-4

    # setup the aligner
    aligner = kaldi_align.MappedAligner.from_files(ali_model, ali_tree, L_fst, None,
                                 disambig, None, 
                                 beam=10,
                                 transition_scale=1.0, 
                                 self_loop_scale=0.1, 
                                 acoustic_scale=0.1)
    den_graph = kaldi_chain.DenominatorGraph(den_fst, model_config["label_size"])

    model.train()
    for epoch in range(args.num_epochs): 

        # anneal learning rate
        if epoch > args.anneal_lr_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.anneal_lr_ratio

        run_train_epoch(model, optimizer,
                       train_dataloader, 
                       epoch, 
                       chain_trans_model,
                       chain_tree,
                       supervision_opts,
                       aligner,
                       den_graph,
                       chain_opts,
                       args)

        # save model
        if hvd.rank() == 0:
            checkpoint={}
            checkpoint['model']=model.state_dict()
            checkpoint['optimizer']=optimizer.state_dict()
            checkpoint['epoch']=epoch
            output_file=args.exp_dir + '/chain.model.'+ str(epoch) +'.tar'
            th.save(checkpoint, output_file)

def run_train_epoch(model, optimizer, dataloader, epoch, trans_model, tree, supervision_opts, aligner, den, chain_opts, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    grad_norm = utils.AverageMeter('grad_norm', ':.4e')
    progress = utils.ProgressMeter(len(dataloader), batch_time, losses, grad_norm, 
                             prefix="Epoch: [{}]".format(epoch))

    criterion = ops.ChainObjtiveFunction.apply
    end = time.time()
    for i, batch in enumerate(dataloader):
        feat = batch["x"] 
        label = batch["y"]                               
        num_frs = batch["num_frs"]                       
        utt_ids = batch["utt_ids"]                       
        aux = batch["aux"]  #word labels for se loss
    
        x = feat.to(th.float32)
        x = x.unfold(1, 1, supervision_opts.frame_subsampling_factor).squeeze(-1)
        x = x.cuda()   
        y = label.squeeze(2) 

        loss = 0.0
        prediction = model(x)
        for j in range(len(num_frs)):                   
            trans_ids = y[j, :num_frs[j]].tolist()
            phone_ali = aligner.to_phone_alignment(trans_ids)
            
            phones = list()
            durations = list()
            for item in phone_ali:
                phones.append(item[0])
                durations.append(item[2])
               
            proto_supervision = kaldi_chain.alignment_to_proto_supervision(supervision_opts, phones, durations)
            supervision = kaldi_chain.proto_supervision_to_supervision(tree, trans_model, proto_supervision, True)

            loglike_j = prediction[j, :supervision.frames_per_sequence,:]
            loss += criterion(loglike_j, den, supervision, chain_opts) 
            
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping (th 5.0)
        norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        grad_norm.update(norm)

        # update the loss
        tot_frs = np.array(num_frs).sum()
        losses.update(loss.item()/tot_frs)

        # measure the elapsed time
        batch_time.update(time.time() - end)

        # save model
        if hvd.rank() == 0 and i % args.save_freq == 0:
            checkpoint={}
            checkpoint['model']=model.state_dict()
            checkpoint['optimizer']=optimizer.state_dict()
            output_file=args.exp_dir + '/chain.model.'+ str(i) +'.tar'
            th.save(checkpoint, output_file)

        if hvd.rank() == 0 and i % args.print_freq == 0:
            progress.print(i)

if __name__ == '__main__':
    main()
