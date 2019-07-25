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
import horovod.torch as hvd

#import pykaldi related modules
import kaldi.fstext as kaldi_fst
import kaldi.hmm as kaldi_hmm
import kaldi.matrix as kaldi_matrix
import kaldi.lat as kaldi_lat
import kaldi.decoder as kaldi_decoder
import kaldi.util as kaldi_util
from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions

from data import SpeechDataset, SeqDataloader
from models import LSTMStack, NnetAM
from ops import ops
from utils import utils

def main():
    parser = argparse.ArgumentParser()                                                                                 
    parser.add_argument("-config")                                                                                    
    parser.add_argument("-data", help="data yaml file")
    parser.add_argument("-data_path", default='', type=str, help="path of data files") 
    parser.add_argument("-seed_model", help="the seed nerual network model")                                                                                  
    parser.add_argument("-exp_dir", help="the directory to save the outputs")
    parser.add_argument("-transform", help="feature transformation matrix or mvn statistics") 
    parser.add_argument("-criterion", type=str, choices=["mmi", "mpfe", "smbr"], help="set the sequence training crtierion") 
    parser.add_argument("-trans_model", help="the HMM transistion model, used for lattice generation") 
    parser.add_argument("-prior_path", help="the prior for decoder, usually named as final.occs in kaldi setup")
    parser.add_argument("-den_dir", help="the decoding graph directory to find HCLG and words.txt files")
    parser.add_argument("-lr", type=float, help="set the learning rate")
    parser.add_argument("-ce_ratio", default=0.1, type=float, help="the ratio for ce regularization")
    parser.add_argument("-momentum", default=0, type=float, help="set the momentum")                                      
    parser.add_argument("-batch_size", default=32, type=int, help="Override the batch size in the config")                         
    parser.add_argument("-data_loader_threads", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("-max_grad_norm", default=5, type=float, help="max_grad_norm for gradient clipping")                     
    parser.add_argument("-sweep_size", default=100, type=float, help="process n hours of data per sweep (default:60)")
    parser.add_argument("-num_epochs", default=1, type=int, help="number of training epochs (default:1)") 
    parser.add_argument('-print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-save_freq', default=1000, type=int, metavar='N', help='save model frequency (default: 1000)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config['data_path'] = args.data_path

    config["sweep_size"] = args.sweep_size

    print("pytorch version:{}".format(th.__version__))

    with open(args.data) as f:
        data = yaml.safe_load(f)
        config["source_paths"] = [j for i, j in data['clean_source'].items()]

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
    optimizer = th.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Broadcast parameters and opterimizer state from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    if os.path.isfile(args.seed_model):
        checkpoint = th.load(args.seed_model)                                            
        state_dict = checkpoint['model']                                            
        model.load_state_dict(state_dict)                                           
        print("=> loaded checkpoint '{}' ".format(args.seed_model))                      
    else:
        sys.stderr.write('ERROR: The model file %s does not exist!\n'%(model_file))
        sys.exit(0)      

    HCLG = args.den_dir + "/HCLG.fst"
    words_txt = args.den_dir + "/words.txt"
    silence_phones = args.den_dir + "/phones/silence.csl"

    if not os.path.isfile(HCLG):
        sys.stderr.write('ERROR: The HCLG file %s does not exist!\n'%(HCLG))
        sys.exit(0)
    
    if not os.path.isfile(words_txt):
        sys.stderr.write('ERROR: The words.txt file %s does not exist!\n'%(words_txt))
        sys.exit(0)
 
    if not os.path.isfile(silence_phones):
        sys.stderr.write('ERROR: The silence phone file %s does not exist!\n'%(silence_phones))
        sys.exit(0)
    with open(silence_phones) as f:
        silence_ids = [int(i) for i in f.readline().strip().split(':')]
        f.close()

    if os.path.isfile(args.trans_model):
       trans_model = kaldi_hmm.TransitionModel()
       with kaldi_util.io.xopen(args.trans_model) as ki:
           trans_model.read(ki.stream(), ki.binary)
    else:
       sys.stderr.write('ERROR: The trans_model %s does not exist!\n'%(args.trans_model))
       sys.exit(0)
    
    # now we can setup the decoder
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = config["decoder_config"]["beam"]
    decoder_opts.lattice_beam = config["decoder_config"]["lattice_beam"]
    decoder_opts.max_active = config["decoder_config"]["max_active"]
    acoustic_scale = config["decoder_config"]["acoustic_scale"]
    decoder_opts.determinize_lattice = False  #To produce raw state-level lattice instead of compact lattice
    asr_decoder = MappedLatticeFasterRecognizer.from_files(
        args.trans_model, HCLG, words_txt,
        acoustic_scale=acoustic_scale, decoder_opts=decoder_opts)

    prior = kaldi_util.io.read_matrix(args.prior_path).numpy()
    log_prior = th.tensor(np.log(prior[0]/np.sum(prior[0])), dtype=th.float)

    model.train()
    for epoch in range(args.num_epochs): 

        run_train_epoch(model, optimizer,
                       log_prior.cuda(), 
                       train_dataloader, 
                       epoch, 
                       asr_decoder, 
                       trans_model,
                       silence_ids,
                       args)

        # save model
        if hvd.rank() == 0:
            checkpoint={}
            checkpoint['model']=model.state_dict()
            checkpoint['optimizer']=optimizer.state_dict()
            checkpoint['epoch']=epoch
            output_file=args.exp_dir + '/model.se.'+ str(epoch) +'.tar'
            th.save(checkpoint, output_file)

def run_train_epoch(model, optimizer, log_prior, dataloader, epoch, asr_decoder, trans_model, silence_ids, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    grad_norm = utils.AverageMeter('grad_norm', ':.4e')
    progress = utils.ProgressMeter(len(dataloader), batch_time, losses, grad_norm, 
                             prefix="Epoch: [{}]".format(epoch))

    ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
 
    if args.criterion == "mmi":
        se_criterion = ops.MMIFunction.apply
    else:
        se_criterion = ops.sMBRFunction.apply

    end = time.time()
    for i, batch in enumerate(dataloader, 0):
        feat = batch["x"] 
        label = batch["y"]   #pdf-ids for ce loss
        num_frs = batch["num_frs"]                       
        utt_ids = batch["utt_ids"]                       
        aux = batch["aux"]  #trans_ids for se loss

        x = feat.to(th.float32)                        
        y = label.long() 
        x = x.cuda()                                    
        y = y.cuda()
                                        
        prediction = model(x) 
        ce_loss = ce_criterion(prediction.view(-1, prediction.shape[2]), y.view(-1))
 
        se_loss = 0.0                         
        for j in range(len(num_frs)):                   
            log_like_j=prediction[j,:,:]       
            log_like_j= log_like_j[:num_frs[j],:]        
            log_like_j = log_like_j - log_prior
            #trans_id = label[j, :num_frs[j], 0].tolist()
            trans_id = th.from_numpy(aux[j][0][0].astype(int)).tolist()
        #    print(len(trans_id), num_frs[j])
  
            if args.criterion == "mmi":
                se_loss += se_criterion(log_like_j, asr_decoder, trans_model, trans_id)
            else:
                se_loss += se_criterion(log_like_j, asr_decoder, trans_model, trans_id, args.criterion, silence_ids)

        loss = se_loss.cuda() + args.ce_ratio * ce_loss
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping (th 5.0)
        norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        grad_norm.update(norm)

        # update loss
        tot_frs = np.array(num_frs).sum()
        losses.update(loss.item()/tot_frs)

        # measure elapsed time
        batch_time.update(time.time() - end)

        # save model
        if hvd.rank() == 0 and i % args.save_freq == 0:
            checkpoint={}
            checkpoint['model']=model.state_dict()
            checkpoint['optimizer']=optimizer.state_dict()
            output_file=args.exp_dir + '/model.se.'+ str(i) +'.tar'
            th.save(checkpoint, output_file)

        if hvd.rank() == 0 and i % args.print_freq == 0:
            progress.print(i)

if __name__ == '__main__':
    main()
