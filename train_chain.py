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
import kaldi.cudamatrix as kaldi_cudamatrix
import kaldi.lat as kaldi_lat
import kaldi.decoder as kaldi_decoder
import kaldi.alignment as kaldi_align
import kaldi.util as kaldi_util
import kaldi.chain as kaldi_chain
import kaldi.tree as kaldi_tree
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
    parser.add_argument("-dataPath", default='', type=str, help="path of data files") 
    parser.add_argument("-seed_model", help="the seed nerual network model")                                                                                  
    parser.add_argument("-exp_dir", help="the directory to save the outputs") 
    parser.add_argument("-transform", help="feature transformation matrix or mvn statistics")
    parser.add_argument("-criterion", type=str, choices=["mmi", "mpfe", "smbr"], help="set the sequence training crtierion") 
    parser.add_argument("-trans_model", help="the HMM transistion model directory") 
    parser.add_argument("-prior_path", help="the prior for decoder, usually named as final.occs in kaldi setup")
    parser.add_argument("-den_dir", help="the decoding graph directory to find HCLG and words.txt files")
    parser.add_argument("-lang_dir", help="the lexicon directory to find L.fst")
    parser.add_argument("-lr", type=float, help="set the learning rate")
    parser.add_argument("-ce_ratio", default=0.1, type=float, help="the ratio for ce regularization") 
    parser.add_argument("-momentum", default=0, type=float, help="set the momentum") 
    parser.add_argument("-weight_decay", default=1e-4, type=float, help="set the L2 regularization weight") 
    parser.add_argument("-batch_size", default=32, type=int, help="Override the batch size in the config")                         
    parser.add_argument("-data_loader_threads", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("-max_grad_norm", default=5, type=float, help="max_grad_norm for gradient clipping")                     
    parser.add_argument("-sweep_size", default=100, type=float, help="process n hours of data per sweep (default:100)")
    parser.add_argument("-num_epochs", default=1, type=int, help="number of training epochs (default:1)") 
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

    chain_net = NnetAM(lstm, model_config["hidden_size"]*2, 6048)
    chain_net.cuda()

    # setup the optimizer
    optimizer = th.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Broadcast parameters and opterimizer state from rank 0 to all other processes.
    hvd.broadcast_parameters(chain_net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=chain_net.named_parameters())

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
    else:
        sys.stderr.write('ERROR: The model file %s does not exist!\n'%(model_file))
        sys.exit(0)      

    HCLG = args.den_dir + "/HCLG.fst"
    words_txt = args.den_dir + "/words.txt"
    silence_phones = args.den_dir + "/phones/silence.csl"
    ali_model = args.trans_model + "/final.mdl"
    ali_tree = args.trans_model + "/tree"
    L_fst = args.lang_dir + "/L.fst"
    disambig = args.lang_dir + "/phones/disambig.int"

    den_fst = kaldi_fst.StdVectorFst.read("/datadisk2/lial/librispeech/s5/exp/chain_cleaned/tdnn_1d_sp/den.fst")
    chain_model_path = "/datadisk2/lial/librispeech/s5/exp/chain_cleaned/tdnn_1d_sp/0.trans_mdl"
    chain_tree_path = "/datadisk2/lial/librispeech/s5/exp/chain_cleaned/tdnn_1d_sp/tree"

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
    supervision_opts.frame_subsampling_factor = 1
    supervision_opts.left_tolerance = 5
    supervision_opts.right_tolerance = 5

    # chain training options
    chain_opts = kaldi_chain.ChainTrainingOptions()
    chain_opts.leaky_hmm_coefficient = 1e-4
    chain_opts.xent_regularize = 1e-4

    # setup the aligner
    aligner = kaldi_align.MappedAligner.from_files(ali_model, ali_tree, L_fst, None,
                                 disambig, None, 
                                 beam=config["decoder_config"]["align_beam"],
                                 transition_scale=1.0, 
                                 self_loop_scale=0.1, 
                                 acoustic_scale=0.1)
    # compute the log prior
    prior = kaldi_util.io.read_matrix(args.prior_path).numpy()
    log_prior = th.tensor(np.log(prior[0]/np.sum(prior[0])), dtype=th.float)

    den = kaldi_chain.DenominatorGraph(den_fst, 6048)

    chain_net.train()
    for epoch in range(args.num_epochs): 

        run_train_epoch(model, optimizer,
                       log_prior.cuda(), 
                       train_dataloader, 
                       epoch, 
                       chain_trans_model,
                       chain_tree,
                       supervision_opts,
                       silence_ids,
                       chain_net,
                       aligner,
                       den,
                       chain_opts,
                       args)

        # save model
        if hvd.rank() == 0:
            checkpoint={}
            checkpoint['model']=model.state_dict()
            checkpoint['optimizer']=optimizer.state_dict()
            checkpoint['epoch']=epoch
            output_file=args.exp_dir + '/model.se.'+ str(epoch) +'.tar'
            th.save(checkpoint, output_file)

def run_train_epoch(model, optimizer, log_prior, dataloader, epoch, trans_model, tree, supervision_opts, silence_ids, net, aligner, den, chain_opts, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    grad_norm = utils.AverageMeter('grad_norm', ':.4e')
    progress = utils.ProgressMeter(len(dataloader), batch_time, losses, grad_norm, 
                             prefix="Epoch: [{}]".format(epoch))

    ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    if args.criterion == "mmi":
        criterion = ops.MMIFunction.apply
    else:
        criterion = ops.sMBRFunction.apply

    end = time.time()
    for i, batch in enumerate(dataloader):
        feat = batch["x"] 
        label = batch["y"]                               
        num_frs = batch["num_frs"]                       
        utt_ids = batch["utt_ids"]                       
        aux = batch["aux"]  #word labels for se loss
                                       
        x = feat.to(th.float32)                         
        y = label.long()
        x = x.cuda()    
        y = y.cuda()                                
                                                
        prediction_ali = model(x)
        prediction = net(x)
        #ce_loss = ce_criterion(prediction.view(-1, prediction.shape[2]), y.view(-1))
        #loss = args.ce_ratio * ce_loss
 
        for j in range(len(num_frs)):                   
            loglike = prediction_ali[j,:,:]       
            loglike_j = loglike[:num_frs[j],:]        
            loglike_j = loglike_j - log_prior
            text = th.from_numpy(aux[j][0][0].astype(int)).tolist()
            
            align_in = kaldi_matrix.Matrix(loglike_j.detach().cpu().numpy())
            align_out = aligner.align(align_in, text) 
            trans_ids = align_out["alignment"]
            phone_ali = aligner.to_phone_alignment(trans_ids)
            
            phones = list()
            durations = list()
            for item in phone_ali:
                phones.append(item[0])
                durations.append(item[2])
               
            proto_supervision = kaldi_chain.alignment_to_proto_supervision(supervision_opts, phones, durations)
            supervision = kaldi_chain.proto_supervision_to_supervision(tree, trans_model, proto_supervision, True)

            loglike = prediction[j,:,:]
            loglike_j = loglike[:num_frs[j],:]
            loglike_j = kaldi_matrix.Matrix(loglike_j.detach().cpu().numpy())

            nnet_out = kaldi_cudamatrix.CuMatrix().from_matrix(loglike_j)
            grad_out = kaldi_cudamatrix.CuMatrix().from_size(nnet_out.num_rows(), nnet_out.num_cols())
            print(supervision.num_sequences * supervision.frames_per_sequence)
            print(nnet_out.num_rows())
            print(supervision.label_dim, nnet_out.num_cols())

            loss = kaldi_chain.compute_chain_objf_and_deriv(chain_opts, den, supervision, nnet_out, grad_out, None) 
            
            grad_mat = kaldi_matrix.Matrix(nnet_out.num_rows(), nnet_out.num_cols())
            grad_out.copy_to_mat(grad_mat)
            print(loss)
            print(grad_mat.numpy())

                #loss += se_loss.cuda()
           # except:
           #     print("Warning: failed to align utterance {}, skip the utterance for SE loss".format(utt_ids[j]))        

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
