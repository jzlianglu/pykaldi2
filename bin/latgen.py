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

from kaldi.util.table import MatrixWriter
from kaldi.util.io import read_matrix
import kaldi.util as kaldi_util
import kaldi.hmm as kaldi_hmm                         
import kaldi.matrix as kaldi_matrix                   
import kaldi.lat as kaldi_lat                         
import kaldi.decoder as kaldi_decoder                 
from kaldi.asr import MappedLatticeFasterRecognizer   
from kaldi.decoder import LatticeFasterDecoderOptions 

from data import SpeechDataset, SeqDataloader
from models import LSTMStack, NnetAM

def main():
    parser = argparse.ArgumentParser()                                                                                 
    parser.add_argument("-config")
    parser.add_argument("-model_path")
    parser.add_argument("-data_path")
    parser.add_argument("-prior_path", help="the path to load the final.occs file")
    parser.add_argument("-out_file", help="write out the log-probs to this file")
    parser.add_argument("-transform", help="feature transformation matrix or mvn statistics")
    parser.add_argument("-trans_model", help="the HMM transistion model, used for lattice generation")
    parser.add_argument("-graph_dir", help="the decoding graph directory") 
    parser.add_argument("-batch_size", default=32, type=int, help="Override the batch size in the config")             
    parser.add_argument("-sweep_size", default=200, type=float, help="process n hours of data per sweep (default:60)")            
    parser.add_argument("-data_loader_threads", default=4, type=int, help="number of workers for data loading")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["sweep_size"] = args.sweep_size

    config["source_paths"] = list()
    data_config = dict()

    data_config["type"] = "Eval"
    data_config["wav"] = args.data_path

    config["source_paths"].append(data_config)

    print("job starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))

    transform=None
    if args.transform is not None and os.path.isfile(args.transform):
        with open(args.transform, 'rb') as f:
            transform = pickle.load(f)

    dataset = SpeechDataset(config)
    #data = trainset.__getitem__(0)
    test_dataloader = SeqDataloader(dataset,
                                    batch_size=args.batch_size,
                                    test_only=True,
                                    global_mvn=True,
                                    transform=transform)

    print("Data loader set up successfully!")
    print("Number of minibatches: {}".format(len(test_dataloader)))

    # ceate model
    model_config = config["model_config"]
    lstm = LSTMStack(model_config["feat_dim"], model_config["hidden_size"], model_config["num_layers"], model_config["dropout"], True)
    model = NnetAM(lstm, model_config["hidden_size"]*2, model_config["label_size"])

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.cuda()

    assert os.path.isfile(args.model_path), "ERROR: model file {} does not exit!".format(args.model_path)

    checkpoint = th.load(args.model_path, map_location='cuda:0')                                            
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
    print("=> loaded checkpoint '{}' ".format(args.model_path))                      

    HCLG = args.graph_dir + "/HCLG.fst"                                                            
    words_txt = args.graph_dir + "/words.txt"                                                      
                                                                                             
    if not os.path.isfile(HCLG):                                                                 
        sys.stderr.write('ERROR: The HCLG file %s does not exist!\n'%(HCLG))                     
        sys.exit(0)                                                                              
                                                                                             
    if not os.path.isfile(words_txt):                                                            
        sys.stderr.write('ERROR: The words.txt file %s does not exist!\n'%(words_txt))           
        sys.exit(0)                                                                              
                                                                                             
    if os.path.isfile(args.trans_model):                                                         
       trans_model = kaldi_hmm.TransitionModel()                                                 
       with kaldi_util.io.xopen(args.trans_model) as ki:                                         
           trans_model.read(ki.stream(), ki.binary)                                              
    else:                                                                                        
       sys.stderr.write('ERROR: The trans_model %s does not exist!\n'%(args.trans_model))        
       sys.exit(0)                                                                    
 
    prior = read_matrix(args.prior_path).numpy()
    log_prior = th.tensor(np.log(prior[0]/np.sum(prior[0])), dtype=th.float)

    # now we can setup the decoder
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = config["decoder_config"]["beam"]
    decoder_opts.lattice_beam = config["decoder_config"]["lattice_beam"]
    decoder_opts.max_active = config["decoder_config"]["max_active"]
    acoustic_scale = config["decoder_config"]["acoustic_scale"]
    decoder_opts.determinize_lattice = True  #To produce compact lattice
    asr_decoder = MappedLatticeFasterRecognizer.from_files(
        args.trans_model, HCLG, words_txt,
        acoustic_scale=acoustic_scale, decoder_opts=decoder_opts)

    model.eval()
    with th.no_grad():
        with kaldi_util.table.CompactLatticeWriter("ark:"+args.out_file) as lat_out:
            for data in test_dataloader:
                feat = data["x"]
                num_frs = data["num_frs"]
                utt_ids = data["utt_ids"]
 
                x = feat.to(th.float32)
                x = x.cuda()

                prediction = model(x)
            
                for j in range(len(num_frs)):                                                            
                    loglikes=prediction[j,:,:].data.cpu()                                                      
                                                                                         
                    loglikes_j = loglikes[:num_frs[j],:]                                                 
                    loglikes_j = loglikes_j - log_prior                                                   
            
                    decoder_out = asr_decoder.decode(kaldi_matrix.Matrix(loglikes_j.numpy()))

                    key = utt_ids[j][0]
                    print(key, decoder_out["text"])
 
                    print("Log-like per-frame for utterance {} is {}".format(key, decoder_out["likelihood"]/num_frs[j]))

                    # save lattice
                    lat_out[key] = decoder_out["lattice"]


if __name__ == '__main__':
    main()
