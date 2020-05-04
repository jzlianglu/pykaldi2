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

from kaldi.util.table import MatrixWriter
from kaldi.util.io import read_matrix

from data import SpeechDataset, SeqDataloader
from models import LSTMStack, NnetAM

def main():
#if __name__ == '__main__':
    parser = argparse.ArgumentParser()                                                                                 
    parser.add_argument("-config")                                                                                     
    parser.add_argument("-model_path")                                                                                 
    parser.add_argument("-data_path")
    parser.add_argument("-prior_path", help="the path to load the final.occs file")
    parser.add_argument("-transform", help="feature transformation matrix or mvn statistics")
    parser.add_argument("-out_file", help="write out the log-probs to this file") 
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
    print(transform)
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

    device = th.device("cuda:1" if th.cuda.is_available() else "cpu")
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
 
    prior = read_matrix(args.prior_path).numpy()
    log_prior = th.tensor(np.log(prior[0]/np.sum(prior[0])), dtype=th.float)

    model.eval()
    with th.no_grad():
        with MatrixWriter("ark:"+args.out_file) as llout:
            for i, data in enumerate(test_dataloader):
                feat = data["x"]
                num_frs = data["num_frs"]
                utt_ids = data["utt_ids"]
 
                x = feat.to(th.float32)
                x = x.cuda()
                prediction = model(x)
                # save only unpadded part for each utt in batch                                         
                for j in range(len(num_frs)):                                                            
                    loglikes=prediction[j,:,:].data.cpu()                                                      
                    loglikes_j = loglikes[:num_frs[j],:]                                                 
                    loglikes_j = loglikes_j - log_prior                                                   
                                                                                         
                    llout[utt_ids[j][0]] = loglikes_j

                print("Process batch [{}/{}]".format(i+1, len(test_dataloader)))
            
if __name__ == '__main__':
    main()
