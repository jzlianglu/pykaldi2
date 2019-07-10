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

import numpy as np
import operator
import torch

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler

class ChunkDataloader(DataLoader):

    def __init__(self, dataset, batch_size, num_workers=0, timeout=1000):
        self.dataset = dataset

        super(ChunkDataloader, self).__init__(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              collate_fn=self.collate_fn,
                                              timeout=timeout)

    def collate_fn(self, batch):
        feats, utt_ids, labels = zip(*batch)
        data = {
            "utt_ids": utt_ids,
                "x": torch.FloatTensor(list(feats)),
                "y": torch.LongTensor(list(labels))
        }

        return data

      
class SeqDataloader(DataLoader):
    
    def __init__(self, dataset, batch_size, num_workers=0, distributed=False, test_only=False, timeout=1000):
        
        self.test_only = test_only
        self.dataset = dataset
        self.distributed = distributed
 
        # now decide on a sampler
        #base_sampler = torch.utils.data.SequentialSampler(self.dataset)
        base_sampler = torch.utils.data.RandomSampler(self.dataset)
        
        if not self.distributed:
            sampler = torch.utils.data.BatchSampler(base_sampler, batch_size, False)
            super(SeqDataloader, self).__init__(dataset,
                                           batch_sampler=sampler,
                                           num_workers=num_workers,
                                           collate_fn=self.collate_fn)
        else:
            import horovod.torch as hvd
            sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
            super(SeqDataloader, self).__init__(dataset,
                                           batch_size=batch_size, 
                                           sampler=sampler, 
                                           num_workers=num_workers, 
                                           collate_fn=self.collate_fn, 
                                           drop_last=False,
                                           timeout=timeout)
   

    def collate_fn(self, batch):
        
        def pad_and_concat_feats(inputs):
            max_t = max(inp.shape[0] for inp in inputs)
            num_frs = [inp.shape[0] for inp in inputs]
            shape = (len(inputs), max_t, inputs[0].shape[1])
            input_mat = np.zeros(shape, dtype=np.float32)
            for e, inp in enumerate(inputs):
                input_mat[e, :inp.shape[0], :] = inp
            return input_mat, num_frs
       
        def pad_and_concat_labels(labels):
            max_t = max(l.shape[0] for l in labels)
            num_frs = [l.shape[0] for l in labels]
            shape = (len(labels), max_t, labels[0].shape[1])
            out_label = np.full(shape, -100, dtype=np.int)
            for e, l in enumerate(labels):
                out_label[e, :l.shape[0], :] = l
            return out_label, num_frs

        if self.test_only:
            feats, utt_ids = zip(*batch)
            feats, num_frs = pad_and_concat_feats(feats)
            data = {
                "utt_ids" : utt_ids,
                "num_frs": num_frs,
                "x" : torch.from_numpy(feats)
            }

        else:
            feats, utt_ids, labels, aux = zip(*batch)
            labels, num_labs = pad_and_concat_labels(labels)
            feats, num_frs = pad_and_concat_feats(feats)
            assert num_labs == num_frs, "The numbers of frames and labels are not equal"
            data = {
                "utt_ids" : utt_ids,
                "num_frs": num_frs,
                "x" : torch.from_numpy(feats),
                "y" : torch.from_numpy(labels),
                "aux": aux
            }

        return data 
