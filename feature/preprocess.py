#!/usr/bin/env python
import os
import numpy as np
import sys
sys.path.append("..")
import torch
import pickle
import utils


def cmn(data, axis=1, is_tensor=False):
    """Apply mean normalization on input sequences for every dimensions.
    axis is the dimension in which we perform normalization. """
    if is_tensor:
        data2 = data - torch.mean(data, dim=axis, keepdim=True)
    else:
        data2 = data - np.mean(data, axis=axis, keepdims=True)
    return data2


def apply_cmn(data, axis=1, stream_keys=None):
    """recursively apply mean normalization on input streams.
    Input is a torch tensor or numpy ndarray, or a list of them.
    For speech data, usually we apply the CMN on the time axis. For example, if the input data has a size of
    NxTxD, where N is the number of sentences, T is the number of frames, and D is the number of features, then set axis=1.
    """
    type_of_input = type(data)
    if type(data) is dict:
        all_keys = list(data.keys())
        normed_data = dict()
    else:
        all_keys = [i for i in range(len(data))]
        normed_data = list()

    for i in all_keys:
        if stream_keys is None or i in stream_keys:
            if type(data[i]) is np.ndarray:
                new_data = cmn(data[i], axis=axis, is_tensor=False)
            elif type(data[i]) is list:
                new_data = apply_cmn(data)      # recursive applying of CMN
            elif torch.is_tensor(data[i]):
                new_data = cmn(data[i], axis=axis, is_tensor=True)
            else:
                new_data = data[i]
        else:
            new_data = data[i]
        if type_of_input is dict:
            normed_data[i] = new_data
        else:
            normed_data.append(new_data)

    if type_of_input is tuple:
        return tuple(normed_data)
    else:
        return normed_data


def feature_normalization(data, use_cmn, transform, stream_keys_for_transform, axis=0):
    if use_cmn:
        data = apply_cmn(data, axis=axis, stream_keys=stream_keys_for_transform)
    if transform is not None:
        data = transform.apply(data, stream_keys=stream_keys_for_transform)
    return data


class GlobalMeanVarianceNormalization:
    def __init__(self, mean_vec=None, std_vec=None, mean_norm=True, var_norm=True):
        self.mean_vec = mean_vec
        self.std_vec = std_vec
        self.mean_vec_tensor = None
        self.std_vec_tensor = None
        self.mean_norm = mean_norm
        self.var_norm = var_norm
        self.mean_stats = None
        self.var_stats = None
        self.n_frame = 0

    def learn_mean_and_variance_from_train_loader(self, train_loader, stream_keys=[], n_sample_to_use=200):
        n_sample_used = 0
        for i, data in enumerate(train_loader, 0):       # trainloader is a iterator. This line extract one minibatch at one time
            print("GlobalMeanVarianceNormalization::learn_mean_and_variance_from_train_loader: accumulate stats from minibatch %d" % i)
            for j in stream_keys:
                if type(data.get(j)) is torch.Tensor:
                    tmp_data = [data.get(j)]
                elif type(data.get(j)) is list:     # sometimes, a stream is a list of tensors
                    tmp_data = data.get(j)
                else:
                    continue
                if i == 0 and self.mean_stats is None:
                    self.initialize_stats(tmp_data[0].shape[2])
                for k in tmp_data:
                    self.accumulate_stats(k.numpy())
                    n_sample_used += k.shape[0]
            if n_sample_used > n_sample_to_use:
                break
        self.learn_mean_and_variance_from_stats()

    def initialize_stats(self, dim):
        self.mean_stats = np.zeros((dim,1), dtype=np.float32)
        self.var_stats = np.zeros((dim, 1), dtype=np.float32)
        self.n_frame = 0

    def accumulate_stats(self, data):
        unpad = utils.padding.Padder.unpad_sequence(data)
        data2 = np.vstack(unpad).T
        #D, N, M = data2.shape
        #data2 = np.reshape(data2, (D, N * M))
        self.mean_stats += np.sum(data2, axis=1, keepdims=True)
        self.var_stats += np.sum(data2**2, axis=1, keepdims=True)
        self.n_frame += data2.shape[1]

    def learn_mean_and_variance_from_stats(self):
        self.mean_vec = self.mean_stats / self.n_frame
        self.std_vec = np.sqrt(self.var_stats / self.n_frame - self.mean_vec**2)
        self.std_vec = np.maximum(self.std_vec, 1e-2)   # avoid very small variances
        self.std_vec[np.isnan(self.std_vec)] = 1.0      # avoid nan variance
        self.std_vec[np.isinf(self.std_vec)] = 1.0      # avoid infinity variance
        
        self.mean_vec = self.mean_vec.astype(np.float32).T      # store as single precision row vectors to avoid data transpose.
        self.std_vec = self.std_vec.astype(np.float32).T

    def learn_mean_and_variance(self, data):
        """given data, a NxMxD tensor, find the mean and variance vec.
        N is the number of training samples.
        M is the number of frames per sample,
        and D is the feature dimension. """
        data2 = data.transpose((2,1,0))
        D,N,M = data2.shape
        data2 = np.reshape(data2, (D, N*M))
        self.mean_vec = np.mean(data2, axis=1, keepdims=True)
        self.std_vec = np.std(data2, axis=1, keepdims=True)
        self.std_vec = np.maximum(self.std_vec, 1e-2)   # avoid very small variances
        self.std_vec[np.isnan(self.std_vec)] = 1.0      # avoid nan variance
        self.std_vec[np.isinf(self.std_vec)] = 1.0      # avoid infinity variance

        self.mean_vec = self.mean_vec.astype(np.float32).T      # store as single precision row vectors to avoid data transpose.
        self.std_vec = self.std_vec.astype(np.float32).T

        # verification
        data_normed = self.apply(data)

    def apply(self, data, stream_keys=None):
        """Apply global mean and varinace normalization to input data, which may be a dictionary, list, or tuple.
           If data is a dict, stream_keys defines the entries of the dictionaries that need to be transformed. Each data
           should be a tensor of size NxMxD, where N is the number of sentences, M is the number of frame, and D is the
           number of feature dimensions.
           If data is a list or tuple, stream_keys should be a list of integers that specify the entries of the list or
           tuple that needs to be transformed.
           If stream_keys is None, apply transform to all the entries in the data.
        """

        type_of_input = type(data)
        if type(data) is dict:
            all_keys = list(data.keys())
            normed_data = dict()
        else:
            all_keys = [i for i in range(len(data))]
            normed_data = list()

        for i in all_keys:
            if stream_keys is None or i in stream_keys:
                if type(data[i]) is np.ndarray:
                    new_data = self.apply_on_ndarray(data[i], is_tensor=False)
                elif type(data[i]) is list:
                    new_data = self.apply(data[i])
                elif torch.is_tensor(data[i]):
                    new_data = self.apply_on_ndarray(data[i], is_tensor=True)
                else:
                    new_data = data[i]
            else:
                new_data = data[i]
            if type_of_input is dict:
                normed_data[i] = new_data
            else:
                normed_data.append(new_data)

        if type_of_input is tuple:
            return tuple(normed_data)
        else:
            return normed_data

    def apply_on_ndarray(self, data, is_tensor=False):
        #mask = utils.padding.Padder.get_mask_from_padded(data)
        if self.mean_norm:
            if is_tensor:
                if self.mean_vec_tensor is None:
                    self.mean_vec_tensor = torch.tensor(self.mean_vec)
                data2 = data - self.mean_vec_tensor
            else:
                data2 = data - self.mean_vec
        else:
            data2 = data

        if self.var_norm:
            if is_tensor:
                if self.std_vec_tensor is None:
                    self.std_vec_tensor = torch.tensor(self.std_vec)
                data2 = data2 / self.std_vec_tensor
            else:
                data2 = data2 / self.std_vec
        #data2 = utils.padding.Padder.encode_data_by_mask(data2, mask)
        return data2
