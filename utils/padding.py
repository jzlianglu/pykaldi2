import numpy as np
import torch


invalid_frame_code = -1e10


class Padder:
    def __init__(self):
        """Provide functionality for padding a specific value to a matrix or tensor.
        This is usually used to deal with variable sequence length in a minibatch.
        The reason for padding is to encode the number of valid frames automatically
        in the sequence itself, so we don't need to provide an extra variable. """
        pass

    @staticmethod
    def pad_sequence(data, target_len, padded_value=invalid_frame_code, time_dim=1):
        """ pad a specified number to a tensor.
        Assume input is a 3D tensor.
        If it is a 2D matrix, first extend it to a 3D tensor. """
        ndim = data.ndim
        if ndim == 2:
            if type(data) is torch.Tensor:
                data = torch.unsqueeze(data, 2)
            else:
                data = np.expand_dims(data, 2)

        old_data_shape = list(data.shape)
        new_data_shape = list(data.shape)
        assert new_data_shape[time_dim] <= target_len
        new_data_shape[time_dim] = target_len

        if type(data) is torch.Tensor:
            new_data = torch.ones(new_data_shape, dtype=data.dtype, device=data.device) * padded_value
        else:
            new_data = np.ones(new_data_shape, dtype=data.dtype) * padded_value

        if time_dim == 0:
            new_data[:old_data_shape[0],:,:] = data
        elif time_dim == 1:
            new_data[:, :old_data_shape[1],:] = data
        elif time_dim == 2:
            new_data[:, :, :old_data_shape[2]] = data

        if ndim == 2:
            new_data =new_data[:,:,0]

        return new_data

    @staticmethod
    def encode_data_by_mask(data, mask):
        if np.sum(mask)==0:
            return data

        ndim = data.ndim
        if ndim == 2:
            data2 = np.expand_dims(data, 0)

        last_valid_frame_idx = Padder.get_last_valid_frame_index(mask)
        for i in range(mask.shape[0]):
            data2[i, int(last_valid_frame_idx+1):, 1] = invalid_frame_code

        if ndim==2:
            data2 = data2[0]
        return data2

    @staticmethod
    def get_mask_from_padded(data, code=invalid_frame_code):
        if type(data) is torch.Tensor:
            if len(data.shape)==2:
                data = data.unsqueeze(0)
        elif data.ndim == 2:
            data = np.expand_dims(data, 0)

        mask = data[:, :, 1] == code
        if type(mask) == np.ndarray:
            mask = mask.astype(np.uint8)
            
        return mask

    @staticmethod
    def get_valid_len_from_padded(data, code=invalid_frame_code):
        """Assume input is a 3D tensor of MxTxD.
        M is the number of samples. 
        T is the number of frames. 
        D is the number of dimensions in each feature vector."""
        mask = Padder.get_mask_from_padded(data, code=code)
        last_valid_frame_idx = Padder.get_last_valid_frame_index(mask)

        return last_valid_frame_idx

    @staticmethod
    def get_last_valid_frame_index(mask):
        """Mask is a 2D matrix of MxT, where M is the number of sampples, and T is the number of frames"""
        T = mask.shape[1]
        if np.sum(mask) == 0:
            '''The most common case is that all sequences are of full length. '''
            last_idx = np.ones((mask.shape[0],1)) * T
        else:
            delta = mask[:, 1:] - mask[:, 0:-1]
            last_idx = np.argmax(delta, axis=1)
            for i in range(mask.shape[0]):
                if mask[i, last_idx[i]+1] == 0:
                    last_idx[i] = T-1

        return last_idx

    @staticmethod
    def unpad_sequence(data):
        if data.ndim==2:
            data = np.expand_dims(data, 0)

        last_valid_frame = Padder.get_valid_len_from_padded(data)
        
        unpad = list()
        for i in range(data.shape[0]):
            unpad.append(data[i,:int(last_valid_frame[i]+1),:])

        return unpad