import numpy as np
import scipy
from time import gmtime, strftime
import re
import os

"""
  AverageMeter and ProgressMeter are borrowed from the PyTorch Imagenet example:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

'''
def from_pickle(file):
    import pickle
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def dump2pickle(file, data):
    import pickle
    with open(file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
'''

def remove_from_list_by_index(full_list, unwanted_list):
    # remove items from full_list using index in unwanted_list
    if 1:
        for ii in sorted(unwanted_list, reverse=True):
            del full_list[ii]
    elif 0:
        new_list = [curr_entry for ii, curr_entry in enumerate(full_list) if ii not in unwanted_list]
        full_list = new_list
    elif 0:
        new_list = np.delete(full_list, unwanted_list).tolist()
        full_list = new_list

def my_cat(file_path):
    # read text lines from a text file, like the cat in unix
    with open(file_path) as file:
        lines = [line.rstrip('\n') for line in file]
    return lines

'''
def my_dump(file_path, lines):
    # read text lines from a text file, like the cat in unix
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line+"\n")


def print_args(args):
    options = vars(args)
    keys = list(options.keys())
    keys.sort()
    for x in keys:
        print("    %s : %s" % (x, options[x]))

def str2int_list(x):
    result = [int(i) for i in str2float_list(x)]
    return result


def str2float_list(x):
    if x[0]=='[':
        x = x[1:]
    if x[-1]==']':
        x=x[:-1]

    result = np.fromstring(x, sep=',')
    result_list = [result[i] for i in range(result.size)]

    return result_list


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

'''

def fftconvolve1d(in1, in2, use_gpu=False):
    """1D convolution along the first dimension"""
    m,n1 = in1.shape
    k,n2 = in2.shape
    rlen = m + k - 1
    rlen_p2 = scipy.fftpack.helper.next_fast_len(int(rlen))

    XX = np.fft.rfft(in1, rlen_p2, axis=0)
    YY = np.fft.rfft(in2, rlen_p2, axis=0)
    ret = np.fft.irfft(XX * YY, rlen_p2, axis=0)

    return ret[:rlen,:]

'''
def tic():
    import time
    return time.time()


def toc(start_time):
    import time
    print("Elapsed time is %s seconds." % (str(time.time() - start_time)) )
'''

def convert_data_precision(data, precision):
    data_type = type(data)

    if data_type == dict:
        return data     # do not convert dictionary
    elif data_type==list:
        data_list = data
    elif data_type==tuple:
        data_list = list(data)
    else:
        data_list = [data]

    converted = []
    for seg in data_list:
        if type(seg) == dict:
            converted.append(seg)
        elif type(seg) == list:
            converted.append(convert_data_precision(seg, precision))
        elif precision == seg.dtype:
            converted.append(seg)
        elif precision == "float32" or precision == "float64":
            converted.append(seg.astype(precision))
        elif precision == "int16":
            seg2 = seg / np.max(np.abs(seg)) * 2 ** 15
            converted.append(seg2.astype("int16"))

    if data_type==list:
        return converted
    elif data_type==tuple:
        return tuple(converted)
    else:
        return converted[0]


def get_distribution_template(comment, max=3.0, min=1.0, mean=None, std=None, category=None, pmf=None,
                              distribution='uniform'):
    assert(max>=min)
    template = {'comment': comment + ', mean/std needed if distribution=gaussian', 'max': max, 'min': min, 'mean': mean,
                'std': std, 'category': category, 'pmf': pmf, 'distribution': distribution}
    return template


def get_sample(config, n_sample=1):

    if config['distribution'] == 'binary':
        data = np.random.choice([0, 1], size=n_sample, replace=True, p=config['pmf'])

    elif config['distribution'] == 'discrete':
        data = np.random.choice(config['category'], size=n_sample, replace=True, p=config['pmf'])

    elif config['distribution'] == 'uniform':
        assert float(config['min']) < float(config['max'])
        #print("min = %f, max = %f, n_sample = %d " % (float(config['min']), float(config['max']), n_sample))
        data=np.random.uniform(low=float(config['min']),high=float(config['max']),size=n_sample)

    elif config['distribution'] == 'gaussian':
        data=np.random.normal(loc=float(config['mean']),scale=float(config['std']),size=n_sample)
        data = np.maximum(data, float(config['min']))
        data = np.minimum(data, float(config['max']))

    elif config['distribution'] == 'uniform_int':
        if int(config['min'])==int(config['max']):
            data=int(config['min'])*np.ones((n_sample,),dtype='int32')
        else:
            data=np.random.randint(int(config['min']),high=int(config['max']),size=n_sample)

    else:
        print('Warning: unknown distribution type: %s' % config['distribution'])
        data = []

    return data


def comp_snr(signal, noise):
    Px = np.mean(signal ** 2)
    Pn = np.mean(noise ** 2)
    SNR = 10 * np.log10(Px/Pn)
    return SNR


def comp_noise_scale_given_snr(signal, noise, snr):
    Px = np.mean(signal ** 2)
    Pn = np.mean(noise ** 2)
    scale = np.sqrt(Px/Pn*10 ** ((-snr)/10))
    return scale

'''
def vec_norm(vec):
    return np.sqrt(np.sum(vec**2, axis=0, keepdims=True))


def cosine_sim(a,b):
    if a.ndim == 1:
        a = np.reshape(a, (a.size, 1))
    if b.ndim == 1:
        b = np.reshape(b, (b.size, 1))
    sim = a.T @ b
    a_norm = vec_norm(a)
    b_norm = vec_norm(b)
    denom = a_norm.T @ b_norm
    sim /= denom
    return sim

'''

def euclidean_distance(A, B):
    """
    compute euclidean distance between points in A and B. Both A and B are matrix of shape DxN. D is the dimension.
    N can be different between A and B
    """
    if A.ndim>2 or B.ndim>2:
        raise Exception("Cannot handle tensor")

    if A.ndim == 0 or B.ndim == 0:   # if one input is scalar
        return np.abs(A-B)

    if A.ndim == 1 and B.ndim == 1:
        return scipy.spatial.distance.euclidean(A,B)

    if A.ndim == 1 and B.ndim == 2:     # if input is 1D array, force it to be 2D matrix
        A.reshape(A.size,1)
        return np.sqrt( np.sum((A-B) ** 2, axis=0) )

    if B.ndim == 1 and A.ndim == 2:
        B.reshape(B.size,1)
        return np.sqrt(np.sum((A - B) ** 2, axis=0))

    # if both A and B are matrix
    A2 = A.reshape(A.shape[0], 1, A.shape[1])
    B2 = B.reshape(B.shape[0], B.shape[1], 1)
    return np.sqrt(np.sum((A2-B2)**2, axis=0))


def imagesc(data, show_color_bar=False, title=None, new_figure=False, colormap='jet'):
    import matplotlib.pyplot as plt
    import torch
    if type(data) == torch.Tensor:
        data = data.to('cpu').data.numpy()
    if new_figure:
        plt.figure()
    plt.imshow(data, aspect='auto')
    plt.set_cmap(colormap)
    if show_color_bar:
        plt.colorbar()
    if title is not None:
        plt.title(title)

'''
def get_time():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def format_float4name(x):
    if np.abs(x) - np.floor(np.abs(x)) == 0:
        s = str(x)
    else:
        s = '%2.2E' % x
        result = re.search('E', s)
        idx = result.span()
        digit = float(s[:idx[0]])
        if np.abs(digit) - np.floor(np.abs(digit)) == 0:
            digit = int(digit)
        exp_term = int(np.round(float(s[idx[0]+1:])))
        if exp_term == 0:
            s = str(digit)
        else:
            s = str(digit)+'E'+str(exp_term)
    return s


def get_date():
    return strftime("%Y-%m-%d", gmtime())
'''

def print_progress(finished, n_task, step=1, tag=None):
    if finished % step == 0:
        if tag is None:
            print('  %d out of %d tasks (%2.1f%%) finished - %s' % (finished, n_task, finished / n_task * 100, get_time()))
        else:
            print('  %d out of %d tasks (%2.1f%%) finished - %s - %s' % (finished, n_task, finished / n_task * 100, get_time(), tag))


def set_config(config_dict, key, value, overwrite=True):
    """If overwrite is False, and the key is already set in config_dict, keep the original value."""
    if overwrite is False and key in config_dict.keys() and config_dict[key] is not None:
        return
    config_dict[key] = value

def utt2seg(data, seg_len, seg_shift):
    """ Cut an utterance (MxN matrix) to segments. """
    if data.ndim==1:
        data = np.reshape(data, (1, data.size))
    dim,n_fr = data.shape
    n_seg = int(np.floor( (n_fr-seg_len) / seg_shift )) + 1
    seg = []
    for i in range(n_seg):
        start = i*seg_shift
        stop = start+seg_len
        seg.append(data[:, start:stop])

    return seg


# sample a segment of given length from the input data
# data is a M x N matrix, where M is the feature dimension, and N is the number of time steps
# data2 is of the same size as data. If provided, it will be sampled using the same starting point as data
def sample_segment(data, required_length, zero_padding=True, data2=None):
    required_length = int(required_length)
    n_dim, n_frame = data.shape
    if data2 is not None:
        data2_is_list = True
        if type(data2) != list:
            data2 = [data2]
            data2_is_list = False
        n_dim2 = [i.shape[0] for i in data2]
        n_frame2 = [i.shape[1] for i in data2]
        assert all([n_frame==i for i in n_frame2]), "data and data2 have different lengths!"

    if n_frame <= required_length:
        #print("Warning: data length is shorter than required length\n")
        if zero_padding:
            out_data = np.zeros((n_dim, required_length), data.dtype)
            out_data[:,:n_frame] = data
        else:
            out_data = data
        if data2 is not None:
            if zero_padding:
                out_data2 = []
                for i in range(len(data2)):
                    out_data2.append( np.zeros((n_dim2[i], required_length), data2[i].dtype) )
                    out_data2[i][:, :n_frame] = data2[i]
            else:
                out_data2 = data2
    else:
        # randomly sample an initial point
        start = np.random.randint(0, high=n_frame-required_length, size=1)
        out_data = data[:,start[0]:start[0]+required_length]
        if data2 is not None:
            out_data2 = []
            for i in data2:
                out_data2.append(i[:, start[0]:start[0] + required_length])

    if data2 is not None:
        if data2_is_list:
            return out_data, out_data2
        else:
            return out_data, out_data2[0]
    else:
        return out_data


def moving_average_1d(data, filter_len):
    if filter_len<=1:
        return data
    filter = np.ones((filter_len,)) / filter_len
    return np.convolve(np.squeeze(data), filter, mode='valid')


def label2seg(label):
    # number of state change equals to
    diff = label[1:] - label[:-1]
    idx = np.where(diff!=0)[0]
    n_seg = idx.size
    segments = []
    if n_seg == 0:
        segments.append((1, label.size, label[0]))
        return segments

    for i in range(n_seg):
        if i == 0:
            start = 1
        else:
            start = idx[i - 1] + 1
        stop = idx[i]
        curr_label = label[idx[i]]
        segments.append((start, stop, curr_label))

    segments.append((idx[-1]+1, label.size, label[-1]))

    return segments


def file2utt_id(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]

