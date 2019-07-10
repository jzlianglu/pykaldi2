import numpy as np
import scipy as sp
import scipy.signal
import scipy.fftpack
import six
import os


_use_stored_window=True
if _use_stored_window: # load the pre-computed window coefficients for 80D log filterbanks used in typical acoustic modeling.
    mel_file = os.path.join(os.path.dirname(__file__), 'mel80_window.txt')
    with open(mel_file) as file:
        lines = [line.rstrip('\n') for line in file]
    mel80_window = np.vstack([np.asarray([np.float32(j) for j in i.split(",")]) for i in lines])
else:   # you can generate new window if you want different configurations to the window.
    import librosa
    mel80_window = librosa.filters.mel(16000, 512, n_mels=80, fmax=7690, htk=True)


def get_window(window, wlen):
    if type(window) == str:
        # types:
        # boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen,
        # bohman, blackmanharris, nuttall, barthann
        if window == 'hamming':
            fft_window = np.hamming(wlen)
        elif window == 'bartlett':
            fft_window = np.bartlett(wlen)
        elif window == 'hann' or window == 'hanning':
            fft_window = np.hanning(wlen)
        else:
            #try:
                # scipy.signal.get_window gives non-symmetric results for hamming with even length :(
                #fft_window = scipy.signal.get_window(window, wlen)
            #except:
                #raise Exception('cannot obtain window type {}'.format(window))
            raise Exception('cannot obtain window type {}'.format(window))
        # fft_window = scipy.signal.hamming(win_length, sym=False)
    elif six.callable(window):
        # User supplied a windowing function
        fft_window = window(wlen)
    else:
        # User supplied a window vector.
        # Make it into an array
        fft_window = np.asarray(window)
        assert(len(fft_window) == wlen)

    fft_window.flatten()
    return fft_window


def _enframe(x, shift, length, axis_t=-1, newaxis_t=-1, newaxis_b=-2,
             end='pad', pad_mode='constant', pad_value=0, copy=True):
    axis_t = axis_t % x.ndim
    newaxis_t = newaxis_t % (x.ndim + 1)
    newaxis_b = newaxis_b % (x.ndim + 1)

    if not newaxis_b == newaxis_t - 1:
        transpose_out_flag = True
        tout = list(range(0, x.ndim - 1))  # two less than outdim
        if newaxis_b < newaxis_t:
            tout.insert(newaxis_b, x.ndim - 1)
            tout.insert(newaxis_t, x.ndim)
        else:
            tout.insert(newaxis_t, x.ndim)
            tout.insert(newaxis_b, x.ndim - 1)
        newaxis_b = x.ndim - 1
        newaxis_t = x.ndim
    else:
        transpose_out_flag = False

    # from now on, we rely on this
    assert (newaxis_b == newaxis_t - 1)

    if end == 'pad':
        if (x.shape[axis_t] + shift - length) % shift != 0:
            npad = np.zeros([x.ndim, 2], dtype=np.int)
            npad[axis_t, 1] = shift - ((x.shape[axis_t] + shift - length) % shift)
            x = np.pad(x, pad_width=npad, mode=pad_mode,
                       constant_values=pad_value)
    elif end is None:
        assert (x.shape[axis_t] + shift - length) % shift == 0, \
            '{} = x.shape[axis]({}) + shift({}) - length({})) % shift({})' \
            ''.format((x.shape[axis_t] + shift - length) % shift,
                      x.shape[axis_t], shift, length, shift)
    elif end == 'cut':
        pass
    else:
        raise ValueError(end)

    newshape = list(x.shape)
    del newshape[axis_t]
    newshape.insert(newaxis_b, (x.shape[axis_t] + shift - length) // shift)
    newshape.insert(newaxis_t, length)

    newstrides = list(x.strides)
    del newstrides[axis_t]
    newstrides.insert(newaxis_b, int(shift * x.strides[axis_t]))
    newstrides.insert(newaxis_t, int(x.strides[axis_t]))

    # Alternative to np.ndarray.__new__
    # I am not sure if np.lib.stride_tricks.as_strided is better.
    # return np.lib.stride_tricks.as_strided(
    #     x, shape=shape, strides=strides)
    try:
        if copy == True:
            y = x.copy(order='K')
        else:
            y = x.view()
        # return np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)
        # return xp.ndarray.__new__(xp.ndarray, strides=strides,
        #
        y = np.lib.stride_tricks.as_strided(y, strides=newstrides, shape=newshape)
        if transpose_out_flag:
            y = y.transpose(tuple(tout))
        return y
    except Exception:
        print('strides:', x.strides, ' -> ', newstrides)
        print('shape:', x.shape, ' -> ', newshape)
        print('flags:', x.flags)
        raise


def stft(y, n_fft=2048, hop_length=None, win_length=None, window=None, center=False, do_dither=False,
         dtype=np.complex64):
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 3)

    if window is None:
        window = 'hamming'

    if center:
        assert y.ndim == 1
        y = np.pad(y, win_length-hop_length, mode='constant')

    fft_window = get_window(window, win_length)

    nrfft = int(n_fft // 2) + 1

    if do_dither:
        y = y + np.random.normal(loc=0.0, scale=1e-5, size=y.shape)

    y_frames = _enframe(y, hop_length, win_length, axis_t=0, newaxis_t=1, newaxis_b=0, copy=True)
    # RFFT and Conjugate here to match phase from DPWE code
    #stft_matrix = fft.fft(fft_window.reshape(1,win_length) * y_frames, n=n_fft, axis=1)[:, :nrfft].astype(dtype=dtype)
    stft_matrix = np.fft.fft(fft_window.reshape(1,win_length) * y_frames, n=n_fft, axis=1)[:, :nrfft].astype(dtype=dtype)
    return stft_matrix


def istft(stft_matrix, hop_length=None, win_length=None, window=None, center=False, dtype=np.float32,
          method='vectorized', output_len=None, small_float=1e-10):
    '''
    >>> x=np.arange(10)
    >>> y=np.fft.fft(x, 16)
    >>> np.fft.ifft(y).real[:10]
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    >>> sig_len=10
    >>> y=stft(np.arange(sig_len).astype(dtype=np.float32), n_fft=8, hop_length=3, win_length=5, center=True)
    >>> istft(y, hop_length=3, win_length=5, center=True, output_len=sig_len, method='seq')
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.], dtype=float32)
    >>> istft(y, hop_length=3, win_length=5, center=True, output_len=sig_len, method='vectorized')
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.], dtype=float32)

    :param stft_matrix:
    :param hop_length:
    :param win_length:
    :param window:
    :param center:
    :param dtype:
    :param method:
    :return:
    '''
    n_fft = 2 * (stft_matrix.shape[1] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 3)

    if window is None:
        window = 'hamming'

    fft_window = get_window(window, win_length)

    if fft_window.size > n_fft:
        raise Exception('Size mismatch between n_fft={} and window size={}'.format(n_fft, fft_window.size))

    n_frames = stft_matrix.shape[0]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)
    fft_window_sum = np.zeros(expected_signal_len, dtype=dtype)

    if method == 'vectorized':  # vectorized should be faster, may use more memory
        full_stft = np.concatenate((stft_matrix, stft_matrix.conj()[:, -2:0:-1]), axis=1)
        time_indices = np.arange(n_fft).reshape(1, n_fft) + hop_length * np.arange(n_frames).reshape(n_frames,1)
        #frame_signals = fft.ifft(full_stft).real
        frame_signals = np.fft.ifft(full_stft).real
        fft_windows = np.zeros(time_indices.shape)
        fft_windows[:, :fft_window.shape[0]] = fft_window.reshape(1, fft_window.shape[0])

        np.add.at(y, time_indices.flatten(), frame_signals.flatten())
        np.add.at(fft_window_sum, time_indices.flatten(), fft_windows.flatten())
    else:
        for i in range(n_frames):
            sample = i * hop_length
            spec = stft_matrix[i, :].flatten()
            cspec = spec.conj()
            spec = np.concatenate((spec, cspec[-2:0:-1]), axis=0)
            #ytmp = fft.ifft(spec).real
            ytmp = np.fft.ifft(spec).real

            y[sample:(sample + n_fft)] = y[sample:(sample + n_fft)] + ytmp
            fft_window_sum[sample:(sample + fft_window.size)] += fft_window

    # Normalize the amplitude modulation caused by windowing
    approx_nonzero_indices = fft_window_sum > small_float
    y[approx_nonzero_indices] /= fft_window_sum[approx_nonzero_indices]

    if center:
        padlen = win_length - hop_length
        y = y[padlen:-padlen]
    if output_len is not None:
        y = y[:output_len]
    return y


# maps a tensor of phases to values between -pi and pi
def map_to_primal(x):
    mults = np.round(x / (2 * np.pi))
    x -= mults * 2.0 * np.pi
    return x


class SpectrumAnalyzer:
    def __init__(self, config=None, fs=16000, fft_size=512, frame_len=400, frame_shift=160, window='hamming', do_dither=True, dc_removal=False, use_gpu=False):
        self.fs = fs
        self.fft_size = fft_size
        self.frame_len = frame_len
        self.frame_shift = frame_shift
        self.window = window
        self.do_dither = do_dither
        self.dc_removal = dc_removal
        self.use_gpu = use_gpu

        if config is not None:
            for attr in config:
                setattr(self, attr, config[attr])

        self.n_bin = self.fft_size/2+1
        self.frame_overlap = self.frame_len - self.frame_shift

    def analyze(self, signal):
        stft_coef = stft(signal, win_length=self.frame_len, hop_length=self.frame_shift, n_fft=self.fft_size,
                         window=self.window, center=False, do_dither=self.do_dither)

        return stft_coef.T

    def log_spec(self, signal):
        stft_coef = self.analyze(signal)
        return np.log(np.abs(stft_coef))

    def synthesize(self, stft_matrix, center=True, dtype=np.float32):
        y = istft(stft_matrix.T, hop_length=self.frame_shift, win_length=self.frame_len, window=self.window, center=False)

        return y


def logfbank80(wav):
    # extract the 80D filterbank features from 16kHz speech signal that is popular for acoustic modeling.

    preemphasis = 0.96
    global mel80_window

    t1 = np.sum(mel80_window, 0)
    t1[t1 == 0] = -1
    inv = np.diag(1 / t1)
    mel = mel80_window.dot(inv).T

    wav = wav[1:] - preemphasis * wav[:-1]
    S = stft(wav, n_fft=512, hop_length=160, win_length=400, window=np.hamming(400), center=False).T

    spec_mag = np.abs(S)
    spec_power = spec_mag ** 2
    fbank_power = spec_power.T.dot(mel * 32768 ** 2) + 1
    log_fbank = np.log(fbank_power)

    # Debug code
    '''import matplotlib.pyplot as plt
    import feature_io
    feat_from_feconvert = feature_io.HTKFeat_read(mfcfile).getall()
    plt.figure()
    plt.imshow(np.vstack([S4[:500, :].T, feat_from_feconvert[:500, :].T, feat_from_feconvert[:500, :].T - S4[:500, :].T]))
    '''

    return log_fbank
