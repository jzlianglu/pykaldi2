import numpy as np
from ._sampling import get_distribution_template, get_sample
import scipy.fftpack


def _fftconvolve1d(in1, in2):
    """1D convolution along the first dimension"""
    m, n1 = in1.shape
    k, n2 = in2.shape
    rlen = m + k - 1
    rlen_p2 = scipy.fftpack.helper.next_fast_len(int(rlen))

    XX = np.fft.rfft(in1, rlen_p2, axis=0)
    YY = np.fft.rfft(in2, rlen_p2, axis=0)
    ret = np.fft.irfft(XX * YY, rlen_p2, axis=0)

    # the use of rfft seems to be faster than the rfftn used in scipy.signal.fftconvolve on CPU and numpy 1.16.2. Not
    # sure about the reason.
    #sp1 = np.fft.rfftn(in1, [rlen_p2,1], axes=(0,1))
    #sp2 = np.fft.rfftn(in2, [rlen_p2,1], axes=(0,1))
    #ret2 = np.fft.irfftn(sp1 * sp2, [rlen_p2,1], axes=(0,1))

    return ret[:rlen, :]


def _comp_noise_scale_given_snr(signal, noise, snr):
    Px = np.mean(signal ** 2)
    Pn = np.mean(noise ** 2)
    scale = np.sqrt(Px / Pn * 10 ** ((-snr) / 10))
    return scale


class _NoiseSampler:
    def __init__(self):
        pass

    @staticmethod
    def sample_noise(noise, required_length):
        """Sample noise start point.
        If noise is longer than required length, sample a sub-segment of noise whose length is the same as the required length.
        If noise is shorter than required length, sample a starting point for noise. """
        required_length = int(required_length)
        n_sample, n_ch = noise.shape

        # repeat noise if necessary
        if n_sample <= required_length:
            n_extra = required_length - n_sample
            sampler_config = get_distribution_template('none', max=n_extra, min=0, mean=None, std=None, distribution='uniform_int')
            start = get_sample(sampler_config)[0]
            noise_sampled = np.zeros((required_length,n_ch))
            noise_sampled[start:start+n_sample,:] = noise

        else:
            n_extra = n_sample - required_length
            sampler_config = get_distribution_template('none', max=n_extra, min=0, mean=None, std=None, distribution='uniform_int')
            start = get_sample(sampler_config)[0]
            noise_sampled = noise[start:start+required_length,:]

        return noise_sampled, start

    @staticmethod
    def repeat_noise(noise, required_length):
        """Randomly sample a noise of required length.
        If noise is shorter than required length, repeat noise first. """
        required_length = int(required_length)
        n_sample, n_ch = noise.shape

        # repeat noise if necessary
        if n_sample < required_length:
            repeat_times = int(np.ceil(required_length / n_sample))
            noise = np.tile(noise, (repeat_times, 1))
            n_sample = noise.shape[0]

        # randomly sample an initial point
        if n_sample==required_length:
            start = 0
        else:
            start = np.random.randint(0, high=n_sample - required_length, size=1)[0]
        noise_repeated = noise[start:start + required_length,:]

        return noise_repeated, start


class Distorter:
    """
    Apply room impuse response to input signals. Add noise.
    """
    def __init__(self):
        pass

    @staticmethod
    def apply_rir_and_noise():
        pass

    @staticmethod
    def add_noise(signal, noise, snr, noise_position_scheme='repeat_noise'):
        """
        Add additive noise to signal
        :param signal: TxC matrix, where T is the number of samples and C is the number of channel
        :param noise: T2xC matrix, where T2 is the number of samples of the noise
        :param snr: a scalar that specifies signal-to-noise ratio (SNR)
        :param noise_position_scheme: specify how to position the noise in the final waveform.
        :return: distorted signal waveform
        """
        n_sample,n_ch = signal.shape
        scale = _comp_noise_scale_given_snr(signal, noise, snr)
        noise_scaled = noise * scale
        if noise_position_scheme == 'repeat_noise':
            noise_positioned, idx = _NoiseSampler.repeat_noise(noise_scaled, n_sample)
        elif noise_position_scheme == 'sample_noise':
            noise_positioned, idx = _NoiseSampler.sample_noise(noise_scaled, n_sample)
        else:
            raise Exception("Unknown noise position scheme %s" % (noise_position_scheme))

        distorted = signal + noise_positioned

        return distorted, noise_positioned

    @staticmethod
    def apply_rir(wav, rir, fs=16000, sync=True, get_early_reverb=False, early_reverb_cutoff_time=0.04):
        """
        Apply room impulse response to the input signal.

        :param wav: 1D array of source signal waveform
        :param rir: TxC matrix, where T is the number of samples in RIR waveform, and C is the number of channels.
        :param fs: sampling rate
        :param sync: if set to True, output signal will be sample-synchrnoized to input signal. Note that RIR typically
               causes a time shift of convolved signal.
        :param get_early_reverb: whether to also return early reverbed signal. Early reverbed signal is the obtained by
               convoling the signal with only the early reverb part of the RIR. See definition of early_reverb_cutoff_time
        :param early_reverb_cutoff_time: the duration in terms of seconds of early reverb responses in RIR.
        :return: both reverb and early_reverb signals
        """
        n_sample = wav.size
        n_sample_rir = rir.shape[0]
        wav = wav.reshape(n_sample, 1)
        delay = int(np.argmax(rir,axis=0)[0])

        # generate reverberant speech
        reverb = _fftconvolve1d(rir, wav)

        if get_early_reverb:
            rir_cutoff = int(np.minimum(n_sample_rir, early_reverb_cutoff_time*fs+delay))
            rir_early = rir[:rir_cutoff, :]
            early_reverb = _fftconvolve1d(rir_early, wav)
        else:
            early_reverb = None

        if sync:
            reverb = reverb[delay - 1:delay + n_sample - 1, :]
            if early_reverb is not None:
                early_reverb = early_reverb[delay - 1:delay + n_sample - 1, :]

        return reverb, early_reverb


def _test_speed():
    rir = np.random.randn(8000,1)
    speech = np.random.randn(160000,1)

    def tic():
        import time
        return time.time()

    def toc(start_time):
        import time
        print("Elapsed time is %s seconds." % (str(time.time() - start_time)))

    t1 = tic()
    for i in range(100):
        result1 = _fftconvolve1d(rir, speech)
    toc(t1)
    return_size = result1.size

    t2 = tic()
    for i in range(100):
        result2 = scipy.signal.fftconvolve(rir, speech)[:return_size, :]
    toc(t2)

    t3 = tic()
    # scipy convolve automatically select freq. or time domain convolution. May be much faster than scipy fftconvolve
    # for short inputs.
    for i in range(100):
        result3 = scipy.signal.convolve(rir, speech)[:return_size, :]
    toc(t3)

    import matplotlib.pyplot as plt
    plt.plot(result1 - result2)


#_test_speed()