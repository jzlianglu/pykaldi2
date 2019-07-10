import numpy as np
from.sampling import get_distribution_template, get_sample
import scipy.fftpack


def fftconvolve1d(in1, in2, use_gpu=False):
    """1D convolution along the first dimension"""
    m, n1 = in1.shape
    k, n2 = in2.shape
    rlen = m + k - 1
    rlen_p2 = scipy.fftpack.helper.next_fast_len(int(rlen))

    XX = np.fft.rfft(in1, rlen_p2, axis=0)
    YY = np.fft.rfft(in2, rlen_p2, axis=0)
    ret = np.fft.irfft(XX * YY, rlen_p2, axis=0)

    return ret[:rlen, :]


def comp_noise_scale_given_snr(signal, noise, snr):
    Px = np.mean(signal ** 2)
    Pn = np.mean(noise ** 2)
    scale = np.sqrt(Px / Pn * 10 ** ((-snr) / 10))
    return scale


class NoiseSampler:
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


class Distortor:
    def __init__(self):
        pass

    @staticmethod
    def apply_rir_and_noise():
        pass

    @staticmethod
    def add_noise(signal, noise, snr, noise_position_scheme='repeat_noise'):
        n_sample,n_ch = signal.shape
        scale = comp_noise_scale_given_snr(signal, noise, snr)
        noise_scaled = noise * scale
        if noise_position_scheme == 'repeat_noise':
            noise_positioned, idx = NoiseSampler.repeat_noise(noise_scaled, n_sample)
        elif noise_position_scheme == 'sample_noise':
            noise_positioned, idx = NoiseSampler.sample_noise(noise_scaled, n_sample)
        else:
            raise Exception("Unknown noise position scheme %s" % (noise_position_scheme))

        distorted = signal + noise_positioned

        return distorted, noise_positioned

    @staticmethod
    def apply_rir(wav, rir, fs=16000, sync=True, get_early_reverb=False, early_reverb_cutoff_time=0.04):
        n_sample = wav.size
        n_sample_rir = rir.shape[0]
        wav = wav.reshape(n_sample, 1)
        delay = int(np.argmax(rir,axis=0)[0])

        # generate reverberant speech
        reverb = fftconvolve1d(rir, wav)

        if get_early_reverb:
            rir_cutoff = int(np.minimum(n_sample_rir, early_reverb_cutoff_time*fs+delay))
            rir_early = rir[:rir_cutoff, :]
            early_reverb = fftconvolve1d(rir_early, wav)
        else:
            early_reverb = None

        if sync:
            reverb = reverb[delay - 1:delay + n_sample - 1, :]
            if early_reverb is not None:
                early_reverb = early_reverb[delay - 1:delay + n_sample - 1, :]

        return reverb, early_reverb

