import numpy as np
from . import _sampling


class MixerConfig:
    """
    Configurations on mixing two speech sources.
    """
    def __init__(self):
        self.config = dict()
        self.config['mixed_length'] = {}
        self.config['mixed_length']['scheme'] = 'longest_source'
        self.config['mixed_length']['min'] = 5
        self.config['positioning'] = {}
        self.config['positioning']['scheme'] = 'random_start'
        self.config['ref_source'] = 'first_source'
        self.config['spr'] = _sampling.get_distribution_template(
            'distribution of signal power ratio (SPR) between the speech sources',
            max=2.5, min=-2.5, distribution='uniform')


class Mixer:
    """
    Mixes two speech signals to simulate overlapping speech.
    """
    def __init__(self, config):
        self.config = config

    def mix_signals(self, signals, spr=None, signal2=None):
        """
        :param signals: list of numpy arrays, each represent the waveform of one speech source
        :param spr: speaker-to-interference ratio, the energy ratio between the two speakers.
        :param signal2: optimal second set of signals that have one-to-one correspondence with signals. If provided,
        will scale and shift it in the same way as signal.
        :return: mixed speech and related settings
        """
        # get the size of the sources
        signal_sizes = np.asarray([np.shape(x) for x in signals])
        n_ch = signal_sizes[0, 1]

        # determine the length of the mixed signal
        max_length = np.max(signal_sizes[:,0])
        if self.config['mixed_length']['scheme'] == 'longest_source':
            mixed_length = max_length
        else:
            raise Exception("Mixer::mix_signals: Unknown mixed length scheme %s" % (self.config['mixed_length']['scheme']))

        if spr is None:
            spr = get_sample(self.config['spr'], signal_sizes.shape[0]-1)

        scale = self._compute_source_scales(signals, spr)

        #
        if self.config['positioning']['scheme'] == 'random_start':
            mixed, positioned_source, start_sample_idx = self._mix_by_random_start(signals, mixed_length, scale)
        else:
            raise Exception("Mixer::mix_signals: Unknown mixing scheme %s" % (self.config['positioning']['scheme']))

        if signal2 is not None:
            positioned_source2 = []
            for i in range(len(signals)):
                positioned_source2.append(np.zeros((mixed_length, n_ch)))
                positioned_source2[i][start_sample_idx[i]:start_sample_idx[i] + signal_sizes[i, 0]] = signal2[i]*scale[i]
            return mixed, positioned_source, start_sample_idx, scale, positioned_source2
        else:
            return mixed, positioned_source, start_sample_idx, scale

    def _compute_source_scales(self, signals, spr):
        """Compute the scales of the sources according to the SPR."""
        if self.config['ref_source'] == 'first_source':
            spr = np.insert(spr, 0, 0)
            ref_sig = signals[0]
        else:
            raise Exception("Mixer::mix_signals: Unknown reference source %s" % (self.config['ref_source']))

        ref_power = np.mean(ref_sig ** 2)
        scale = np.zeros((len(signals),1))
        sig_power = np.zeros((len(signals), 1))
        for i in range(len(signals)):
            sig_power[i] = np.mean(signals[i] ** 2)
            scale[i] = np.sqrt(ref_power / sig_power[i] * 10**(spr[i]/10))

        return scale

    def _mix_by_random_start(self, signals, mixed_length, scale):
        """Given the duration of the mixed signal, sample a starting point for every signals"""
        source_sizes = np.asarray([np.shape(x) for x in signals])
        n_ch = source_sizes[0,1]

        mixed = 0
        start_sample_idx = []
        positioned_source = []
        for i in range(len(signals)):
            if source_sizes[i,0] > mixed_length:
                raise Exception("Mixer::mix_by_random_start: source length (%d) is longer than mixed length (%d)" % (source_sizes[i,0], mixed_length))
            elif source_sizes[i,0] == mixed_length:
                positioned_source.append(signals[i])
                start_sample_idx.append(0)
            else:
                n_extra = mixed_length - source_sizes[i, 0]
                sampler_config = get_distribution_template('none', max=n_extra, min=0, mean=None, std=None, distribution='uniform_int')
                start_sample_idx.append(get_sample(sampler_config)[0])
                positioned_source.append(np.zeros((mixed_length, n_ch)))
                positioned_source[i][start_sample_idx[i]:start_sample_idx[i]+source_sizes[i,0]] = signals[i]
            mixed += positioned_source[i] * scale[i]

        return mixed, positioned_source, start_sample_idx

    def _mix_by_repeat_short(self, signals, mixed_length, scale):
        """Given the duration of the mixed signal, repeat signals that are shorter than the length then randomly sample
        one segment from the repeated signal."""
        pass # to be implemented

    def _mix_by_cut_long(self, signals, mixed_length):
        """Given the duration of the mixed signal, cut the signals that are too long to fit the mixed length."""
        pass # to be implemented
