import numpy as np
import json
from ._iso_noise_simulator import generate_isotropic_noise
from ._distorter import Distorter
from ._mixer import Mixer
from .mask import MaskEstimator
from .freq_analysis import SpectrumAnalyzer
from ._sampling import get_distribution_template, get_sample, sample_room, sample_array_position, sample_source_position, sample_source_position_by_random_coordinate
from .config import single_channel_single_source_config, multi_channel_single_source_config, single_channel_multi_source_config, multi_channel_multi_source_config


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


class _Simulator:
    """
    A room simulation class for generating reverberant and noisy speech signals that may be overlapped.
    Supports both single and multi-channel simulation, and single and multi-source simulation.
    """

    def __init__(self, config):
        self._config = config
        self._analyzer = SpectrumAnalyzer(config['analysis'])
        self._mask_estimator = MaskEstimator(self._analyzer)
        
    def _apply_rir(self, source_wav, source_rir, get_early_reverb=True):
        """Apply RIR on sources"""
        reverb = []
        early_reverb = []
        for i in range(len(source_rir)):
            tmp_reverb, tmp_early_reverb = Distorter.apply_rir(source_wav[i], source_rir[i], get_early_reverb=get_early_reverb)
            reverb.append(tmp_reverb)
            early_reverb.append(tmp_early_reverb)

        return reverb, early_reverb

    def _reshape_wav(self, list_of_wav):
        for i in range(len(list_of_wav)):
            if list_of_wav[i].ndim == 1:
                list_of_wav[i] = list_of_wav[i][:, np.newaxis]
        return

    def simulate(self, source_wav, mixer=None, dir_noise_wav=None, source_rir=None, dir_noise_rir=None, iso_noise_wav=None,
                 gen_mask=False, normalize_gain=True):
        """simulate one sentence.

        Parameters
        ----------
        source_wav: list of numpy array, waveforms of clean speech sources.
        mixer: a mixing object that mixes source speech waveforms. Must not be None if more than one source_wav is provided.
        dir_noise_wav: list of numpy array, waveforms of directional noise sources.
        source_rir: list of numpy array
            room impulse response (RIR) to be applied to source_wav. The number of RIRs must be equal or larger than the
            number of source_wav. In ideal case, the RIRs should be sampled from the same room.
        dir_noise_rir: list of numpy array
            RIR to be applied to directional noise waveforms. The number of RIRs must be equal or larger than the number
            of dir_noise_wav. All RIRs of noise and speech sources should be compatible and sampled from the same room.
        iso_noise_wav: numpy array
            Isotropic noise to be added to simulated speech. Need to be compatible with the RIRs in terms of microphone
            array geometry and room.
        gen_mask: bool
            whether to generate time-frequency ideal binary mask.
        normalize_gain: bool
            whether to normalize the gain of the simulated uttreance.

        Returns
        -------
        mixed_noisy: ndarray
            simulated utterance, which may be overlapped, reverberant, and noisy.
        positioned_source_early_reverb: list of ndarray
            all source speech signals with only early reverberation but no noise. Sometimes it is good to be used as
            training target for speech enhancement tasks.
        mask: ndarray
            time-frequency ideal binary mask.
        sent_cfg: dict
            configuration about how to apply simulation to current sentence
        """

        # step 1. get speech and noise source waveforms

        sent_cfg = dict()

        n_source = len(source_wav)
        self._reshape_wav(source_wav)

        n_dir_noise = len(dir_noise_wav) if dir_noise_wav is not None else 0
        if n_dir_noise>0:
            self._reshape_wav(dir_noise_wav)

        # sanity check on RIRs
        use_rir = source_rir is not None and len(source_rir)>0
        if use_rir:
            assert len(source_rir) == n_source, \
                'number of source_rir ({}) does not equal to number of source ({})'.format(len(source_rir), n_source)
            if n_dir_noise > 0:
                assert dir_noise_rir is not None and len(dir_noise_rir) == n_dir_noise, \
                    'number of dir_noise_rir does not equal to number of directional noise sources'.format(len(dir_noise_rir), n_dir_noise)

        # step 2. apply room impulse responses (optional)

        if use_rir:
            self._reshape_wav(source_rir)
            source_reverb, source_early_reverb = self._apply_rir(source_wav, source_rir, get_early_reverb=gen_mask)

            if n_dir_noise > 0:
                self._reshape_wav(dir_noise_rir)
                dir_noise_reverb, dir_noise_early_reverb = self._apply_rir(dir_noise_wav, dir_noise_rir,
                                                                           get_early_reverb=False)
        else:
            source_reverb, source_early_reverb = source_wav, source_wav
            if n_dir_noise > 0:
                dir_noise_reverb, dir_noise_early_reverb = dir_noise_wav, dir_noise_wav

        # step 3. mix speech sources and noises (optional)

        if n_source == 1:
            mixed = source_reverb[0]
            positioned_source = source_reverb
            start_sample_idx = [0]
            scale = np.ones((1, 1))
            positioned_source_early_reverb = source_early_reverb
        else:
            sent_cfg['spr'] = get_sample(simu_cfg['mixing']['spr'], sent_cfg['n_source'] - 1)
            mixed, positioned_source, start_sample_idx, scale, positioned_source_early_reverb = \
                mixer.mix_signals(source_reverb, sent_cfg['spr'], signal2=source_early_reverb)

        # add the combined sources with directional noise and isotropic noise using the SNR
        mixed_noisy=mixed
        if n_dir_noise>0:
            sent_cfg['dir_snr'] = get_sample(self._config['dir_noise']['snr'], n_sample=n_dir_noise)
            for i in range(n_dir_noise):
                tmp_mixed_noisy, scaled_dir_noise = Distorter.add_noise(mixed, dir_noise_reverb[i],
                                                                        sent_cfg['dir_snr'][i],
                                                                        noise_position_scheme='sample_noise')
                mixed_noisy += scaled_dir_noise

        # step 4. add isotropic noise (optional)

        n_sample, n_ch = mixed_noisy.shape
        if iso_noise_wav is not None:
            sent_cfg['iso_snr'] = get_sample(simu_cfg['iso_noise']['snr'])
            tmp_mixed_noisy, scaled_iso_noise = Distorter.add_noise(mixed, iso_noise_wav, sent_cfg['iso_snr'],
                                                                    noise_position_scheme='repeat_noise')
            mixed_noisy += scaled_iso_noise

        # step 5. generate ideal time-frequency mask (optional, usually used as neural network training target)

        if gen_mask:
            mask = []
            for i in range(len(positioned_source_early_reverb)):
                mask.append(self._mask_estimator.get_mask_from_parallel_data(positioned_source_early_reverb[i][:, 0],
                                                                            mixed_noisy[:, 0]))
        else:
            mask = None

        # step 6. normalize the gain of the simulated speech (optional, usually applied if RIR is applied, due to that
        # RIR may change the gain of the speech signal significantly.

        if normalize_gain:
            gain_norm_scale = 0.5 / np.max(np.abs(mixed_noisy))
            mixed_noisy *= gain_norm_scale
            if positioned_source_early_reverb[0] is not None:
                positioned_source_early_reverb = [i*gain_norm_scale for i in positioned_source_early_reverb]
            sent_cfg['gain_norm_scale'] = gain_norm_scale

        return mixed_noisy, positioned_source_early_reverb, mask, sent_cfg


class SimpleSimulator:
    """Single speech source simulator. """
    def __init__(self, array_geometry=None, use_rir=True, use_noise=True, snr_range=(0,30)):
        """
        Parameters
        ----------
        array_geometry: numpy array
            a MxN matrix, where M=2 or 3 is the 2D or 3D coordinates, and N is the number of channels.
            If not given, assuming using single channel simulation.
        use_rir: bool, whether to apply room impulse response (RIR). Multi-channel simulation should always apply RIR.
        use_noise: bool, whether to add noise
        snr_range: tuple of numbers, representing the min and max of SNR.
        """
        single_channel=True
        if array_geometry is not None and array_geometry.ndim>1 and array_geometry.shape[1] > 1:
            single_channel = True
        if single_channel:
            cfg = single_channel_single_source_config(use_reverb=use_rir, use_noise=use_noise, snr_range=snr_range)
        else:
            cfg = multi_channel_single_source_config(array_geometry, use_iso_noise=use_reverb,
                                                     use_dir_noise=use_noise, snr_range=snr_range)
        self.simulator = _Simulator(cfg)

    def __call__(self, source_wav, dir_noise_wavs=None, source_rir=None, dir_noise_rirs=None, iso_noise_wav=None,
                 gen_mask=False, min_length=None, normalize_gain=True):
        """simulate one sentence.

        Parameters
        ----------
        source_wav: numpy array, single-channel clean speech waveform.
        dir_noise_wav: list of numpy array, directional noise waveforms.
        source_rir: numpy array
            room impulse response (RIR) to be applied to source_wav. For multi-channel simulation, the array geometry
            of the RIR should be compatible with the array_geometry used to initialize the object.
        dir_noise_rir: list of numpy array
            RIR to be applied to directional noise waveforms. The number of RIRs must be equal or larger than the number
            of dir_noise_wav. All RIRs of noise and speech sources should be compatible and sampled from the same room.
        iso_noise_wav: numpy array
            Isotropic noise to be added to simulated speech. Need to be compatible with the RIRs in terms of microphone
            array geometry and room.
        gen_mask: bool, whether to generate time-frequency ideal binary mask.
        min_length: float, minimum duration of the simulated sentences in seconds.
        normalize_gain: bool, whether to normalize the gain of the simulated uttreance.
        """
        mixed_noisy, positioned_source_early_reverb, mask, sent_cfg = self.simulator.simulate(
            source_wav=[source_wav],
            dir_noise_wav=dir_noise_wavs,
            source_rir=None if source_rir is None else [source_rir],
            dir_noise_rir=dir_noise_rirs,
            iso_noise_wav=iso_noise_wav,
            gen_mask=gen_mask,
            normalize_gain=normalize_gain)

        return mixed_noisy, positioned_source_early_reverb, mask, sent_cfg


class MultiSourceSimulator:
    """Multiple speech source simulator. Used to generate overlapping speech. """
    def __init__(self, array_geometry=None, use_rir=True, use_noise=True, snr_range=(0,30), n_source_range=(2,2),
                 spr_range=(-2.5,2.5)):
        """
        Parameters
        ----------
        array_geometry: numpy array
            a MxN matrix, where M=2 or 3 is the 2D or 3D coordinates, and N is the number of channels.
            If not given, assuming using single channel simulation.
        use_rir: bool, whether to apply room impulse response (RIR). Multi-channel simulation should always apply RIR.
        use_noise: bool, whether to add noise
        snr_range: tuple of 2 numbers, representing the min and max of SNR.
        n_source_range: tuple of 2 numbers, representing the min and max number of speech sources.
        spr_range: tuple of 2 numbers, representing the min and max of signal-to-interference ratio between speech sources.
        """
        single_channel=True
        if array_geometry is not None and array_geometry.ndim>1 and array_geometry.shape[1] > 1:
            single_channel = True
        if single_channel:
            cfg = single_channel_multi_source_config(n_source_range=n_source_range, spr_range=spr_range,
                                                     use_reverb=use_rir, use_noise=use_noise,
                                                     snr_range=snr_range)
        else:   # multi-channel simulation always uses RIR (reverb)
            cfg = multi_channel_multi_source_config(array_geometry, n_source_range=n_source_range, spr_range=spr_range,
                                                    use_dir_noise=use_noise, use_iso_noise=use_noise,
                                                    snr_range=snr_range)
        self.simulator = _Simulator(cfg)

    def __call__(self, source_wavs, dir_noise_wavs=None, source_rirs=None, dir_noise_rirs=None, iso_noise_wav=None,
                 gen_mask=False, min_length=None, normalize_gain=True):
        """simulate one sentence.

        Parameters
        ----------
        source_wavs: numpy array, single-channel clean speech waveform.
        dir_noise_wavs: list of numpy array, directional noise waveforms.
        source_rirs: numpy array
            room impulse response (RIR) to be applied to source_wav. For multi-channel simulation, the array geometry
            of the RIR should be compatible with the array_geometry used to initialize the object.
        dir_noise_rir:s list of numpy array
            RIR to be applied to directional noise waveforms. The number of RIRs must be equal or larger than the number
            of dir_noise_wav. All RIRs of noise and speech sources should be compatible and sampled from the same room.
        iso_noise_wav: numpy array
            Isotropic noise to be added to simulated speech. Need to be compatible with the RIRs in terms of microphone
            array geometry and room.
        gen_mask: bool, whether to generate time-frequency ideal binary mask.
        min_length: float, minimum duration of the simulated sentences in seconds.
        normalize_gain: bool, whether to normalize the gain of the simulated uttreance.
        """
        mixed_noisy, positioned_source_early_reverb, mask, sent_cfg = self.simulator.simulate(
            source_wav=source_wavs,
            dir_noise_wav=dir_noise_wavs,
            source_rir=source_rirs,
            dir_noise_rir=dir_noise_rirs,
            iso_noise_wav=iso_noise_wav,
            gen_mask=gen_mask,
            normalize_gain=normalize_gain)

        return mixed_noisy, positioned_source_early_reverb, mask, sent_cfg
