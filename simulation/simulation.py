import numpy as np
import json
from . import rirgen
from . import farfield_simulator as iso_noise_simulator
from .distortor import Distortor
from .mixer import Mixer
from .mask import MaskEstimator
from . import feature
from .sampling import get_distribution_template, get_sample, sample_room, sample_array_position, sample_source_position, sample_source_position_by_random_coordinate
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


def set_config(config_dict, key, value, overwrite=True):
    """If overwrite is False, and the key is already set in config_dict, keep the original value."""
    if overwrite is False and key in config_dict.keys() and config_dict[key] is not None:
        return
    config_dict[key] = value


class Simulator:
    """
    A versatile room simulation class for generating reverberant and noisy speech signals that may be overlapped.
    Supports both single and multi-channel simulation, and single and multi-source simulation.
    """

    def __init__(self, config, speech_streams=None, noise_streams=None, rir_streams=None, iso_noise_streams=None, DEBUG=False):
        """
        Parameters
        ----------
        config : dict
            a dictionary that defines various settings of the simulation. The dict should be generating from a function
            from config.py.
        speech_streams : list
            A list of SpeechDataStream objects, each defines a speech data corpus. Speech waveforms will be sampled from
            the speech_streams if user does not provide clean speech source waveforms when calling self.simulate().
        noise_streams : list
            A list of DataStream objects, each defines a noise corpus. Noise waveforms will be sampled from the
            noise_streams if user does not provide directional noise waveforms when calling self.simulate().
        rir_streams : list
            A list of RIRStream objects, each defines a room impulse response (RIR) corpus. RIR will be sampled from
            rir_streams if user does not provide RIR when calling self.simulate(). If rir_streams is None, RIR will be
            dynamically simulated if config['reverb']['user_reverb'] is True.
        iso_noise_streams:
            A list of DataStream objects, each defines an isotropic noise corpus that will be used in multi-channel
            simulation. Usually used ONLY for multi-channel simulation. If iso noise waveform is not provided when user
            calls self.simulate(), it will be sampled from iso_noise_streams. If this stream is None, noise will be
            dynamically simulated (highly not recommended as it is very slow).
        """
        self._load_config(config)
        self.speech_streams = speech_streams    # speech_streams is a list of streams, so we can support multiple clean speech corpus. Same to other input streams.
        self.noise_streams = noise_streams
        self.rir_streams = rir_streams
        self.iso_noise_streams = iso_noise_streams
        self.DEBUG = DEBUG

        self.analyzer = feature.SpectrumAnalyzer(config['analysis'])
        self.mask_estimator = MaskEstimator(self.analyzer)

        # sanity check
        if self.config['iso_noise']['use_iso_noise'] and self.config['iso_noise']['use_corpus'] and self.iso_noise_streams is None:
            print("Warning: no isotropic noise corpus provided!\n")

        if self.config['reverb']['use_reverb'] and self.config['reverb']['use_corpus'] and self.rir_streams is None:
            print("Warning: no rir corpus provided!\n")

        if self.config['dir_noise']['use_dir_noise'] and self.noise_streams is None:
            print("Warning: no noise corpus provided!\n")

        # We support multiple input streams for each source. So we need a uniform sampler to sample the streams
        self._speech_streams_sampler = self._gen_stream_sampler(self.speech_streams, 'Speech streams sampler')
        self._noise_streams_sampler = self._gen_stream_sampler(self.noise_streams, 'Directional noise streams sampler')
        self._rir_streams_sampler = self._gen_stream_sampler(self.rir_streams, 'RIR streams sampler')
        self._iso_streams_sampler = self._gen_stream_sampler(self.iso_noise_streams, 'Isotropic noise streams sampler')

    def _gen_stream_sampler(self, streams, name):
        if streams is not None:
            n_entrys = [stream.get_number_of_data() for stream in streams]
            n_entrys = np.asarray(n_entrys)
            
            pmf = n_entrys / np.sum(n_entrys)
            streams_sampler = get_distribution_template(name, category=np.arange(len(streams)), pmf=pmf, distribution='discrete')
        else:
            streams_sampler = None
        return streams_sampler

    def _load_config(self, config):
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = json.load(open(config))
        # check config
        if not self.config['dir_noise']['use_dir_noise']:
            self.config['dir_noise']['num_dir_noise']['max'] = 0
            self.config['dir_noise']['num_dir_noise']['min'] = 0

        self.config['array']['n_mic'] = len(config['array']['mic_positions'][0])

    def _get_rir(self, source_position, mic_position, room_size, t60):
        n_spk = source_position.shape[1]
        RIR = []
        for i in range(n_spk):
            rir = rirgen.xp_rirgen(room_size.reshape(3, 1), source_position[:,i].reshape(3, 1), mic_position, t60=t60,
                                   hpfilt=True, habets_compat=True)
            rir = np.squeeze(rir).transpose()
            if rir.ndim==1:
                rir = rir.reshape(rir.size,1)
            RIR.append(np.asarray(rir))
        return RIR

    def _apply_rir(self, source_wav, source_rir, get_early_reverb=True):
        """Apply RIR on sources"""
        reverb = []
        early_reverb = []
        for i in range(len(source_rir)):
            tmp_reverb, tmp_early_reverb = Distortor.apply_rir(source_wav[i], source_rir[i], get_early_reverb=get_early_reverb)
            reverb.append(tmp_reverb)
            early_reverb.append(tmp_early_reverb)

        return reverb, early_reverb

    def simulate_given_utt_id(self, source_stream_idx, sent_cfg=None, gen_mask=False, spk_id=None, utt_id=None, unwanted_utt_id=None, min_length=None):
        """Sometimes, we want to simulate with a specific speaker or utterances.
        if spk_id is not given, utt_id must be given
        if utt_id is given, use it
        if utt_id is not given, but unwanted_utt_id is given, sample sentences excluding unwanted utterances.
        """
        if spk_id is None and utt_id is None:
            raise Exception("Simulator::simulate_given_utt_id: spk_id and utt_id cannot be both None. ")

        sent_cfg = self.sample_sent_cfg(sent_cfg)
        sent_cfg['source_stream_idx'] = source_stream_idx

        # sample utt_id that satisfy requirements
        if utt_id is not None:
            sent_cfg['source_utt_id'] = utt_id
        else:
            tmp = self.speech_streams[source_stream_idx].sample_utt_from_spk(spk_id, unwanted_utt_id=unwanted_utt_id,
                                                                             replace=False, load_data=False,
                                                                             load_vad=False, min_length=min_length)
            sent_cfg['source_utt_id'] = tmp[1]

        sent_cfg['source_speakers'] = [self.speech_streams[source_stream_idx].utt2spk[i] for i in sent_cfg['source_utt_id']]

        return self.simulate(sent_cfg, gen_mask=gen_mask, min_length=min_length)

    def _reshape_wav(self, list_of_wav):
        for i in range(len(list_of_wav)):
            if list_of_wav[i].ndim == 1:
                list_of_wav[i] = list_of_wav[i][:, np.newaxis]
        return

    def simulate(self, source_wav=None, dir_noise_wav=None, source_rir=None, dir_noise_rir=None, iso_noise_wav=None,
                 sent_cfg=None, gen_mask=False, min_length=None, normalize_gain=True):
        """simulate one sentence.

        Parameters
        ----------
        source_wav: list of numpy array
            clean speech waveforms. If not provided, sample sentences from self.speech_streams.
        dir_noise_wav: list of numpy array
            directional noise waveforms. If not provided, sample from self.noise_streams. If self.noise_streams is None, do not use directional noise.
        source_rir: list of numpy array
            room impulse response (RIR) to be applied to source_wav. The number of RIRs must be equal or larger than the
            number of source_wav. The RIRs overrides the microphone array settings in in self.confg['array'].
            In ideal case, the RIRs should be sampled from the same room.
        dir_noise_rir: list of numpy array
            RIR to be applied to directional noise waveforms. The number of RIRs must be equal or larger than the number
            of dir_noise_wav. All RIRs of noise and speech sources should be compatible and sampled from the same room.
        iso_noise_wav: numpy array
            Isotropic noise to be added to simulated speech. Need to be compatible with the RIRs in terms of microphone
            array geometry and room.
        sent_cfg: dict
            input configuration for the utterance, usually set to None.
        gen_mask: bool
            whether to generate time-frequency ideal binary mask.
        min_length: float
            minimum duration of the simulated sentences in seconds.
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
            record all the details of the simulation for the current utterance.
        """

        assert source_wav is not None or self.speech_streams is not None

        # sample configuration for the sentence. Note that those settings already in sent_cfg will be kept.
        sent_cfg = self.sample_sent_cfg(sent_cfg)

        # step 1. get speech and noise source waveforms

        if source_wav is not None:
            n_source = len(source_wav)
            sent_cfg['n_source'] = n_source
        else:            # if no input_wavs is provided, sample clean waveforms
            n_source = sent_cfg['n_source']
            if 'source_stream_idx' not in sent_cfg.keys():
                sent_cfg['source_stream_idx'] = get_sample(self._speech_streams_sampler)[0]
            if 'source_utt_id' not in sent_cfg.keys() or 'source_speakers' not in sent_cfg.keys():
                sent_cfg['source_speakers'], sent_cfg['source_utt_id'], source_wav, source_vad = \
                    self.speech_streams[sent_cfg['source_stream_idx']].sample_spk_and_utt(n_spk=n_source,
                                                                                          n_utt_per_spk=1,
                                                                                          load_data=True,
                                                                                          load_vad=True,
                                                                                          min_length=min_length)
            else:
                _, _, source_wav, source_vad = self.speech_streams[sent_cfg['source_stream_idx']].read_utt_with_id(
                    sent_cfg['source_utt_id'], load_data=True, load_vad=True)
        self._reshape_wav(source_wav)

        if dir_noise_wav is not None:
            n_dir_noise = len(dir_noise_wav)
            sent_cfg['n_dir_noise'] = n_dir_noise
        else:           # if noise is not provided, sample from noise stream
            if self._noise_streams_sampler is not None:
                n_dir_noise = sent_cfg['n_dir_noise']
                sent_cfg['noise_stream_idx'] = get_sample(self._noise_streams_sampler)[0]
                dir_noise_wav, _ = self.noise_streams[sent_cfg['noise_stream_idx']].sample_data(n_dir_noise)
            else:
                n_dir_noise = sent_cfg['n_dir_noise']
        if n_dir_noise>0:
            self._reshape_wav(dir_noise_wav)

        # step 2. apply room impulse responses (optional)

        if self.config['reverb']['use_reverb']:
            if source_rir is not None:
                self._reshape_wav(source_rir)
                if dir_noise_rir is not None:
                    self._reshape_wav(dir_noise_rir)
                elif n_dir_noise > 0:
                    dir_noise_rir = self._get_rir(sent_cfg['dir_noise_position'],
                                                  sent_cfg['array_position']['mic_position'],
                                                  sent_cfg['room_size'], sent_cfg['t60'])

            elif 'source_rir_wav' in sent_cfg:
                source_rir = sent_cfg['source_rir_wav']
                if sent_cfg['dir_noise_position'].size > 0:
                    dir_noise_rir = sent_cfg['dir_noise_rir_wav']
            else:
                source_rir = self._get_rir(sent_cfg['source_position'], sent_cfg['array_position']['mic_position'],
                                          sent_cfg['room_size'], sent_cfg['t60'])
                if sent_cfg['dir_noise_position'].size > 0:
                    dir_noise_rir = self._get_rir(sent_cfg['dir_noise_position'],
                                                  sent_cfg['array_position']['mic_position'],
                                                  sent_cfg['room_size'], sent_cfg['t60'])

            # Apply RIRs to sources and directional noises
            source_reverb, source_early_reverb = self._apply_rir(source_wav, source_rir, get_early_reverb=True)
            if n_dir_noise > 0:
                dir_noise_reverb, dir_noise_early_reverb = self._apply_rir(dir_noise_wav, dir_noise_rir,
                                                                          get_early_reverb=False)
        else:
            source_reverb, source_early_reverb = source_wav, source_wav
            if n_dir_noise > 0:
                dir_noise_reverb, dir_noise_early_reverb = dir_noise_wav, dir_noise_wav

        # step 3. mix speech sources and noises (optional)

        if len(source_reverb)==1:
            mixed = source_reverb[0]
            positioned_source = source_reverb
            start_sample_idx = [0]
            scale = np.ones((1, 1))
            positioned_source_early_reverb = source_early_reverb
        else:
            # Sample positions of sources in the mixtures waveform, mix the sources according to SPR
            source_mixer = Mixer(self.config['mixing'])
            mixed, positioned_source, start_sample_idx, scale, positioned_source_early_reverb = \
                source_mixer.mix_signals(source_reverb, sent_cfg['spr'], signal2=source_early_reverb)

        # Mix the combined sources with directional noise and isotropic noise using the SNR
        mixed_noisy=mixed
        for i in range(n_dir_noise):
            tmp_mixed_noisy, scaled_dir_noise = Distortor.add_noise(mixed, dir_noise_reverb[i],
                                                                    sent_cfg['dir_snr'][i],
                                                                    noise_position_scheme='sample_noise')
            mixed_noisy += scaled_dir_noise

        # step 4. add isotropic noise (optional)

        n_sample, n_ch = mixed_noisy.shape
        if self.config['iso_noise']['use_iso_noise']:
            if iso_noise_wav is not None:
                pass
            elif self.config['iso_noise']['use_corpus']:
                sent_cfg['iso_noise_stream_idx'] = get_sample(self._iso_streams_sampler)[0]
                iso_noise_wav, iso_noise_files = self.iso_noise_streams[sent_cfg['iso_noise_stream_idx']].sample_data()
                iso_noise_wav = iso_noise_wav[0]
                if iso_noise_wav.shape[1]>1 and n_ch==1:
                    iso_channel_idx = np.random.randint(0, high=iso_noise_wav.shape[1])
                    iso_noise_wav = iso_noise_wav[:,[iso_channel_idx]]
                else:
                    assert iso_noise_wav.shape[1]==n_ch, "Simulator::Simulate: isotropic noise's number of channels (%d) does not match the requirement (%d)" % (iso_noise_wav.shape[1], n_ch)
            else:
                if n_ch == 1:  # for single channel simulation, we add stationary white noise or provided noise
                    iso_noise_wav = np.random.normal(size=(n_sample, n_ch))
                else:
                    iso_noise_wav = iso_noise_simulator.generate_isotropic_noise(
                        sent_cfg['array_position']['mic_position'].T, n_sample,
                        samp_rate=self.config['analysis']['fs'],
                        type=self.config['iso_noise']['type'],
                        spectrum=self.config['iso_noise']['spectrum_type']).T

            tmp_mixed_noisy, scaled_iso_noise = Distortor.add_noise(mixed, iso_noise_wav, sent_cfg['iso_snr'],
                                                                    noise_position_scheme='repeat_noise')
            mixed_noisy += scaled_iso_noise

        # step 5. generate ideal time-frequency mask (optional, usually used as neural network training target)

        if gen_mask:
            mask = []
            for i in range(len(positioned_source_early_reverb)):
                mask.append(self.mask_estimator.get_mask_from_parallel_data(positioned_source_early_reverb[i][:, 0],
                                                                            mixed_noisy[:, 0]))
        else:
            mask = None

        # step 6. normalize the gain of the simulated speech (optional, usually applied if RIR is applied, due to that
        # RIR may change the gain of the speech signal significantly.

        if normalize_gain:
            gain_norm_scale = 0.5 / np.max(np.abs(mixed_noisy))
            mixed_noisy *= gain_norm_scale
            positioned_source_early_reverb = [i*gain_norm_scale for i in positioned_source_early_reverb]
            sent_cfg['gain_norm_scale'] = gain_norm_scale

        if self.DEBUG:
            import matplotlib.pyplot as plt
            n_col = n_source + 1
            for i in range(n_source):
                plt.subplot(2, n_col, i+1)
                imagesc(np.log(np.abs(self.analyzer.analyze(positioned_source_early_reverb[i][:, 0]))),
                                  title='source ' + str(i), show_color_bar=False)
                if mask is not None:
                    plt.subplot(2, n_col, i+n_col+1)
                    imagesc(mask[i], title='source '+str(i), show_color_bar=False)
            plt.subplot(2, n_col, n_col)
            imagesc(np.log(np.abs(self.analyzer.analyze(mixed_noisy[:,0]))), title='mixed', show_color_bar=False)

        return mixed_noisy, positioned_source_early_reverb, mask, sent_cfg

    def sample_sent_cfg(self, sent_cfg=None):
        """ Sample configurations for a sentence and save it in sent_cfg.
        If a sent_cfg is given as input, existing fields will be kept.

        Parameters
        ----------
        sent_cfg: dict
            input configuration for the utterance, usually set to None.

        Returns
        -------
        sent_cfg: dict
            record all the details of the simulation for the current utterance.
        """

        if sent_cfg is None:
            sent_cfg = dict()

        n_noise = get_sample(self.config['dir_noise']['num_dir_noise'])[0]
        sent_cfg['n_dir_noise'] = n_noise
        if self.config['reverb']['use_reverb'] and 'room_size' not in sent_cfg.keys():
            # if use reverb and room_size is not set, re-sample all room related geometry configuration
            reverb_config = self.config['reverb']
            set_config(sent_cfg, 'use_rir_corpus', reverb_config['use_corpus'], overwrite=False)

            if reverb_config['use_corpus'] is False or self._rir_streams_sampler is None:  # sample room and source/noies positions etc
                cnt = 0
                while True:
                    room_size = sample_room(self.config['room'])
                    min_t60 = rirgen.min_t60_of_room(room_size)
                    t60 = get_sample(reverb_config['t60'])
                    # Accept the t60 if it is larger than the minimum t60 for the room.
                    # Note that Sabin's equation is not accurate for room with high absorption coefficients.
                    # So for a big room with very small T60, e.g. 0.1s, the absorption coefficient may have to be
                    # more than 1.0. This is not acceptable and we should avoid having small T60 for large rooms.
                    if t60 > min_t60:
                        sent_cfg['room_size'] = room_size
                        sent_cfg['t60'] = t60
                        break
                    if cnt > 1000 and cnt % 100==0:
                        print("Simulator::sample_sent_cfg: cannot find acceptable room size and T60 time after 1000 trials."
                            "You should probably change room size and T60 distributions.")
                    cnt = cnt + 1

                sent_cfg['array_position'] = sample_array_position(self.config['array'], sent_cfg['room_size'])
                sent_cfg['source_position'] = sample_source_position(self.config['sources'], sent_cfg['room_size'],
                                                                     sent_cfg['array_position']['array_ctr'])
                sent_cfg['n_source'] = sent_cfg['source_position'].shape[1]
                sent_cfg['dir_noise_position'] = sample_source_position_by_random_coordinate(
                    self.config['dir_noise'], n_noise, sent_cfg['room_size'], sent_cfg['array_position']['array_ctr'])

            else:  # sample a pre-computed RIR
                stream_idx = get_sample(self._rir_streams_sampler)
                rir_stream = self.rir_streams[int(stream_idx)]

                n_spk = get_sample(self.config['sources']['num_spk'])[0]
                sent_cfg['n_source'] = n_spk

                rir_wav, sent_cfg['room_size'], sent_cfg['array_position'], positions, sent_cfg['t60'] = \
                    rir_stream.sample_rir(n_spk + n_noise)
                assert rir_wav[0].shape[1] >= self.config['array']['n_mic'], (
                    "Simulator::sample_sent_cfg: number of channels in RIR wav [%d] is smaller than the number of microphones [%d]" % (
                        rir_wav[0].shape[1], self.config['array']['n_mic']))

                sent_cfg['source_rir_wav'] = rir_wav[:n_spk]
                sent_cfg['source_position'] = positions[:, :n_spk]
                sent_cfg['dir_noise_rir_wav'] = rir_wav[n_spk:]
                sent_cfg['dir_noise_position'] = positions[:, n_spk:]
        else:
            sent_cfg['n_source'] = get_sample(self.config['sources']['num_spk'])[0]

        # Sample SPR and SNRs. If already set, do not overwrite these configurations
        if sent_cfg['n_source'] > 1:
            spr = get_sample(self.config['mixing']['spr'], sent_cfg['n_source']-1)
            set_config(sent_cfg, 'spr', spr, overwrite=False)

        global_snr = get_sample(self.config['global_snr'])
        set_config(sent_cfg, 'global_snr', global_snr, overwrite=False)

        if self.config['dir_noise']['use_dir_noise']:
            dir_snr = get_sample(self.config['dir_noise']['snr'], n_sample=sent_cfg['n_dir_noise'])
            set_config(sent_cfg, 'dir_snr', dir_snr, overwrite=False)

        if self.config['array']['n_mic']>1 and self.config['iso_noise']['use_iso_noise']:
            iso_snr = get_sample(self.config['iso_noise']['snr'])
            set_config(sent_cfg, 'iso_snr', iso_snr, overwrite=False)

        return sent_cfg


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
        self.simulator = Simulator(cfg)

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
            source_rir=[source_rir],
            dir_noise_rir=dir_noise_rirs,
            iso_noise_wav=iso_noise_wav,
            gen_mask=gen_mask,
            min_length=min_length,
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
        self.simulator = Simulator(cfg)

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
            min_length=min_length,
            normalize_gain=normalize_gain)

        return mixed_noisy, positioned_source_early_reverb, mask, sent_cfg
