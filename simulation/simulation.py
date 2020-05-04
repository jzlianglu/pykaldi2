import numpy as np
import json
import os
import scipy
import sys
sys.path.append('..')

from . import rirgen
from . import farfield_simulator as iso_noise_simulator
from .distortor import Distortor
from .mixer import Mixer
from .mask import MaskEstimator
from utils import utils as sig_utils
from feature import feature


class Simulator:
    """
    A versatile room simulation class for generating reverberant and noisy speech signals that may be overlapped.
    Supports both single and multi-channel simulation, and single and multi-source simulation.

    Attributes
    ----------
    config : dict
        a dictionary that defines various settings of the simulation. The dict should be generating from a function
        from config.py.
    speech_streams : list
        A list of SpeechDataStream objects, each defines a speech data corpus.
    noise_streams : list
        A list of DataStream objects, each defines a noise corpus. Can be set to None if not using noise.
    rir_streams : list
        A list of RIRStream objects, each defines a room impulse response (RIR) corpus. Can be set to None if not using
        RIR.
    iso_noise_streams:
        A list of DataStream objects, each defines an isotropic noise corpus that will be used in multi-channel
        simulation. Can be set to None if not using noise or multi-channel simulation.

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(self, config, speech_streams, noise_streams=None, rir_streams=None, iso_noise_streams=None, DEBUG=False):
        self.load_config(config)
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
        self.speech_streams_sampler = self.gen_stream_sampler(self.speech_streams, 'Speech streams sampler')
        self.noise_streams_sampler = self.gen_stream_sampler(self.noise_streams, 'Directional noise streams sampler')
        self.rir_streams_sampler = self.gen_stream_sampler(self.rir_streams, 'RIR streams sampler')
        self.iso_streams_sampler = self.gen_stream_sampler(self.iso_noise_streams, 'Isotropic noise streams sampler')

    def gen_stream_sampler(self, streams, name):
        if streams is not None:
            n_entrys = [stream.get_number_of_data() for stream in streams]
            n_entrys = np.asarray(n_entrys)
            print(n_entrys) 
            pmf = n_entrys / np.sum(n_entrys)
            streams_sampler = sig_utils.get_distribution_template(name, category=np.arange(len(streams)), pmf=pmf, distribution='discrete')
        else:
            streams_sampler = None
        return streams_sampler

    def load_config(self, config):
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = json.load(open(config))
        # check config
        if not self.config['dir_noise']['use_dir_noise']:
            self.config['dir_noise']['num_dir_noise']['max'] = 0
            self.config['dir_noise']['num_dir_noise']['min'] = 0

        self.config['array']['n_mic'] = len(config['array']['mic_positions'][0])

    def sample_room(self, config):
        room = np.zeros((3,))
        room[0] = sig_utils.get_sample(config['length'], n_sample=1)[0]
        room[1] = sig_utils.get_sample(config['width'], n_sample=1)[0]
        room[2] = sig_utils.get_sample(config['height'], n_sample=1)[0]
        return room

    def sample_array_position(self, config, room_size, use_gaussian=False):
        if config['geometry'] == "princeton":
            pass
            #all_mic = self.array.gen_mic_position_wrt_center()
        elif config['geometry'] == "single_channel":
            all_mic = np.zeros((3,1))
        else:
            raise Exception("Unknown array geometry")

        array_ctr = np.zeros((3,))
        if use_gaussian:
            name = ['length_ratio', 'width_ratio', 'height_ratio']
            for i in range(len(name)):
                boundary = np.asarray([config[name[i]][j] * room_size[i] for j in ['min', 'max']])
                sampler = sig.signal.sampling.GaussianSampler(mean=np.mean(boundary), std=(boundary[1]-boundary[0])/6, min=boundary[0], max=boundary[1])
                array_ctr[i] = sampler.get_sample()[0]
                array_ctr = array_ctr.reshape(3, 1)

        else:
            array_ctr[0] = sig_utils.get_sample(config['length_ratio'], n_sample=1)[0]
            array_ctr[1] = sig_utils.get_sample(config['width_ratio'], n_sample=1)[0]
            array_ctr[2] = sig_utils.get_sample(config['height_ratio'], n_sample=1)[0]
            array_ctr = array_ctr.reshape(3, 1) * room_size.reshape(3, 1)

        array = {}
        array['array_ctr'] = array_ctr
        array['mic_position'] = array['array_ctr'] + all_mic

        return array

    def sample_source_position(self, config, room_size, array_center):
        # sample the number of speakers
        n_spk = sig_utils.get_sample(config['num_spk'])
        n_spk = n_spk[0]

        if config['position_scheme'] == "minimum_angle":
            source_position,_ = self.sample_source_position_by_min_angle(config, n_spk, room_size, array_center)
        elif config['position_scheme'] == "random_coordinate":
            source_position = self.sample_source_position_by_random_coordinate(config, n_spk, room_size, array_center)
        else:
            raise Exception("Unknown speech source position scheme: %s" % config['position_scheme'])

        return source_position

    def sample_source_position_by_random_coordinate(self, config, n_spk, room_size, array_center, forbidden_rect=None):
        source_position = np.zeros((3,n_spk))

        d_from_wall = config['min_dist_from_wall'] if "min_dist_from_wall" in config.keys() else [0.0, 0.0]
        d_from_array = config['min_dist_from_array'] if "min_dist_from_array" in config.keys() else 0.1
        d_from_other = config['min_dist_from_other'] if "min_dist_from_other" in config.keys() else 0.2
        x_distribution = sig_utils.get_distribution_template("comment", min=d_from_wall[0], max=room_size[0]-d_from_wall[0])
        y_distribution = sig_utils.get_distribution_template("comment", min=d_from_wall[0], max=room_size[1]-d_from_wall[1])
        if "height" in config.keys():
            z_distribution = config['height']
        else:
            z_distribution = sig_utils.get_distribution_template("comment", min=0.0, max=room_size[2])

        for i in range(n_spk):
            cnt = 0
            while 1:
                x = sig_utils.get_sample(x_distribution)
                y = sig_utils.get_sample(y_distribution)
                z = sig_utils.get_sample(z_distribution)
                curr_pos = np.asarray([x, y, z])
                if sig_utils.euclidean_distance(curr_pos[:2], array_center[:2]) >= d_from_array:
                    if forbidden_rect is None or (np.prod(curr_pos[0]-forbidden_rect[0,:])>0 or np.prod(curr_pos[1]-forbidden_rect[1,:])>0):
                        if i==0 or (sig_utils.euclidean_distance(curr_pos[:2], source_position[:2,:i]) >= d_from_other).all():
                            source_position[:,i] = curr_pos[:,0]
                            break
                if cnt > 1000:
                    raise Exception("Maximum number (1000) of trial finished but still not able to find acceptable position for speaker position. ")

        return source_position

    def sample_source_position_in_meeting_room(self, config, n_spk, room_size, array_center, table, seat_circle):
        source_position = np.zeros((3,n_spk))

        d_from_wall = config['min_dist_from_wall'] if "min_dist_from_wall" in config.keys() else [0.0, 0.0]
        d_from_array = config['min_dist_from_array'] if "min_dist_from_array" in config.keys() else 0.1
        d_from_other = config['min_dist_from_other'] if "min_dist_from_other" in config.keys() else 0.2
        if "height" in config.keys():
            z_distribution = config['height']
        else:
            z_distribution = sig_utils.get_distribution_template("comment", min=0.0, max=room_size[2])

        seat_sampler = sig.signal.sampling.AroundRectangleSampler2D(room_size[:2], seat_circle, table)

        for i in range(n_spk):
            cnt = 0
            while 1:
                xy = seat_sampler.get_sample()[0]
                z = sig_utils.get_sample(z_distribution)
                curr_pos = np.hstack([np.asarray(xy), z]).reshape((3,1))
                if sig_utils.euclidean_distance(curr_pos[:2], array_center[:2]) >= d_from_array:
                    if i==0 or (sig_utils.euclidean_distance(curr_pos[:2], source_position[:2,:i]) >= d_from_other).all():
                        source_position[:,i] = curr_pos[:,0]
                        break
                if cnt > 1000:
                    raise Exception("Maximum number (1000) of trial finished but still not able to find acceptable position for speaker position. ")

        return source_position

    def sample_source_position_by_min_angle(self, config, n_spk, room_size, array_center):
        all_v = []

        for i in range(n_spk):
            cnt = 0
            while True:
                this_v = np.random.uniform(0, 359, size=1)[0]
                L = [this_v - x for x in all_v]
                if len(L) == 0:
                    all_v.append(this_v)
                    break
                elif np.min(np.abs(L)) > float(config['between_source_angle']['min']):
                    all_v.append(this_v)
                    break
                if cnt>1000:
                    print("Simulator::sample_source_position_by_min_angle: Warning: still cannot find acceptable source positions after 1000 trials.")
                cnt +=1

        all_v = np.asarray(all_v) / 180 * np.pi
        all_h = sig_utils.get_sample(config['height'], n_sample=n_spk)

        r = np.zeros((n_spk,))
        for i in range(n_spk):
            ryp, rym = self.find_ry(room_size[1], array_center[1], all_v[i])
            rxp, rxm = self.find_rx(room_size[0], array_center[0], all_v[i])

            rp = np.min([ryp, rxp])
            r[i] = np.random.uniform(0.1 * rp, 0.9 * rp, size=1)[0]

        all_source = np.zeros((n_spk, 3))

        for i in range(n_spk):
            all_source[i, 0] = r[i] * np.cos(all_v[i]) + array_center[0]
            all_source[i, 1] = r[i] * np.sin(all_v[i]) + array_center[1]

        all_source[:, 2] = all_h

        return all_source, [all_v, all_h, r]

    def find_ry(self, R, M, angle):
        if angle >= 0 and angle < np.pi / 2:
            ryp = (R - M) / np.sin(angle)
            rym = (M) / np.sin(angle)
        if angle >= np.pi / 2 and angle < np.pi:
            ryp = (R - M) / np.sin(np.pi - angle)
            rym = (M) / np.sin(np.pi - angle)
        if angle >= np.pi and angle < np.pi / 2 * 3:
            ryp = (M) / np.sin(angle - np.pi)
            rym = (R - M) / np.sin(angle - np.pi)
        if angle >= np.pi / 2 * 3 and angle < 2 * np.pi:
            ryp = M / np.sin(2 * np.pi - angle)
            rym = (R - M) / np.sin(2 * np.pi - angle)

        return ryp, rym

    def find_rx(self, R, M, angle):
        if angle >= 0 and angle < np.pi / 2:
            ryp = (R - M) / np.cos(angle)
            rym = (M) / np.cos(angle)
        if angle >= np.pi / 2 and angle < np.pi:
            ryp = (-M) / np.cos(angle)
            rym = (R - M) / np.cos(np.pi - angle)
        if angle >= np.pi and angle < np.pi / 2 * 3:
            ryp = (M) / np.cos(angle - np.pi)
            rym = (R - M) / np.cos(angle - np.pi)
        if angle >= np.pi / 2 * 3 and angle < 2 * np.pi:
            ryp = (R - M) / np.cos(2 * np.pi - angle)
            rym = (M) / np.cos(2 * np.pi - angle)
        return ryp, rym

    def sample_noise_position(self, config, room_size, array_center):
        # sample the number of directional noises
        n_noise = sig_utils.get_sample(config['num_dir_noise'])
        n_noise = n_noise[0]

        source_position = self.sample_source_position_by_random_coordinate(config, n_noise, room_size, array_center)
        return source_position

    def get_rir(self, source_position, mic_position, room_size, t60):
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

    def get_isotropic_noise(self, mic_position, n_sample, config, fs=16000):
        """mic_position is Nx3, where N is the number of microphones"""
        noise = iso_noise_simulator.generate_isotropic_noise(mic_position, n_sample, fs, type=config['type'], spectrum=config['spectrum_type'])
        return noise.T

    def sample_room_and_get_rir(self, sent_config=None):
        """sample a room and compute RIR.
        This is useful if we want to pre-compute RIR and save it to files to save time in online simulation.
        Note that we only get source RIR. It's usually good to have large number of candidate source RIRs, e.g. 10,
        so we can sample the source RIRs in the online simulation. """
        if sent_config is None:
            sent_config = self.sample_sentence_config()

        source_rir = self.get_rir(sent_config['source_position'], sent_config['array_position']['mic_position'],
                                  sent_config['room_size'], sent_config['t60'])

        return source_rir, sent_config

    def gen_rir(self, sent_config=None):
        """sample a room and compute RIR.
        This is useful if we want to pre-compute RIR and save it to files to save time in online simulation.
        Note that we only get source RIR. It's usually good to have large number of candidate source RIRs, e.g. 10,
        so we can sample the source RIRs in the online simulation. """
        sent_config = self.sample_sentence_config()
        source_rir = self.get_rir(sent_config['source_position'], sent_config['array_position']['mic_position'],
                                  sent_config['room_size'], sent_config['t60'])

        return source_rir, sent_config

    def simulate_given_utt_id(self, source_stream_idx, sent_config=None, gen_mask=False, spk_id=None, utt_id=None, unwanted_utt_id=None, min_length=None):
        """Sometimes, we want to simulate with a specific speaker or utterances.
        if spk_id is not given, utt_id must be given
        if utt_id is given, use it
        if utt_id is not given, but unwanted_utt_id is given, sample sentences excluding unwanted utterances.
        """
        if spk_id is None and utt_id is None:
            raise Exception("Simulator::simulate_given_utt_id: spk_id and utt_id cannot be both None. ")

        sent_config = self.sample_sentence_config(sent_config)
        sent_config['source_stream_idx'] = source_stream_idx

        # sample utt_id that satisfy requirements
        if utt_id is not None:
            sent_config['source_utt_id'] = utt_id
        else:
            tmp = self.speech_streams[source_stream_idx].sample_utt_from_spk(spk_id, unwanted_utt_id=unwanted_utt_id, replace=False, load_data=False, load_vad=False, min_length=min_length)
            sent_config['source_utt_id'] = tmp[1]

        sent_config['source_speakers'] = [self.speech_streams[source_stream_idx].utt2spk[i] for i in sent_config['source_utt_id']]

        return self.simulate(sent_config, gen_mask=gen_mask, min_length=min_length)

    def simulate(self, sent_config=None, gen_mask=False, min_length=None, normalize_gain=True):
        """simulate one utterance.

        Parameters
        ----------
        sent_config: dict
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
        sent_config: dict
            record all the details of the simulation for the current utterance.
        """

        # sample configuration for the sentence. Note that those settings already in sent_config will be kept.
        sent_config = self.sample_sentence_config(sent_config)

        # Sample clean sources
        n_source = sent_config['n_source']
        if 'source_stream_idx' not in sent_config.keys():
            sent_config['source_stream_idx'] = sig_utils.get_sample(self.speech_streams_sampler)[0]
        if 'source_utt_id' not in sent_config.keys() or 'source_speakers' not in sent_config.keys():
            sent_config['source_speakers'], sent_config['source_utt_id'], source_wav, source_vad = \
                self.speech_streams[sent_config['source_stream_idx']].sample_spk_and_utt(n_spk=n_source, n_utt_per_spk=1, load_data=True, load_vad=True, min_length=min_length)
        else:
            _, _, source_wav, source_vad = self.speech_streams[sent_config['source_stream_idx']].read_utt_with_id(
                sent_config['source_utt_id'], load_data=True, load_vad=True)

        n_dir_noise = sent_config['n_dir_noise']
        if n_dir_noise > 0 and self.noise_streams_sampler is not None:
            sent_config['noise_stream_idx'] = sig_utils.get_sample(self.noise_streams_sampler)[0]
            dir_noise_wav, dir_noise_files = self.noise_streams[sent_config['noise_stream_idx']].sample_data(n_data=n_dir_noise)

        # Load or generate RIR
        if self.config['reverb']['use_reverb']:
            if sent_config['use_rir_corpus']:
                source_rir = sent_config['source_rir_wav']
                if sent_config['dir_noise_position'].size > 0:
                    dir_noise_rir = sent_config['dir_noise_rir_wav']
            else:
                source_rir = self.get_rir(sent_config['source_position'], sent_config['array_position']['mic_position'], sent_config['room_size'], sent_config['t60'])
                if sent_config['dir_noise_position'].size>0:
                    dir_noise_rir = self.get_rir(sent_config['dir_noise_position'], sent_config['array_position']['mic_position'], sent_config['room_size'], sent_config['t60'])

            # Apply RIRs to sources and directional noises
            source_reverb, source_early_reverb = self.apply_rir(source_wav, source_rir, get_early_reverb=True)
            if n_dir_noise > 0:
                dir_noise_reverb, dir_noise_early_reverb = self.apply_rir(dir_noise_wav, dir_noise_rir, get_early_reverb=False)
        else:
            source_reverb, source_early_reverb = source_wav, source_wav
            if n_dir_noise > 0:
                dir_noise_reverb, dir_noise_early_reverb = dir_noise_wav, dir_noise_wav


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
                source_mixer.mix_signals(source_reverb, sent_config['spr'], signal2=source_early_reverb)

        # Mix the combined sources with directional noise and isotropic noise using the SNR
        mixed_noisy=mixed
        for i in range(n_dir_noise):
            tmp_mixed_noisy,scaled_dir_noise = Distortor.add_noise(mixed, dir_noise_reverb[i], sent_config['dir_snr'][i], noise_position_scheme='sample_noise')
            mixed_noisy += scaled_dir_noise

        n_sample, n_ch = mixed_noisy.shape
        if self.config['iso_noise']['use_iso_noise']:
            if self.config['iso_noise']['use_corpus']:
                sent_config['iso_noise_stream_idx'] = sig_utils.get_sample(self.iso_streams_sampler)[0]
                iso_noise, iso_noise_files = self.iso_noise_streams[sent_config['iso_noise_stream_idx']].sample_data()
                iso_noise = iso_noise[0]
                if iso_noise.shape[1]>1 and n_ch==1:
                    iso_channel_idx = np.random.randint(0, high=iso_noise.shape[1])
                    iso_noise = iso_noise[:,[iso_channel_idx]]
                else:
                    assert iso_noise.shape[1]==n_ch, "Simulator::Simulate: isotropic noise's number of channels (%d) does not match the requirement (%d)" % (iso_noise.shape[1], n_ch)
            else:
                if n_ch == 1:  # for single channel simulation, we add stationary white noise or provided noise
                    iso_noise = np.random.normal(size=(n_sample, n_ch))
                else:
                    iso_noise = self.get_isotropic_noise(sent_config['array_position']['mic_position'].T, n_sample,
                                                             self.config['iso_noise'], fs=self.config['analysis']['fs'])

            tmp_mixed_noisy, scaled_iso_noise = Distortor.add_noise(mixed, iso_noise, sent_config['iso_snr'],
                                                                    noise_position_scheme='repeat_noise')
            mixed_noisy += scaled_iso_noise

        # Generate mask
        if gen_mask:
            mask = []
            for i in range(len(positioned_source_early_reverb)):
                mask.append(self.mask_estimator.get_mask_from_parallel_data(positioned_source_early_reverb[i][:,0], mixed_noisy[:,0]))
        else:
            mask = None

        if self.DEBUG:
            import matplotlib.pyplot as plt
            n_col = n_source + 1
            for i in range(n_source):
                plt.subplot(2, n_col, i+1)
                sig_utils.imagesc(np.log(np.abs(self.analyzer.analyze(positioned_source_early_reverb[i][:,0]))), title='source '+str(i), show_color_bar=False)
                plt.subplot(2, n_col, i+n_col+1)
                sig_utils.imagesc(mask[i], title='source '+str(i), show_color_bar=False)
            plt.subplot(2, n_col, n_col)
            sig_utils.imagesc(np.log(np.abs(self.analyzer.analyze(mixed_noisy[:,0]))), title='mixed', show_color_bar=False)
            #plt.show()
            #wait = input("PRESS ENTER TO CONTINUE.")


        if normalize_gain:
            gain_norm_scale = 0.5 / np.max(mixed_noisy)
            mixed_noisy *= gain_norm_scale
            positioned_source_early_reverb = [i*gain_norm_scale for i in positioned_source_early_reverb]
            sent_config['gain_norm_scale'] = gain_norm_scale

        return mixed_noisy, positioned_source_early_reverb, mask, sent_config

    def sample_sentence_config(self, sent_config=None):
        """ Sample configurations for a sentence and save it in sent_config.
        If a sent_config is given as input, existing fields will be kept.

        Parameters
        ----------
        sent_config: dict
            input configuration for the utterance, usually set to None.

        Returns
        -------
        sent_config: dict
            record all the details of the simulation for the current utterance.
        """

        if sent_config is None:
            sent_config = dict()

        if self.config['reverb']['use_reverb']:
            if 'room_size' not in sent_config.keys():
                # if room_size is not set, re-sample all room related geometry configuration
                reverb_config = self.config['reverb']
                sig_utils.set_config(sent_config, 'use_rir_corpus', reverb_config['use_corpus'], overwrite=False)

                if reverb_config['use_corpus'] is False or self.rir_streams_sampler is None:  # sample room and source/noies positions etc
                    cnt = 0
                    while True:
                        # sample a room
                        sent_config['room_size'] = self.sample_room(self.config['room'])
                        # sample T60
                        min_t60 = rirgen.min_t60_of_room(sent_config['room_size'])

                        t60 = sig_utils.get_sample(reverb_config['t60'])
                        # accept the t60 if it is larger than the minimum t60 for the room.
                        # Note that Sabin's equation is not accurate
                        # for room with high absorption coefficients. So for a big room with very small T60, e.g. 0.1s, the absorption
                        # coefficient may have to be more than 1.0. This is not acceptable and we should avoid having small T60 for large rooms.
                        if t60 > min_t60:
                            sent_config['t60'] = t60
                            break
                        if cnt>1000:
                            print("Simulator::sample_sentence_config: Warning: still cannot find acceptable room size and T60 time after 1000 trials."
                                  "You should probably change your room size distribution and T60 distribution.")
                        cnt = cnt + 1

                    # sample array location
                    sent_config['array_position'] = self.sample_array_position(self.config['array'],
                                                                               sent_config['room_size'])

                    # sample source locations
                    sent_config['source_position'] = self.sample_source_position(self.config['sources'],
                                                                                 sent_config['room_size'],
                                                                                 sent_config['array_position']['array_ctr'])
                    sent_config['n_source'] = sent_config['source_position'].shape[1]

                    # sample noise locations
                    sent_config['dir_noise_position'] = self.sample_noise_position(self.config['dir_noise'],
                                                                                   sent_config['room_size'],
                                                                                   sent_config['array_position'][
                                                                                       'array_ctr'])
                    sent_config['n_dir_noise'] = sent_config['dir_noise_position'].shape[1]

                else:  # sample a pre-computed RIR
                    stream_idx = sig_utils.get_sample(self.rir_streams_sampler)
                    rir_stream = self.rir_streams[int(stream_idx)]

                    n_spk = sig_utils.get_sample(self.config['sources']['num_spk'])[0]
                    sent_config['n_source'] = n_spk
                    n_noise = sig_utils.get_sample(self.config['dir_noise']['num_dir_noise'])[0]
                    sent_config['n_dir_noise'] = n_noise

                    rir_wav, sent_config['room_size'], sent_config['array_position'], positions, sent_config['t60'] = rir_stream.sample_rir(n_spk + n_noise)
                    assert rir_wav[0].shape[1] >= self.config['array']['n_mic'], (
                    "Simulator::sample_sentence_config: number of channels in RIR wav [%d] is smaller than the number of microphones [%d]" % (
                    rir_wav[0].shape[1], self.config['array']['n_mic']))

                    sent_config['source_rir_wav'] = rir_wav[:n_spk]
                    sent_config['source_position'] = positions[:, :n_spk]
                    sent_config['dir_noise_rir_wav'] = rir_wav[n_spk:]
                    sent_config['dir_noise_position'] = positions[:, n_spk:]
        else:
            sent_config['n_source'] = sig_utils.get_sample(self.config['sources']['num_spk'])[0]
            sent_config['n_dir_noise'] = sig_utils.get_sample(self.config['dir_noise']['num_dir_noise'])[0]

        # Sample SPR and SNRs. If already set, do not overwrite these configurations
        if sent_config['n_source'] > 1:
            spr = sig_utils.get_sample(self.config['mixing']['spr'], sent_config['n_source']-1)
            sig_utils.set_config(sent_config, 'spr', spr, overwrite=False)

        global_snr = sig_utils.get_sample(self.config['global_snr'])
        sig_utils.set_config(sent_config, 'global_snr', global_snr, overwrite=False)

        if self.config['dir_noise']['use_dir_noise']:
            dir_snr = sig_utils.get_sample(self.config['dir_noise']['snr'], n_sample=sent_config['n_dir_noise'])
            sig_utils.set_config(sent_config, 'dir_snr', dir_snr, overwrite=False)

        if self.config['array']['n_mic']>1 and self.config['iso_noise']['use_iso_noise']:
            iso_snr = sig_utils.get_sample(self.config['iso_noise']['snr'])
            sig_utils.set_config(sent_config, 'iso_snr', iso_snr, overwrite=False)

        return sent_config

    def apply_rir(self, source_wav, source_rir, get_early_reverb=True):
        """Apply RIR on sources"""
        reverb = []
        early_reverb = []
        for i in range(len(source_rir)):
            tmp_reverb, tmp_early_reverb = Distortor.apply_rir(source_wav[i], source_rir[i], get_early_reverb=get_early_reverb)
            reverb.append(tmp_reverb)
            early_reverb.append(tmp_early_reverb)

        return reverb, early_reverb


def simulate(source_streams, dir_noise_streams=None, rir_streams=None, iso_noise_streams=None, gen_mask=False,
             geometry='single_channel', n_utt_to_simulate=1, n_source=[1, 1], n_dir_noise=[1, 2], snr=[0, 20],
             t60=[0.1, 0.5], DEBUG=False):
    cfg_ms = sig.signal.simulation.config.gen_default_config(geometry=geometry)
    cfg_ms['global_snr']['max'] = np.max(snr)
    cfg_ms['global_snr']['min'] = np.min(snr)
    cfg_ms['sources']['num_spk']['max'] = np.max(n_source)
    cfg_ms['sources']['num_spk']['min'] = np.min(n_source)
    cfg_ms['iso_noise']['use_iso_noise'] = True
    cfg_ms['iso_noise']['use_corpus'] = iso_noise_streams is not None
    cfg_ms['dir_noise']['use_dir_noise'] = dir_noise_streams is not None
    cfg_ms['dir_noise']['num_dir_noise']['max'] = np.max(n_dir_noise)
    cfg_ms['dir_noise']['num_dir_noise']['min'] = np.min(n_dir_noise)
    cfg_ms['reverb']['use_reverb'] = True
    cfg_ms['reverb']['use_corpus'] = rir_streams is not None
    cfg_ms['reverb']['t60']['max'] = np.max(t60)
    cfg_ms['reverb']['t60']['min'] = np.min(t60)

    simulator = sig.signal.simulation.Simulator(cfg_ms, source_streams, noise_streams=dir_noise_streams,
                                                rir_streams=rir_streams, iso_noise_streams=iso_noise_streams)

    analyzer = sig.signal.feature.SpectrumAnalyzer()

    all_simulated = dict()
    config = dict()
    for n in range(n_utt_to_simulate):
        simulated = simulator.simulate(gen_mask=gen_mask)
        simulated = list(simulated)
        for i in range(2):
            simulated[i] = sig.signal.utils.convert_data_precision(simulated[i], 'int16')
        if simulated[2] is not None:
            for i in range(len(simulated[2])):
                simulated[2][i] = simulated[2][i].astype(np.bool)

        utt_id = 'utt_' + str(n)

        # also store the speaker information for fast indexing when loading the database
        utt_config = dict()
        utt_config['source_speakers'] = simulated[3]['source_speakers']
        utt_config['source_utt_id'] = simulated[3]['source_utt_id']
        utt_config['global_snr'] = simulated[3]['global_snr']
        utt_config['t60'] = simulated[3]['t60']
        utt_config['array_position'] = simulated[3]['array_position']
        utt_config['source_position'] = simulated[3]['source_position']
        utt_config['dir_noise_position'] = simulated[3]['dir_noise_position']
        utt_config['room_size'] = simulated[3]['room_size']
        config[utt_id] = utt_config
        all_simulated[utt_id] = simulated

        if DEBUG:
            import matplotlib.pyplot as plt
            n_source = len(simulated[1])
            plt.figure(1)
            plt.plot(simulated[0])
            for i in range(n_source):
                plt.plot(simulated[1][i])
            plt.figure(2)
            plt.subplot(2, (n_source + 1), 1)
            sig.signal.utils.imagesc(np.maximum(0, analyzer.log_spec(simulated[0][:,0:1])), show_color_bar=True)
            for i in range(n_source):
                plt.subplot(2, (n_source + 1), i + 2)
                sig.signal.utils.imagesc(np.maximum(0, analyzer.log_spec(simulated[1][i][:,0:1])), show_color_bar=True)
                if simulated[2] is not None:
                    plt.subplot(2, (n_source + 1), i + n_source + 3)
                    sig.signal.utils.imagesc(simulated[2][i].astype(np.float32), show_color_bar=True)

        sig_utils.print_progress(n+1, n_utt_to_simulate, 1)

    return all_simulated, config
