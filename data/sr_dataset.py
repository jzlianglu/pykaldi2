"""
Copyright (c) 2019 Microsoft Corporation. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import sys
import torch.utils.data as data
import glob
import numpy as np
from multiprocessing import Queue
import zipfile
import io
import json 
from simulation import SimpleSimulator, imagesc
from simulation.freq_analysis import stft
import reader
import os


def _utt2seg(data, seg_len, seg_shift):
    """ Cut an utterance (MxN matrix) to segments. """
    if data.ndim == 1:
        data = np.reshape(data, (1, data.size))
    dim, n_fr = data.shape
    n_seg = int(np.floor((n_fr - seg_len) / seg_shift)) + 1
    seg = []
    for i in range(n_seg):
        start = i * seg_shift
        stop = start + seg_len
        seg.append(data[:, start:stop])

    return seg


class DataBuffer:
    """This is the class that generate data for neural network training and used with dynamic data simulation.
    The job is to prepare speech corpus into individual training samples. E.g. feature-label pairs.
    It may also need to call speech simulation to simulated distorted multi-channel array data.
    This class generates samples and put them in a buffer (a FIFO queue). """

    def __init__(self, data_generator, buffer_size=1000, preload_size=100, randomize=True):
        self.data_generator = data_generator
        self.buffer_size = buffer_size
        self.preload_size = preload_size
        self.buffer = []
        self.randomize = randomize

    def get(self):
        """Generate required number of training samples. """
        if len(self.buffer) < self.buffer_size:     # maintain minimum number of entries in the buffer.
            while len(self.buffer) < self.buffer_size + self.preload_size:
                tmp_data = self.data_generator.generate()
                self.buffer += tmp_data

        if self.randomize:
            return_idx = np.random.randint(len(self.buffer))
        else:
            return_idx = 0
        data = self.buffer.pop(return_idx)

        return data

    def get_len(self):
        return self.data_generator.get_len()


class SpeechDataset(data.Dataset):

    def __init__(self, config):
        self.transform=None
        self.sequence_mode = config["data_config"]["sequence_mode"]

        # load the three types of source data
        dir_noise_streams = None
        if "dir_noise_paths" in config:
            dir_noise_streams = self._load_streams(config["dir_noise_paths"], config['data_path'], is_speech=False)

        rir_streams = None
        if "rir_paths" in config:
            rir_streams = self._load_streams(config["rir_paths"], config['data_path'], is_speech=False, is_rir=True)

        source_streams = self._load_streams(config["source_paths"], config['data_path'], is_speech=True)
        self.source_stream_sizes = [i.get_number_of_data() for i in source_streams]
        if self.sequence_mode:
            self.source_stream_cum_sizes = [self.source_stream_sizes[0]]
            for i in range(1, len(self.source_stream_sizes)):
                self.source_stream_cum_sizes.append(self.source_stream_cum_sizes[-1] + self.source_stream_sizes[i])

        generator_config = DataGeneratorSequenceConfig(
            use_reverb=config["data_config"]["use_reverb"],
            use_noise=config["data_config"]["use_dir_noise"],
            snr_range=[config["data_config"]["snr_min"], config["data_config"]["snr_max"]],
            n_hour_per_epoch=config["sweep_size"],
            sequence_mode=self.sequence_mode,
            load_label=config["data_config"]["load_label"],
            seglen=config["data_config"]["seg_len"], 
            segshift=config["data_config"]["seg_shift"],
            use_cmn=config["data_config"]["use_cmn"],
            simulation_prob=config['data_config']['simulation_prob']
        )

        data_generator = DataGeneratorTrain(source_streams, dir_noise_streams, rir_streams, generator_config, DEBUG=False)
        if self.sequence_mode:
            self.data_buffer = data_generator
        else:
            self.data_buffer = DataBuffer(data_generator, buffer_size=20000, preload_size=200, randomize=True)

        self.sample_len_seconds = config["data_config"]["seg_len"] * 0.01 # default sampling rate: 100Hz
        self.stream_idx_for_transform = [0]

    def _load_streams(self, source_list, data_path, is_speech=True, is_rir=False):
        source_streams = list()
        for i in range(len(source_list)):
            corpus_type = source_list[i]['type']
            corpus_wav_path = data_path+source_list[i]['wav']
            label_paths = []
            label_names = []
            if 'label' in source_list[i]:
                label_paths.append(data_path+source_list[i]['label'])
                label_names.append('label')
            else:
                corpus_label_path = None
            if 'aux_label' in source_list[i]:
                label_paths.append(data_path+source_list[i]['aux_label'])
                label_names.append('aux_label')
            else:
                corpus_label_path = None
            print("%s::_load_streams: loading %s from %s..." % (self.__class__.__name__, corpus_type, corpus_wav_path))
            curr_stream = reader.stream.gen_stream_from_zip(corpus_wav_path,
                                                            label_files=label_paths,
                                                            label_names=label_names,
                                                            is_speech_corpus=is_speech,
                                                            is_rir=is_rir,
                                                            get_duration=False,
                                                            corpus_name=corpus_type,
                                                            file_extension='wav')
            source_streams.append(curr_stream)

        return source_streams

    def __getitem__(self, index):
        if self.sequence_mode:
            # find the stream index and utterance index corresponding to the given index
            stream_idx = -1
            for i in range(len(self.source_stream_cum_sizes)):
                if index < self.source_stream_cum_sizes[i]:
                    stream_idx = i
                    break
            if stream_idx == -1:
                raise Exception('index larger than available number of sentences. ')
            if stream_idx==0:
                utt_idx = index
            else:
                utt_idx = index - self.source_stream_cum_sizes[stream_idx-1]
            data = self.data_buffer.generate((stream_idx, utt_idx))[0]
#            data = self.data_buffer.get((stream_idx, utt_idx))
        else:
            data = self.data_buffer.get()

        if self.transform is not None:
            data = self.transform.apply(data, stream_keys=self.stream_idx_for_transform)

        return data

    def sample_in_seconds(self):
        return self.sample_len_seconds

    def __len__(self):
        if self.sequence_mode:
            return np.sum(self.source_stream_sizes)
        else:
            return self.data_buffer.get_len()

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class DataGeneratorSequenceConfig:
    """
    Define the configurations of data generation.
    """
    def __init__(self, use_reverb, use_noise, snr_range, n_hour_per_epoch=10, sequence_mode=False, load_label=True, min_seglen=0, seglen=500, segshift=500, use_cmn=False, gain_norm=False, simulation_prob=0.5):
        self.n_hour_per_epoch = n_hour_per_epoch
        self.load_label = load_label
        self.segment_config = {}
        self.use_reverb = use_reverb
        self.use_noise = use_noise
        self.snr_range = snr_range
        self.segment_config['sequence_mode'] = sequence_mode
        self.segment_config['seglen'] = seglen   # length of segments in terms of frames
        self.segment_config['segshift'] = segshift
        self.segment_config['min_seglen'] = min_seglen
        self.n_segment_per_epoch = int(3600 * n_hour_per_epoch / self.segment_config['seglen'] * 100)
        self.gain_norm = gain_norm
        self.use_cmn = use_cmn
        self.simulation_prob = simulation_prob


class DataGeneratorTrain:
    """
    Generate simulated speech utterances from clean speech streams, noise streams, and rir streams. Responsible for
    sampling of the data from the streams, and call SimpleSimulator to do the simulation. Also responsible for extract
    features and make training samples.
    """
    _window_file = 'mel80_window.txt'  # the file that stores the Mel scale window coefficients

    def __init__(self, source_streams, noise_streams, rir_streams, config, DEBUG=False):
        """
        :param source_streams: a list of SpeechDataStream objects, containing the clean speech source files names and
        meta-data such as label, utterance ID, and speaker ID.
        :param noise_streams: a list of DataStream objects, containing noise file names.
        :param rir_streams: a list of RIRDataStream objects, containing RIR file names and meta data information.
        :param config: an object of type DataGeneratorSequenceConfig
        :param DEBUG: if set to DEBUG mode, will plot the filterbanks and label.
        """
        self._source_streams = source_streams
        self._source_streams_prior = self._get_streams_prior(source_streams)
        self._rir_streams = rir_streams
        self._rir_streams_prior = self._get_streams_prior(rir_streams)
        self._noise_streams = noise_streams
        self._noise_streams_prior = self._get_streams_prior(noise_streams)

        self._data_len = config.n_segment_per_epoch
        self._single_source_simulator = SimpleSimulator(use_rir=config.use_reverb, use_noise=config.use_noise, snr_range=config.snr_range)
        self._config = config
        self._DEBUG = DEBUG
        self._gen_window()

    def _get_streams_prior(self, streams):
        if streams is None:
            return None
        else:
            n_entrys = np.asarray([stream.get_number_of_data() for stream in streams])
            return n_entrys / np.sum(n_entrys)

    def _gen_window(self):
        # load the pre-computed window coefficients for 80D log filterbanks used in typical acoustic modeling.
        # the window is computed by the following code
        # import librosa
        # self._window = librosa.filters.mel(16000, 512, n_mels=80, fmax=7690, htk=True)
        mel_file = os.path.join(os.path.dirname(__file__), self._window_file)
        with open(mel_file) as file:
            lines = [line.rstrip('\n') for line in file]
        self._window = np.vstack([np.asarray([np.float32(j) for j in i.split(",")]) for i in lines])

    def _logfbank_extractor(self, wav):
        # typical log fbank extraction for 16kHz speech data
        preemphasis = 0.96

        t1 = np.sum(self._window, 0)
        t1[t1 == 0] = -1
        inv = np.diag(1 / t1)
        mel = self._window.dot(inv).T

        wav = wav[1:] - preemphasis * wav[:-1]
        S = stft(wav, n_fft=512, hop_length=160, win_length=400, window=np.hamming(400), center=False).T

        spec_mag = np.abs(S)
        spec_power = spec_mag ** 2
        fbank_power = spec_power.T.dot(mel * 32768 ** 2) + 1
        log_fbank = np.log(fbank_power)

        return log_fbank

    def generate(self, index=None):
        """

        :param index: a tuple of 2 entries (source_stream_idx, utt_idx) that specifies which clean source file to use
        for simulation. If not provided, will randomly choose one clean source file from the clean source streams.
        :return: a list of training samples
        """
        seg_len = self._config.segment_config['seglen']
        seg_shift = self._config.segment_config['segshift']

        if index is None:       # if no index is given, let the simulator do random sampling
            # sample a clean speech stream
            source_stream_idx = np.random.choice(np.arange(len(self._source_streams)), replace=True, p=self._source_streams_prior)
            # sample a clean speech utterance
            _, utt_id, source_wav, _ = self._source_streams[source_stream_idx].sample_spk_and_utt(n_spk=1,
                                                                                                  n_utt_per_spk=1,
                                                                                                  load_data=True)
        else:    # if index is given, use the specified sentence
            assert len(index) == 2
            source_stream_idx = index[0]
            utt_id = [self._source_streams[source_stream_idx].utt_id[index[1]]]
            _, _, source_wav, _ = self._source_streams[source_stream_idx].read_utt_with_id(utt_id, load_data=True)

        if np.random.random() > self._config.simulation_prob:
            simulated_wav = source_wav[0]
        else:
            if self._noise_streams is None:
                noise_wavs = None
            else:
                noise_stream_idx = np.random.choice(np.arange(len(self._noise_streams)), replace=True,
                                                p=self._noise_streams_prior)
                noise_wavs, noise_files = self._noise_streams[noise_stream_idx].sample_data()
    
            if self._rir_streams is None:
                source_rir = None
                noise_rirs = None
            else:
                rir_stream_idx = np.random.choice(np.arange(len(self._rir_streams)), replace=True, p=self._rir_streams_prior)
                n_rir = 1 if noise_wavs is None else 1+len(noise_wavs)
                rir_wav, room_size, array_position, positions, t60 = self._rir_streams[rir_stream_idx].sample_rir(n_rir)
                source_rir = rir_wav[0]
                noise_rirs = rir_wav[1:]
    
            simulated_wav, _, mask, config = self._single_source_simulator(source_wav[0],
                                                                           dir_noise_wavs=noise_wavs,
                                                                           source_rir=source_rir,
                                                                           dir_noise_rirs=noise_rirs,
                                                                           gen_mask=False, normalize_gain=self._config.gain_norm)

        fbank = self._logfbank_extractor(simulated_wav[:,0])

        if self._config.load_label:
            _, label = self._source_streams[source_stream_idx].read_label_with_id(utt_id)

            frame_label = label['label'][0].T
            if 'aux_label' in label:
                aux_label = label['aux_label']
            else:
                aux_label = np.zeros((1,1))
 
            if np.abs(frame_label.shape[0] - fbank.shape[0])>5:
                print("DataGeneratorTrain::generate: Warning: filterbank and label have significantly different number of frames. ")
            
            n_fr = np.minimum(frame_label.shape[0], fbank.shape[0])
            frame_label = frame_label[:n_fr,:]
            fbank = fbank[:n_fr,:]

        if self._config.use_cmn:
            fbank = reader.preprocess.cmn(fbank, axis=0)

        if self._config.segment_config['sequence_mode']:
            if self._config.load_label:
                train_samples = [(fbank, utt_id, frame_label, aux_label)]
            else:
                train_samples = [(fbank, utt_id)]
        else: 
            fbank_seg = _utt2seg(fbank.T, seg_len, seg_shift)
            if len(fbank_seg) == 0:
                return []

            if self._config.load_label:
                label_seg = _utt2seg(frame_label.T, seg_len, seg_shift)
                train_samples = [(fbank_seg[i].T, utt_id, label_seg[i].T) for i in range(len(label_seg))]
            else:
                train_samples = [(fbank_seg[i].T, utt_id) for i in range(len(fbank_seg))]
        
            if self._DEBUG:
                import matplotlib.pyplot as plt
                n_sample = len(train_samples)
                for i in range(n_sample):
                    plt.subplot(n_sample,2,i*2+1)
                    imagesc(train_samples[i][0].T)
                    plt.subplot(n_sample,2,i*2+2)
                    plt.plot(train_samples[i][2])

        return train_samples

    def get_len(self):
        return self._data_len
