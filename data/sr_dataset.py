#import pickle
import sys
import torch.utils.data as data
import glob
import numpy as np
from multiprocessing import Queue
import zipfile
import io
import json 

import simulation as simu
import reader
import feature
from utils import utils

class SpeechDataset(data.Dataset):

    def __init__(self, config, transform=None):
        self.transform=transform
        self.min_len_wav = (config["data_config"]["seg_len"]-1) * config["data_config"]["frame_shift"] + config["data_config"]["frame_len"]
        self.sequence_mode = config["data_config"]["sequence_mode"]

        # IO layer: set up the data sources

        # load the three types of source data
        dir_noise_streams = None
        rir_streams = None
        source_streams = None
        if "dir_noise_paths" in config:
            dir_noise_streams = self.load_streams(config["dir_noise_paths"], 
                                                  is_speech=False) 

        if "rir_paths" in config:
            rir_streams = self.load_streams(config["rir_paths"], 
                                           is_speech=False, 
                                           is_rir=True)

        source_streams = self.load_streams(config["source_paths"], 
                                           is_speech=True) 
      
        self.source_stream_sizes = [i.get_number_of_data() for i in source_streams]

        if self.sequence_mode:
            self.source_stream_cum_sizes = [self.source_stream_sizes[0]]
            for i in range(1, len(self.source_stream_sizes)):
                self.source_stream_cum_sizes.append(self.source_stream_cum_sizes[-1] + self.source_stream_sizes[i])

        # Simulation layer: set up the simulator
        # get the single channel single source simulation configuration
        use_reverb = config["data_config"]["use_reverb"]
        use_noise = config["data_config"]["use_dir_noise"]
        snr_range = [config["data_config"]["snr_min"], config["data_config"]["snr_max"]]
        t60_range = [config["data_config"]["t60_min"], config["data_config"]["t60_max"]]
        cfg_simu = simu.config.single_channel_single_source_config(use_reverb=use_reverb,
                                                                   use_noise=use_noise,
                                                                   snr_range=snr_range,
                                                                   t60_range=t60_range)

        #print("data simulation config{}".format(json.dumps(cfg_simu, sort_keys=True, indent=4)))

        # single source simulation shares the input streams with multi-source simulation, but uses different configurations
        simulator_ss = simu.Simulator(cfg_simu, source_streams, noise_streams=dir_noise_streams, rir_streams=rir_streams, iso_noise_streams=None)

        generator_config = DataGeneratorSequenceConfig(
            n_hour_per_epoch=config["sweep_size"],
            sequence_mode=self.sequence_mode,
            load_label=config["data_config"]["load_label"],
            seglen=config["data_config"]["seg_len"], 
            segshift=config["data_config"]["seg_shift"],
        )

        data_generator = DataGeneratorTrain(simulator_ss, generator_config, DEBUG=False)
        if self.sequence_mode:
            self.data_buffer = data_generator
        else:
            self.data_buffer = feature.data_generation.DataBuffer(data_generator, 
                                                                       buffer_size=20000,
                                                                       preload_size=200, 
                                                                       randomize=True)
#        self.data_buffer = feature.data_generation.DataBuffer(data_generator)
        self.use_cmn = config["data_config"]["use_cmn"]
        self.sample_len_seconds = config["data_config"]["seg_len"] * 0.01 # default sampling rate: 100Hz
        self.stream_idx_for_transform = [0]

    def load_streams(self, source_list, is_speech=True, is_rir=False):
        source_streams = list()
        for i in range(len(source_list)):
            corpus_type = source_list[i]['type']
            corpus_wav_path = source_list[i]['wav']
            label_paths = []
            label_names = []
            if 'label' in source_list[i]:
                label_paths.append(source_list[i]['label'])
                label_names.append('label')
            else:
                corpus_label_path = None
            if 'aux_label' in source_list[i]:
                label_paths.append(source_list[i]['aux_label'])
                label_names.append('aux_label')
            else:
                corpus_label_path = None
            print("%s::load_streams: loading %s from %s..." % (self.__class__.__name__, corpus_type, corpus_wav_path))
            curr_stream = reader.stream.gen_speech_stream_from_zip(corpus_wav_path,
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

        if self.use_cmn:
            data = feature.preprocess.apply_cmn(data, axis=0, stream_keys=[0])
            
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
    def __init__(self, n_hour_per_epoch=10, sequence_mode=False, load_label=True, min_seglen=0, seglen=500, segshift=500, gain_norm=False):
        self.n_hour_per_epoch = n_hour_per_epoch
        self.load_label = load_label
        self.segment_config = {}
        self.segment_config['sequence_mode'] = sequence_mode
        self.segment_config['seglen'] = seglen   # length of segments in terms of frames
        self.segment_config['segshift'] = segshift
        self.segment_config['min_seglen'] = min_seglen
        self.n_segment_per_epoch = int(3600 * n_hour_per_epoch / self.segment_config['seglen'] * 100)
        self.gain_norm = gain_norm

class DataGeneratorTrain(feature.data_generation.DataGenerator):
    def __init__(self, single_source_simulator, config, DEBUG=False):
        super().__init__(n_stream=2, data_len=config.n_segment_per_epoch)
        self.DEBUG = DEBUG
        self.single_source_simulator = single_source_simulator
        self.analyzer = self.single_source_simulator.analyzer
        self.config = config

    def generate(self, index=None, n_sample=1):
        seg_len = self.config.segment_config['seglen']
        seg_shift = self.config.segment_config['segshift']
        analyzer = self.single_source_simulator.analyzer
        min_len_sample = seg_len*analyzer.frame_shift+analyzer.frame_overlap

        if index is None:       # if no index is given, let the simulator do random sampling
            mixed_wav, early_reverb, mask, config = self.single_source_simulator.simulate(min_length=min_len_sample, normalize_gain=self.config.gain_norm)
        else:    # if index is given, use the specified sentence
            assert len(index) == 2
            sent_config = dict()
            sent_config['n_source'] = 1
            sent_config['source_stream_idx'] = index[0]
            sent_config['source_utt_id'] = [self.single_source_simulator.speech_streams[index[0]].utt_id[index[1]]]
            sent_config['source_speakers'] = [self.single_source_simulator.speech_streams[index[0]].utt2spk[sent_config['source_utt_id'][0]]]
            mixed_wav, early_reverb, mask, config = self.single_source_simulator.simulate(sent_config=sent_config, normalize_gain=self.config.gain_norm)

        speech_stream = self.single_source_simulator.speech_streams[config['source_stream_idx']]
        fbank = feature.feature.logfbank80(mixed_wav[:,0])
        utt_id = config['source_utt_id']

        if self.config.load_label:
            _, label = speech_stream.read_label_with_id(config['source_utt_id'])
            frame_label = label['label'][0].T
            aux_label = label['aux_label']
 
            if np.abs(frame_label.shape[0] - fbank.shape[0])>5:
                print("DataGeneratorTrain::generate: Warning: filterbank and label have significantly different number of frames. ")
            
            n_fr = np.minimum(frame_label.shape[0], fbank.shape[0])
            frame_label = frame_label[:n_fr,:]
            fbank = fbank[:n_fr,:]
       
        if self.config.segment_config['sequence_mode']:
            if self.config.load_label:
                train_samples = [(fbank, utt_id, frame_label, aux_label)]
            else:
                train_samples = [(fbank, utt_id)]
        else: 
            fbank_seg = utils.utt2seg(fbank.T, seg_len, seg_shift)

            if self.config.load_label:
                label_seg = utils.utt2seg(frame_label.T, seg_len, seg_shift)
                train_samples = [(fbank_seg[i].T, utt_id, label_seg[i].T) for i in range(len(label_seg))]
            else:
                train_samples = [(fbank_seg[i].T, utt_id) for i in range(len(fbank_seg))]
        
            if self.DEBUG:
                import matplotlib.pyplot as plt
                n_sample = len(train_samples)
                for i in range(n_sample):
                    plt.subplot(n_sample,2,i*2+1)
                    utils.imagesc(train_samples[i][0].T)
                    plt.subplot(n_sample,2,i*2+2)
                    plt.plot(train_samples[i][1])

        return train_samples
