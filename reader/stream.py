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

import os
import io
import numpy as np
import glob
import zipfile
import pickle
import copy
from . import reader


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


def wavlist2uttlist(wav_list):
    utt_list = []
    for i in range(len(wav_list)):
        terms = wav_list[i].split("\t")
        utt_list.append(os.path.splitext(os.path.basename(terms[0]))[0])
    return utt_list


def get_relative_path(root, file_list):
    rel_file_list = []
    for i in range(len(file_list)):
        rel_file_list.append(os.path.relpath(file_list[i], root))

    return rel_file_list


class DataStream:
    """
    An DataStream object holds a list of data, which can be either ndarrays or file names. The object provides interface
    to sample or read the data in the list. If the list holds file names, suitable reader object needs to be provided.
    """
    def __init__(self, data=None, precision='float32', is_file=True, reader=None, frame_rate=100, root=None):
        self.data = data
        self.precision = precision
        self.is_file = is_file
        self.reader = reader
        self.frame_rate = frame_rate
        if is_file and reader is None:
            print("Warning: is_file is set to True but reader is not provided\n")
        self.data_len = None
        self.num_of_data = 0
        self.root = root

    def close(self):
        """close any file stream in the reader if exist. """
        if self.reader is not None:
            self.read.close()

    def get_number_of_data(self):
        if self.num_of_data == 0:
            self.num_of_data = len(self.data)

        return self.num_of_data

    def get_data(self, index):
        index = np.asarray(index)
        index = index.reshape(index.size)
        data = []
        name = []
        for i in range(index.size):
            if self.is_file:
                curr_file = self.get_full_path(self.data[index[i]])
                curr_data = self.get_data_from_file( curr_file )
                name.append(curr_file)
            else:
                curr_data = self.data[index[i]]
                name.append("NUMERIC")

            curr_data = reader.convert_data_precision(curr_data, self.precision)
            data.append(curr_data)

        return data, name

    def set_data_len(self):
        if self.data_len is not None and len(self.data_len) == len(self.data):
            # assume the data_len is already available. Do nothing
            return

        self.data_len = []
        for i in range(len(self.data)):
            tmp_data_len = self.get_data_len(i)
            self.data_len.append(tmp_data_len)

    def get_data_len(self, index):
        index = np.asarray(index)
        index = index.reshape(index.size)
        if self.data_len is not None and len(self.data_len) == len(self.data):
            data_len = [self.data_len[i] for i in index]
        else:
            data_len = []
            for i in range(index.size):
                if self.is_file:
                    curr_file = self.get_full_path(self.data[index[i]])
                    curr_data = self.get_data_len_from_file(curr_file)
                else:
                    curr_data = self.data[index[i]].shape[1]

                data_len.append(curr_data)

        return data_len

    def sample_data(self, n_data=1, replace=False, read_data_file=True):
        if replace:
            idx = np.random.randint(0, len(self.data), n_data)
        else:
            idx = np.random.choice(len(self.data), n_data, replace=False)
        if read_data_file:
            return self.get_data(idx)
        else:
            sampled_file_list = [self.data[i] for i in idx]
            return sampled_file_list

    def get_data_from_file(self, file_name):
        data = self.reader.read(file_name)
        if isinstance(data, tuple):
            data = data[0]
        return reader.convert_data_precision(data, self.precision)

    def get_data_len_from_file(self, file_name):
        data_len = self.reader.get_len(file_name)
        if isinstance(data_len, tuple):
            data_len = data_len[0]
        return data_len

    def get_full_path(self, file_name):
        if self.root is not None:
            if self.root.find('@') >= 0:
                return self.root+file_name
            else:
                return os.path.join(self.root, file_name)
        else:
            return file_name


class RIRStream (DataStream):
    """RIRStream is a special DataStream. There is one more attribute of the class, i.e. the configuration of the RIRs.
    When we sample RIR channels, we need to also return the corresponding configurations, e.g. room size, source and
    microphone array positions, reverberation times, etc. """
    def __init__(self, data=None, config=None, precision='float32', is_file=True, reader=None, frame_rate=100, root=None):
        super().__init__(data, precision, is_file, reader, frame_rate, root)
        self.config = config

    def sample_rir(self, n_position, replace=False):
        """Sample an RIR file first, then sample n_position rir positions from the file. """
        if replace:
            rir_idx = np.random.randint(0, len(self.data), 1)
        else:
            rir_idx = np.random.choice(len(self.data), 1, replace=False)
        rir_idx = rir_idx[0]

        cfg = self.config[rir_idx]
        n_position_in_rir = cfg['n_position']
        rir_len = cfg['n_sample']

        assert n_position_in_rir >= n_position
        chosen_positions = np.random.choice(n_position_in_rir, n_position, replace=False)

        if 1:   # first option is to read all the positions' rir and take the required ones
            rir_wav, rir_file = self.get_data(rir_idx)
            rir_wav = rir_wav[0]
            n_ch = rir_wav.shape[1]
            rir_wav = np.reshape(rir_wav, (rir_len, n_position_in_rir, n_ch), order='F')
            rir_wav_chosen = []
            for i in range(n_position):
                rir_wav_chosen.append(rir_wav[:,chosen_positions[i],:])

        else:   # second option is to read only the required rirs
            # TODO: do partial reading to save load time.
            pass

        return rir_wav_chosen, cfg['room_size'], cfg['array_position'], cfg['source_position'][:, chosen_positions], cfg['t60']


class SpeechDataStream:
    """
       General class for organizing and accessing speech corpus. Provide interface to sample speaker or utterances from
       the corpus. We can load data and/or vad of the sentences if available.
       Specific corpus (e.g. WSJ) should inherit this class and overload the function that generate spk id from utt id.
    """
    def __init__(self, utt_id, data_stream, utt2spk=None, vad_stream=None, label_streams=None):
        """
        We need to at least provide a list of utterance IDs and a DataStream object that holds data for those utternaces.
        :param utt_id: a list of unique ID for utterances
        :param data_stream: a DataStream object that holds the data of the utterances. data_stream.data is a list that have 1-to-1 correspondence to utt_id
        :param vad_stream: similar to data_stream, but holds vad of the utterances.
        :param label_streams: a dictionary of DataStreams that hold various kinds of labels, e.g. frame-level phone labels and word labels.
        """
        self.utt_id = utt_id
        self.data_stream = data_stream
        self.vad_stream = vad_stream
        self.label_streams = label_streams

        self.spk_id = []
        if utt2spk is None:
            self.utt2spk = {}
        else:
            self.utt2spk = utt2spk
        self.spk2utt = {}

        # generate the mappings
        self.gen_mapping()

    def close(self):
        # close any file stream if exists
        if self.data_stream is not None:
            self.data_stream.close()
        if self.vad_stream is not None:
            self.vad_stream.close()
        if self.label_streams is not None:
            for label_name,label_stream in self.label_streams.items():
                label_stream.close()

    def remove_spk(self, spk):
        # remove all data related to a speaker
        utt_of_spk = self.spk2utt[spk]
        utt_idx_list = []
        for i in utt_of_spk:
            self.utt2spk.pop(i)
            utt_idx_list.append(self.utt_id.index(i))

        remove_from_list_by_index(self.utt_id, utt_idx_list)

        if self.data_stream is not None:
            remove_from_list_by_index(self.data_stream.data, utt_idx_list)
            if self.data_stream.data_len is not None:
                remove_from_list_by_index(self.data_stream.data_len, utt_idx_list)

        if self.vad_stream is not None:
            remove_from_list_by_index(self.vad_stream.data, utt_idx_list)
        if self.label_streams is not None:
            for label_name, label_stream in self.label_streams.items():
                remove_from_list_by_index(label_stream.data, utt_idx_list)

        # regenerate the mappings
        self.spk_id = []
        self.spk2utt = {}
        self.gen_mapping()

    def get_number_of_data(self):
        return self.data_stream.get_number_of_data()

    def get_data_len(self, idx):
        return self.data_stream.get_data_len(idx)

    def get_data_len_by_id(self, id):
        if isinstance(id, list):
            idx = [self.utt_id.index(id[i]) for i in range(len(id))]
        else:
            idx = self.utt_id.index(id)
        return self.data_stream.get_data_len(idx)

    def read_utt(self, idx, load_data=False, load_vad=False, load_label=False):
        if type(idx) is np.ndarray:
            idx = idx.reshape(idx.size)
        elif type(idx) is not list:
            idx = [idx]
        name = []
        data = []
        vad = []
        for i in idx:
            name.append(self.utt_id[i])
            if load_data:
                tmp_data = self.data_stream.get_data(i)
                if isinstance(tmp_data, tuple):
                    tmp_data = tmp_data[0]
                data += tmp_data
            if load_vad and self.vad_stream is not None:
                tmp_vad = self.vad_stream.get_data(i)
                if isinstance(tmp_vad, tuple):
                    tmp_vad = tmp_vad[0]
                vad+=tmp_vad

        if load_label and self.label_streams is not None:
            _, label = self.read_label(idx)
            return idx, name, data, vad, label
        else:
            return idx,name,data,vad        # backward compatible

    def read_utt_with_id(self, id, load_data=False, load_vad=False, load_label=False):
        if isinstance(id, list):
            idx = [self.utt_id.index(i) for i in id]
        else:
            idx = self.utt_id.index(id)

        return self.read_utt(np.asarray(idx), load_data=load_data, load_vad=load_vad, load_label=load_label)

    def read_label(self, idx):
        """Input is a list of integer indexes"""
        name = []
        label = []

        for i in idx:
            name.append(self.utt_id[i])

        label = dict()
        for label_name, label_stream in self.label_streams.items():
            curr_label = []
            for i in idx:
                tmp_label = label_stream.get_data(i)
                if isinstance(tmp_label, tuple):
                    tmp_label = tmp_label[0]
                curr_label += tmp_label
            label[label_name] = curr_label

        return name, label

    def read_label_with_id(self, id):
        if isinstance(id, list):
            idx = [self.utt_id.index(i) for i in id]
        else:
            idx = self.utt_id.index(id)
        return self.read_label(np.asarray(idx))

    def sample_utt(self, n_utt=1, replace=False, load_data=False, load_vad=False, load_label=False, min_length=None):
        cnt = 0
        while True:
            idx = np.random.choice(len(self.utt_id), n_utt, replace=replace)
            if self.check_min_len_requirement(idx, min_length=min_length):
                break
            cnt += 1
            if cnt >= 100 and cnt % 100 == 0:
                print("SpeechDataStream::sample_utt: Warning: not able to find data with length longer than %d after %d attempts. " % (min_length, cnt))

        return self.read_utt(idx, load_data=load_data, load_vad=load_vad, load_label=load_label)

    def sample_spk(self, n_spk, replace=False, unwanted_spk_id=None):
        spk_list = self.spk_id
        if unwanted_spk_id is not None:
            candidate_set = [i for i in range(len(spk_list)) if spk_list[i] not in unwanted_spk_id]
        else:
            candidate_set = [i for i in range(len(spk_list))]

        if replace is False:
            assert n_spk <= len(candidate_set)

        if replace:
            idx = np.random.randint(0, len(scandidate_set), n_spk)
        else:
            idx = np.random.choice(len(candidate_set), n_spk, replace=False)

        spk_idx = np.asarray([candidate_set[i] for i in idx])
        spk_idx = spk_idx.reshape(spk_idx.size)
        spk = []
        for i in spk_idx:
            spk.append(self.spk_id[i])

        return spk_idx,spk

    def check_min_len_requirement(self, data_idx, min_length=None):
        if min_length is None:
            return True
        else:
            satisfy_min_len_requirement = True
            for i in data_idx:
                data_len = np.asarray(self.data_stream.get_data_len(i))
                data_len = np.reshape(data_len, (data_len.size,1))
                if data_len[0,0]<min_length:
                    satisfy_min_len_requirement = False
                    break

        return satisfy_min_len_requirement

    def sample_utt_from_spk(self, spk, unwanted_utt_id=None, n_utt=1, replace=False, load_data=False, load_vad=False, load_label=False, min_length=None):
        if spk not in self.spk2utt:
            print("Speaker %s not in the corpus, return empty. " % spk)
            return None
        
        utt_of_spk = self.spk2utt[spk]
        if unwanted_utt_id is not None and len(unwanted_utt_id)>0:
            candidate_set = [i for i in range(len(utt_of_spk)) if utt_of_spk[i] not in unwanted_utt_id]
        else:
            candidate_set = [i for i in range(len(utt_of_spk))]

        if replace is False:
            assert n_utt <= len(candidate_set)

        cnt = 0
        while True:
            idx = np.random.choice(candidate_set, n_utt, replace=replace)
            sampled_utt_id = [utt_of_spk[i] for i in idx]
            global_idx = [self.utt_id.index(i) for i in sampled_utt_id]
            if self.check_min_len_requirement(global_idx, min_length=min_length):
                break
            cnt += 1
            if cnt > 100:
                print("SpeechDataStream::sample_utt_from_spk: Warning: not able to find data with length longer than %d for speaker %s after 100 attempts. " % (min_length, spk))

        return self.read_utt_with_id(sampled_utt_id, load_data=load_data, load_vad=load_vad, load_label=load_label)

    # sample n_spk speakers, then sample n_utt for each speaker
    def sample_spk_and_utt(self, n_spk=1, n_utt_per_spk=1, replace=False, load_data=False, load_vad=False,
                           load_label=False, min_length=None):
        spk_idx, speakers = self.sample_spk(n_spk, replace=False)
        utt_id = []
        data = []
        vad = []
        label = []
        for i in range(len(speakers)):
            if load_label and self.label_streams is not None:
                idx, tmp_utt_id, tmp_data, tmp_vad, tmp_label = self.sample_utt_from_spk(speakers[i], n_utt=n_utt_per_spk,
                                                                          replace=replace, load_data=load_data,
                                                                          load_vad=load_vad, load_label=load_label,
                                                                          min_length=min_length)
                label += tmp_label
            else:
                idx, tmp_utt_id, tmp_data, tmp_vad = self.sample_utt_from_spk(speakers[i], n_utt=n_utt_per_spk,
                                                                              replace=replace, load_data=load_data,
                                                                              load_vad=load_vad, load_label=load_label,
                                                                              min_length=min_length)
            utt_id+= tmp_utt_id
            data+=tmp_data
            vad+=tmp_vad

        if load_label and self.label_streams is not None:
            return speakers, utt_id, data, vad, label
        else:
            return speakers, utt_id, data, vad

    def gen_mapping(self):
        for i in range(len(self.utt_id)):
            utt_id = self.utt_id[i]
            if utt_id in self.utt2spk:
                spk_id = self.utt2spk[utt_id]
            else:
                spk_id = self.utt_id2spk_id(utt_id)
                self.utt2spk[utt_id] = spk_id

            self.spk2utt_insert_utt(spk_id, utt_id)

        self.spk_id = list(self.spk2utt.keys())

    def spk2utt_insert_utt(self, spk, utt):
        if spk in self.spk2utt:
            self.spk2utt[spk].append(utt)
        else:
            self.spk2utt[spk] = [utt]

    def utt_id2spk_id(self, utt_id):
        """Different corpus usually has different ways to convert utt_id to spk_id
        So we should overload the class and provide the implementation in the subclasses."""
        pass


class LibriDataStream (SpeechDataStream):
    def utt_id2spk_id(self, utt_id):
        return utt_id.split("-")[0]


class WSJDataStream(SpeechDataStream):
    def utt_id2spk_id(self, utt_id):
        return utt_id[:3]


def gen_speech_stream_from_list(wav_list, utt2spk, get_duration=True, use_zip=True):
    """ Generate speech stream from wav file list and utt2spk. """

    wav_reader = reader.ZipWaveIO(precision="float32")
    utt_id = wavlist2uttlist(wav_list)

    data_stream = DataStream(wav_list, is_file=True, reader=wav_reader)
    speech_stream = SpeechDataStream(utt_id, data_stream, utt2spk=utt2spk)

    if get_duration:
        speech_stream.data_stream.set_data_len()

    return speech_stream


def gen_speech_stream_from_zip(zip_path, file_extension='wav', label_files=None, label_names=None, utt2spk=None,
                               corpus_name=None, is_speech_corpus=True, is_rir=False, get_duration=False):
    """ Generate speech stream from zip file and utt2spk. The zip file contains wavfiles.
    Parameters
    -----------
    zip_path: path of the zip file that contains the waveforms.
    label_files: list of label files. Each line of label_files contains label for one utterance and have following
        format:
            utt_id_1 label_1
            utt_id_2 label_2
            ...
        where utt_id_1 and utt_id_2 are utterance IDs of the sentences and can be any string as far as each utterance
        has an unique ID. The utt_ids must be compatible with the file_names (excluding extension) in the zip file.
    file_extension: define the extension of the files in the zip file. Used to filter out non-waveform files.
    label_files: list of strings specifying the name of the label_files, e.g. "frame_label', 'word_label', etc.
    utt2spk: a dictionary mapping from utterance ID to speaker ID. If not provided, corpus_name must be provided.
    is_speech_corpus: bool, whether the zip contains speech.
    is_rir: bool, whether the zip contains RIR. If True, expect a config file in the zip that contains the meta data
        info about the RIRs.
    get_duration: bool, whether to get duration of the waveforms

    Returned:
        An object of type SpeechDataStream, RIRDataStream, or DataStream.
    """
    wav_reader = reader.ZipWaveIO(precision="float32")
    zip_file = zipfile.ZipFile(zip_path)
    all_list = zip_file.namelist()
    wav_list = [i for i in all_list if os.path.splitext(i)[1][1:] == file_extension]
    utt_id_wav = wavlist2uttlist(wav_list)
    # sort wav_list by utterance ID
    tmp = sorted(zip(utt_id_wav, wav_list))
    utt_id_wav = [i[0] for i in tmp]
    wav_list = [i[1] for i in tmp]

    def get_label(label_lines, selected_utt_id):
        selected_label_list = []
        for line in label_lines:
            tmp = line.split(" ")
            utt_id = tmp[0]
            if utt_id in selected_utt_id:
                tmp_label = np.asarray([int(j) for j in tmp[1:] if len(j)>0])[np.newaxis,:]
                selected_label_list.append(tmp_label)
        return selected_label_list

    if label_files is not None:
        # Find the intersection of the utterance IDs
        selected_utt_id = set(utt_id_wav)
        utt_id_label = []
        label_file_lines = []
        for i in range(len(label_files)):
            lines = my_cat(label_files[i])
            lines.sort()        # each lines start with utterance ID, hence effectively sort the labels with utterance ID.
            curr_utt_id_label = [i.split(" ")[0] for i in lines]
            selected_utt_id = set(curr_utt_id_label) & selected_utt_id
            utt_id_label.append(curr_utt_id_label)
            label_file_lines.append(lines)

        # Build DataStream for each label types
        label_streams = dict()
        if label_names is None:
            label_names = ['label_'+str(i) for i in range(len(label_files))]
        for i in range(len(label_files)):
            selected_label_list = get_label(label_file_lines[i], selected_utt_id)   # selected_label_list is sorted, as label_file_lines[i] is sorted.
            label_streams[label_names[i]] = DataStream(selected_label_list, is_file=False, reader=None)

        selected_wav_list = [wav_list[i] for i in range(len(wav_list)) if utt_id_wav[i] in selected_utt_id]
        selected_utt_id = list(selected_utt_id)
        selected_utt_id.sort()
        # note that selected_wav_list, selected_label_list, and selected_utt_id are all sorted by utterance ID. So they
        # are guaranteed to have one-to-one correspondence if the utterance IDs are unique.
    else:
        label_streams = None
        selected_utt_id = utt_id_wav
        selected_wav_list = wav_list

    root_dir = zip_path + '@/'
    if is_speech_corpus:
        assert utt2spk is not None or corpus_name is not None

        data_stream = DataStream(selected_wav_list, is_file=True, reader=wav_reader, root=root_dir)
        if corpus_name == 'LibriSpeech':
            corpus_stream = LibriDataStream(selected_utt_id, data_stream, label_streams=label_streams)
        elif corpus_name == 'WSJ':
            corpus_stream = WSJDataStream(selected_utt_id, data_stream, label_streams=label_streams)
        else:       # for unknown corpus, you need to provide the utt2spk mapping.
            corpus_stream = SpeechDataStream(selected_utt_id, data_stream, utt2spk=utt2spk, label_streams=label_streams)
    elif is_rir:
        for i in all_list:
            if os.path.splitext(i)[1][1:] == 'pkl':
                config_file = i
                break

        byte_chunk = zip_file.read(config_file)
        byte_stream = io.BytesIO(byte_chunk)
        config = pickle.load(byte_stream)
        zip_base = os.path.splitext(os.path.basename(zip_path))[0]
        wav_list = [zip_base+'/'+i['file'] for i in config]

        data_stream = RIRStream(wav_list, config=config, is_file=True, reader=wav_reader, root=root_dir)
        corpus_stream = data_stream
    else:
        data_stream = DataStream(selected_wav_list, is_file=True, reader=wav_reader, root=root_dir)
        corpus_stream = data_stream

    if get_duration:
        if is_speech_corpus:
            corpus_stream.data_stream.set_data_len()
            corpus_stream.data_stream.reader = reader.ZipWaveIO(precision="float32")
        else:
            corpus_stream.set_data_len()
            corpus_stream.reader = reader.ZipWaveIO(precision="float32")

    return corpus_stream

