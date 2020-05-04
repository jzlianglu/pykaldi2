import numpy as np
import soundfile as sf
import re
from . import zip_io
import pickle
import h5py


def convert_data_precision(data, precision):
    data_type = type(data)

    if data_type == dict:
        return data     # do not convert dictionary
    elif data_type==list:
        data_list = data
    elif data_type==tuple:
        data_list = list(data)
    else:
        data_list = [data]

    converted = []
    for seg in data_list:
        if type(seg) == dict:
            converted.append(seg)
        elif type(seg) == list:
            converted.append(convert_data_precision(seg, precision))
        elif precision == seg.dtype:
            converted.append(seg)
        elif precision == "float32" or precision == "float64":
            converted.append(seg.astype(precision))
        elif precision == "int16":
            seg2 = seg / np.max(np.abs(seg)) * 2 ** 15
            converted.append(seg2.astype("int16"))

    if data_type==list:
        return converted
    elif data_type==tuple:
        return tuple(converted)
    else:
        return converted[0]


class BinaryIO:
    """
    Provide interface to read write files into binary file
    """
    def __init__(self):
        pass

class HDF5IO:
    """
    Provide interface to read write files into a HDF5 file
    """
    def __init__(self, file_name=None):
        if file_name is not None:
            self.open(file_name)

    def open(self, file_name):
        self.h5 = h5py.File(file_name, 'r')
        self.keys = list(self.h5.keys())

    def write(self, file_name):
        with h5py.File(file_name, "w") as f:
            dset = f.create_dataset("mydataset", (100,), dtype='i')


class ZipIO:
    """Reader class that manages reading and writing from zip file. """
    def __init__(self):
        self.zip_reader = zip_io.zip_or_dir()

    def read(self, file_name):
        pass

    def write(self, file_name, data):
        pass

    def close(self, zip_file=None):
        if zip_file is None:
            self.zip_reader.close_all()
        else:
            self.zip_reader.close(zip_file)


class ZipPickleIO(ZipIO):
    """Provide interface to read data structures saved in pickle format.
    Note that this class is only responsible for loading data from pickle, but does not interpret the content of the
    pickle file. """

    def read(self, file_name):
        data = self.zip_reader.read_pickle(file_name)
        return data

    def write(self, file_name, data_dict):
        self.zip_reader.write_pickle(file_name, data_dict)


class ZipNpyIO(ZipIO):
    """Provide interface to read data structures saved in npy format in a zip file.
    """
    def read(self, file_name):
        data = self.zip_reader.read_npy(file_name)
        return data

    def write(self, file_name, data):
        self.zip_reader.write_npy(file_name, data)

    def get_len(self, file_name):
        data = self.zip_reader.read_npy(file_name)
        return data.shape[0]



class WaveIO:
    """
    Provide interface to read waveforms of sentences. Support 1 or more channels, and several precision types, such as
    "float32", "float64", and "int16".
    For multi-channel recordings, support two cases: 1) one channel per file; 2) all channels in one file.
    Support reading arbitrary sample start and end.
    Support choosing of channels from multi-channel recordings.
    """
    def __init__(self, fs=16000, n_ch=1, precision="float32", channel_in_diff_files=False):
        self.fs = fs
        self.n_ch = n_ch
        self.precision = precision
        self.channel_in_diff_files = channel_in_diff_files

    def get_len(self, file_name):
        wav_file_name, select_channel, select_sample, channel_id, sample_idx = self.parse_file_name(file_name)

        if select_sample:
            data_len = sample_idx[1]-sample_idx[0]+1
        else:
            file = sf.SoundFile(wav_file_name)  # we don't need to really read the data
            data_len = len(file)

        return data_len

    def read(self, file_name):
        if self.n_ch>1 and self.channel_in_diff_files:
            wav = []
            for i in range(self.n_ch):
                tmp_wav,fs = self.read_a_file(file_name[i])
                wav.append(tmp_wav)
            wav = np.asarray(wav)
        else:
            wav,fs = self.read_a_file(file_name)

        # always have 2D matrix
        if wav.ndim == 1:
            wav = wav.reshape(wav.size,1)

        return wav,fs

    def read_a_file(self, file_name):
        wav_file_name, select_channel, select_sample, channel_id, sample_idx = self.parse_file_name(file_name)

        if select_sample:
            wav,fs = sf.read(wav_file_name, start=sample_idx[0]-1, stop=sample_idx[1])
        else:
            wav,fs = sf.read(wav_file_name)

        if select_channel:
            wav = wav[:,channel_id-1]

        wav = convert_data_precision(wav, self.precision)

        return wav,fs

    def write(self, file_name, data):
        sf.write(file_name, data, self.fs)

    def parse_file_name(self, file_name):
        """
        file_name can have following formats:
            *.wav\tsample=start_sample,end_sample\tchannel=ch_idx1,ch_idx2
        \t here represent tab
        start_sample and end_sample are integer sample indexes starting from 1
        ch_idx1 ch_idx2, etc are channel indexces starting from 1
        """

        terms = re.split('\t+', file_name)
        wav_file_name = terms[0]

        select_channel=False
        select_sample=False
        channel_id = None
        sample_idx = None
        for i in range(1,len(terms)):
            items = terms[i].split("=")
            if items[0] == "channel":
                select_channel = True
                channel_id = np.asarray(list(map(int, items[1].split(","))))

            elif items[0] == "sample":
                select_sample = True
                sample_idx = list(map(int, items[1].split(",")))

        return wav_file_name, select_channel, select_sample, channel_id, sample_idx

class ZipWaveIO (WaveIO):
    """
    Provide interface to read write files into a Zip file
    Use zip_or_dir class from Hakan for dealing with the zip format.
    """
    def __init__(self, fs=16000, n_ch=1, precision="float32", channel_in_diff_files=False):
        super().__init__(fs=fs, n_ch=n_ch, precision=precision, channel_in_diff_files=channel_in_diff_files)
        self.zip_reader = zip_io.zip_or_dir(fs=fs)

    def read_a_file(self, file_name):
        wav_file_name, select_channel, select_sample, channel_id, sample_idx = self.parse_file_name(file_name)

        fs,wav = self.zip_reader.read_wav(wav_file_name)

        if wav.ndim==1:
            wav = np.reshape(wav, (wav.size,1))

        if select_sample:
            wav = wav[sample_idx[0]-1 : sample_idx[1],:]

        if select_channel:
            wav = wav[:,channel_id-1]

        wav = convert_data_precision(wav, self.precision)

        return wav,fs

    def write_a_file(self, file_name, data, normalize=False):
        self.zip_reader.write_wav(data, file_name, sample_rate=self.fs, normalize=normalize, zip_mode='a')

    def get_len(self, file_name):
        wav_file_name, select_channel, select_sample, channel_id, sample_idx = self.parse_file_name(file_name)

        if select_sample:
            data_len = sample_idx[1]-sample_idx[0]+1
        else:
            fs, wav = self.zip_reader.read_wav(wav_file_name)
            data_len = wav.shape[0]

        return data_len

    def close(self, zip_file=None):
        if zip_file is None:
            self.zip_reader.close_all()
        else:
            self.zip_reader.close(zip_file)
