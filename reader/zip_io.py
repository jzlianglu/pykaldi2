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
import zipfile
import soundfile
import numpy as np
import pickle
import scipy
import scipy.io.wavfile


class zip_or_dir(object):
    def __init__(self, fs=16000, dtype=np.float32):
        '''
        handles a zip file or directory of audio files
        :param fs:
        :param dtype:
        '''
        self.zip_objects = dict()
        self.zip_modes = dict()
        self.dtype = dtype
        self.fs = fs
        self.set_zip_object = self.get_zip_object

    def get_zip_object(self, zipname, zip_mode='a'):
        if zipname in self.zip_objects:
            assert(zip_mode == self.zip_modes[zipname])
        else:
            try:
                if os.path.isfile(zipname) and zipname[-4:].lower() == '.zip':
                    self.zip_objects[zipname] = zipfile.ZipFile(zipname, zip_mode, compression=zipfile.ZIP_DEFLATED)
                    self.zip_modes[zipname] = zip_mode
                elif zipname[-4:].lower() == '.zip':
                    assert(zip_mode == 'w' or zip_mode == 'a')
                    os.makedirs(os.path.dirname(zipname), exist_ok=True)
                    self.zip_objects[zipname] = zipfile.ZipFile(zipname, zip_mode, compression=zipfile.ZIP_DEFLATED)
                    self.zip_modes[zipname] = zip_mode
                else:
                    raise RuntimeError('Could not find archive {}'.format(zipname))
            except:
                raise Exception('Problem with zip file {}'.format(zipname))
        return self.zip_objects[zipname]

    def get_zip_obj_and_filename(self, filestring, zip_mode='a'):
        try:
            if filestring.find('@' + os.sep) >= 0:
                zipname, filename = filestring.split('@' + os.sep)
            elif filestring.find('@' + '/') >= 0:
                zipname, filename = filestring.split('@' + '/')
        except:
            raise Exception('error in finding zip filename.')
        obj = self.get_zip_object(zipname, zip_mode=zip_mode)
        return obj, filename

    def write_wav(self, x_array, wavfilename, sample_rate=16000, normalize=True, zip_mode='a'):
        if wavfilename.find('@' + '/') >= 0:
            memfile = io.BytesIO()
            write_wav(x_array, memfile, sample_rate=sample_rate, normalize=normalize)
            zip_obj, file_inzip = self.get_zip_obj_and_filename(wavfilename, zip_mode=zip_mode)
            zip_obj.writestr(file_inzip, memfile.getbuffer())
            memfile.close()
        else:
            write_wav(x_array, wavfilename,  sample_rate=sample_rate, normalize=normalize)

    def read_pickle(self, filename):
        if filename.find('@' + '/') >= 0:
            zip_obj, file_inzip = self.get_zip_obj_and_filename(filename, zip_mode='r')
            byte_chunk = zip_obj.read(file_inzip)
            byte_stream = io.BytesIO(byte_chunk)
            data = pickle.load(byte_stream)
        else:
            data = pickle.load(open(filename, 'rb'))
        return data

    def write_pickle(self, filename, data_dict, zip_mode='a'):
        if filename.find('@' + '/') >= 0:
            memfile = io.BytesIO()
            pickle.dump(data_dict, memfile, protocol=pickle.HIGHEST_PROTOCOL)
            zip_obj, file_inzip = self.get_zip_obj_and_filename(filename, zip_mode=zip_mode)
            zip_obj.writestr(file_inzip, memfile.getbuffer())
            memfile.close()
        else:
            pickle.dump(data_dict, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def read_npy(self, filename):
        if filename.find('@' + '/') >= 0:
            zip_obj, file_inzip = self.get_zip_obj_and_filename(filename, zip_mode='r')
            byte_chunk = zip_obj.read(file_inzip)
            byte_stream = io.BytesIO(byte_chunk)
            data = np.load(byte_stream)
        else:
            data = np.load(filename)
        return data

    def write_npy(self, filename, data, zip_mode='a'):
        if filename.find('@' + '/') >= 0:
            memfile = io.BytesIO()
            np.save(memfile, data)
            zip_obj, file_inzip = self.get_zip_obj_and_filename(filename, zip_mode=zip_mode)
            zip_obj.writestr(file_inzip, memfile.getbuffer())
            memfile.close()
        else:
            np.save(file_name, data)

    def read_wav(self, wavfilename):
        if wavfilename.find('@' + '/') >= 0:
            zip_obj, file_inzip = self.get_zip_obj_and_filename(wavfilename, zip_mode='r')
            byte_chunk = zip_obj.read(file_inzip)
            byte_stream = io.BytesIO(byte_chunk)

            with soundfile.SoundFile(byte_stream, 'r') as f:
                fs_read = f.samplerate
                x = f.read()
        else:
            with soundfile.SoundFile(wavfilename, 'r') as f:
                fs_read = f.samplerate
                x = f.read()
        if fs_read != self.fs:
            x = resampy.resample(x, fs_read, self.fs)
            fs_read = self.fs
        return fs_read, x.astype(self.dtype)

    def walk(self, zipordirname):
        if zipordirname[-4:].lower() == '.zip':
            obj = self.get_zip_object(zipordirname, zip_mode='r')
            for filename in obj.namelist():
                if not filename.endswith('.wav') and not filename.endswith('.flac'):
                    continue
                yield '{}@/{}'.format(zipordirname, filename)
        else:
            for root, directories, filenames in os.walk(zipordirname):
                for filename in filenames:
                    if not filename.endswith('.wav') and not filename.endswith('.flac'):
                        continue
                    audio_file = os.path.join(root, filename)
                    yield audio_file

    def close(self, zipfilename):
        if zipfilename in self.zip_objects:
            self.zip_objects[zipfilename].close()
            self.zip_objects.pop(zipfilename, None)
            self.zip_modes.pop(zipfilename, None)

    def close_all(self):
        zip_files = list(self.zip_objects.keys())
        for zipfilename in zip_files:
            self.zip_objects[zipfilename].close()
            self.zip_objects.pop(zipfilename, None)
            self.zip_modes.pop(zipfilename, None)

    def __del__(self):
        for name in self.zip_objects:
            self.zip_objects[name].close()


class zip_io(object):
    def __init__(self, zip_file, mode='r', fs=16000, dtype=np.float32):
        '''
        :param zip_file:
        :param mode: 'a' or 'w'
        '''
        os.makedirs(os.path.dirname(zip_file), exist_ok=True)
        assert (mode == 'a' or mode == 'w' or mode == 'r')
        self.zip_obj = zipfile.ZipFile(zip_file, mode, compression=zipfile.ZIP_DEFLATED)
        self.dtype = dtype
        self.fs = fs

    def write_wav(self, x_array, wavfilename):
        memfile = io.BytesIO()
        write_wav(x_array, memfile, sample_rate=self.fs, normalize=True)
        self.zip_obj.writestr(wavfilename, memfile.getbuffer())
        memfile.close()

    def read_wav(self, wavfilename):
        byte_chunk = self.zip_obj.read(wavfilename)
        byte_stream = io.BytesIO(byte_chunk)

        with soundfile.SoundFile(byte_stream, 'r') as f:
            fs = f.samplerate
            x = f.read()
        assert(fs == self.fs)
        return fs, x.astype(self.dtype)

    def walk(self):
        for name in  self.zip_obj.namelist():
            yield name

    def write_file(self, file_to_write):
        self.zip_obj.write(file_to_write)

    def __del__(self):
        self.zip_obj.close()



def write_wav(data, path, sample_rate=16000, normalize=False):
    """ Write the audio data ``data`` to the wav file ``path``

    Args:
        data (numpy.ndarray) : Numpy array containing the audio data
        path (string) : Path to the wav file to which the data will be written
        sample_rate (int) : Sampling rate with which the data will be stored
        normalize (bool) : Enable/disable signal normalization

    Returns:
        float : Normalization factor (returns 1 when ``normalize``==``False``)
    """

    data = data.copy()
    int16_max = np.iinfo(np.int16).max
    int16_min = np.iinfo(np.int16).min

    if normalize:
        if not data.dtype.kind == 'f':
            data = data.astype(np.float)
        norm_coeff = 1. / np.max(np.abs(data)) * int16_max / int16_min
        data *= norm_coeff
    else:
        norm_coeff = 1.

    if data.dtype.kind == 'f':
        data *= int16_max

    sample_to_clip = np.sum(data > int16_max)
    if sample_to_clip > 0:
        print('Warning, clipping {} samples'.format(sample_to_clip))
    data = np.clip(data, int16_min, int16_max)
    data = data.astype(np.int16)

    if data.ndim > 1:
        data = data.T

    scipy.io.wavfile.write(path, sample_rate, data)
    return norm_coeff
