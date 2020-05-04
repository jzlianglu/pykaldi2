import numpy as np

"""
class DataBuffer:
   This is the class that generate data for neural network training
    The job is to prepare speech corpus into individual training samples. E.g. feature-label pairs.
    It may also need to call speech simulation to simulated distorted multi-channel array data.
    This class generates samples and put them in a buffer (a FIFO queue).

    def __init__(self, data_generator, buffer_size=1000):
        self.data_generator = data_generator
        self.buffer_size = buffer_size
        self.buffer = []

    def get(self, index=None):
        Generate required number of training samples.
        cnt = 0
        while len(self.buffer) == 0:
            tmp_data = self.data_generator.generate(index=index)
            self.buffer += tmp_data
            cnt += 1
            if cnt > 10:
                print("DataBuffer::get: Warning: %d attempts to get data failed!" % cnt)
        data = self.buffer[0]
        self.buffer = self.buffer[1:]

        return data

    def get_len(self):
        return self.data_generator.get_len()
"""

class DataBuffer:
    """This is the class that generate data for neural network training and used with dynamic data simulation.
    The job is to prepare speech corpus into individual training samples. E.g. feature-label pairs.
    It may also need to call speech simulation to simulated distorted multi-channel array data.
    This class generates samples and put them in a buffer (a FIFO queue). """

    def __init__(self, data_generator, buffer_size=100, preload_size=10, randomize=True):
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

class DataGenerator:
    def __init__(self, n_stream=1, data_len=3600):
        self.n_stream = n_stream
        self.data_len = data_len
        self.DEBUG = False

    def generate(self, index=None, n_sample=1):
        """ Generate a list of training samples. Each list element is a tuple of n_stream data tensors. """
        pass

    def get_len(self):
        return self.data_len
