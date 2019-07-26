import numpy as np
import logging
from ._geometry import Rectangle


log = logging.getLogger(__name__)


def get_distribution_template(comment, max=3.0, min=1.0, mean=None, std=None, category=None, pmf=None,
                              distribution='uniform'):
    """
    Produce a template for distribution.
    :param comment: description of the distribution
    :param max: max value of the distribution
    :param min: min value of the distribution
    :param mean: mean value of the distribution
    :param std: standard deviation of the distribution
    :param category: category of discrete distribution
    :param pmf: probability mass function of discrete distribution
    :param distribution: type of distribution.
    :return: a dictionary that defines a distribution.
    """
    assert(max>=min)
    template = {'comment': comment + ', mean/std needed if distribution=gaussian', 'max': max, 'min': min, 'mean': mean,
                'std': std, 'category': category, 'pmf': pmf, 'distribution': distribution}
    return template


def get_sample(config, n_sample=1):
    """
    Sample values from a defined distribution.
    :param config: a distribution template dictionary produced by get_distribution_template()
    :param n_sample: number of samples to produce
    :return: sampled values
    """
    if config['distribution'] == 'binary':
        data = np.random.choice([0, 1], size=n_sample, replace=True, p=config['pmf'])

    elif config['distribution'] == 'discrete':
        data = np.random.choice(config['category'], size=n_sample, replace=True, p=config['pmf'])

    elif config['distribution'] == 'uniform':
        assert float(config['min']) < float(config['max'])
        data=np.random.uniform(low=float(config['min']),high=float(config['max']),size=n_sample)

    elif config['distribution'] == 'gaussian':
        data=np.random.normal(loc=float(config['mean']),scale=float(config['std']),size=n_sample)
        data = np.maximum(data, float(config['min']))
        data = np.minimum(data, float(config['max']))

    elif config['distribution'] == 'uniform_int':
        if int(config['min'])==int(config['max']):
            data=int(config['min'])*np.ones((n_sample,),dtype='int32')
        else:
            data=np.random.randint(int(config['min']),high=int(config['max']),size=n_sample)

    else:
        log.warning('Warning: unknown distribution type: %s' % config['distribution'])
        data = []

    return data


class _Sampler(object):
    def __init__(self, description):
        self.description = description

    def get_sample(self, n_sample=1):
        pass

    def __repr__(self):
        lines = []
        lines.append(super().__repr__())
        lines.append("%s: %s" % (self.__class__.__name__, self.description))
        lines.append(self.print_config())
        return "\n".join(lines)

    def print_config(self):
        pass    # to be printed by subclasses


class UniformSampler(_Sampler):
    """Sampler from uniform distribution. """
    def __init__(self, max=1.0, min=0.0, description="Uniform sampler"):
        super().__init__(description)
        self.max = float(max)
        self.min = float(min)
        assert self.min < self.max

    def get_sample(self, n_sample=1):
        data = np.random.uniform(low=self.min, high=self.max, size=n_sample)
        return data

    def print_config(self):
        return "Range = [%f, %f]" % (self.min, self.max)


class UniformIntSampler(_Sampler):
    """Sampler from uniform integer distribution. """
    def __init__(self, max=1, min=0, description="Uniform integer sampler"):
        super().__init__(description)
        self.max = int(max)
        self.min = int(min)
        assert self.min <= self.max

    def get_sample(self, n_sample=1):
        if self.min == self.max:
            data = self.min * np.ones((n_sample,),dtype='int32')
        else:
            data = np.random.randint(self.min,high=self.max,size=n_sample)
        return data

    def print_config(self):
        return "Range = [%d, %d]" % (self.min, self.max)


class GaussianSampler(_Sampler):
    """Gaussian sampler with min/max limits."""
    def __init__(self, mean=0.0, std=1.0, max=None, min=None, description="Gaussian sampler"):
        super().__init__(description)
        self.mean = mean
        self.std = std
        assert self.std > 0

        self.max = max
        self.min = min
        if self.max is not None and self.min is not None:
            assert self.min < self.max

    def get_sample(self, n_sample=1):
        max_v = self.max if self.max is not None else np.finfo(np.float64).max
        min_v = self.min if self.min is not None else np.finfo(np.float64).min
        boundary = np.array([min_v, max_v]).reshape((1, 2))

        data = np.zeros((n_sample,))
        n_valid_collected = 0
        for j in range(10000):
            tmp_data = np.random.normal(loc=self.mean,scale=self.std,size=n_sample).reshape((n_sample,1))
            valid_idx = np.where(np.prod(np.sign(tmp_data - boundary), axis=1) < 0)
            valid_idx = valid_idx[0]
            if len(valid_idx)>0:
                end_idx = np.minimum(n_valid_collected+len(valid_idx), n_sample)
                n_sample_taken = end_idx-n_valid_collected
                data[n_valid_collected:end_idx] = tmp_data[valid_idx[:n_sample_taken],0]
                n_valid_collected += n_sample_taken
            if n_valid_collected>= n_sample:
                break
            if j > 100 and j % 100 == 0:
                log.warning("Simulator::sample_array_position: Warning: cannot find a suitable value after %d tries." % j)
        if n_valid_collected < n_sample:
            log.warning("Simulator::sample_array_position: Warning: only %d valid samples collected after %d tries." %(n_valid_collected, j))
        elif n_valid_collected > n_sample:
            data = data[:n_sample]

        return data

    def print_config(self):
        lines = []
        lines.append("Mean = %f, Standard deviation = %f" % (self.mean, self.std))
        if self.max is not None:
            lines.append("Maximum is %f" % self.max)
        if self.min is not None:
            lines.append("Minimum is %f" % self.min)
        return "\n".join(lines)


class DiscreteSampler(_Sampler):
    """Sampler from a discrete/categorical distribution. """
    def __init__(self, category, pmf, description="Discrete random variable sampler"):
        super().__init__(description)
        self.category = category
        self.pmf = pmf
        assert len(category) == len(pmf)
        assert all(np.asarray(pmf)>=0) and all(np.asarray(pmf)<=1)

    def get_sample(self, n_sample=1):
        data = np.random.choice(self.category, size=n_sample, replace=True, p=self.pmf)
        return data

    def print_config(self):
        lines = []
        lines.append("Categories = [%s]" % (", ".join([str(i) for i in self.category])))
        lines.append(("Probabilities = [%s]" % ",".join([str(i) for i in self.pmf])))
        return "\n".join(lines)


class BinarySampler(DiscreteSampler):
    """Sampler from a binary distribution."""
    def __init__(self, prob=0.5, description="Binary variable sampler"):
        super().__init__([0, 1], [1-prob, prob], description)


def sample_room(config):
    """
    Sampling room length, width, and height.
    :param config: a dictionary defined in RoomConfig.
    :return: a 1D array containing the length, width, and height of the room.
    """
    room = np.zeros((3,))
    room[0] = get_sample(config['length'], n_sample=1)[0]
    room[1] = get_sample(config['width'], n_sample=1)[0]
    room[2] = get_sample(config['height'], n_sample=1)[0]
    return room


def sample_array_position(config, room_size, use_gaussian=False):
    """
    Sample the position of the array center
    :param config: the configuration of array positioning defined in ArrayPositionConfig
    :param room_size: 1D array of room sizes
    :param use_gaussian: whether to use Gaussian distribution
    :return: a dictionary that contains two entries: array_ctr and mic_position
        array_ctr is a 1D array of the xyz coordinates of the array's center point.
        mic_position is a 3xC matrix, where C is the number of microphones, and each column is the xyz coordinates of a
        microphone
    """
    array_ctr = np.zeros((3,))
    if use_gaussian:
        name = ['length_ratio', 'width_ratio', 'height_ratio']
        for i in range(len(name)):
            boundary = np.asarray([config[name[i]][j] * room_size[i] for j in ['min', 'max']])
            sampler = GaussianSampler(mean=np.mean(boundary), std=(boundary[1] - boundary[0]) / 6,
                                                          min=boundary[0], max=boundary[1])
            array_ctr[i] = sampler.get_sample()[0]
            array_ctr = array_ctr.reshape(3, 1)

    else:
        array_ctr[0] = get_sample(config['length_ratio'], n_sample=1)[0]
        array_ctr[1] = get_sample(config['width_ratio'], n_sample=1)[0]
        array_ctr[2] = get_sample(config['height_ratio'], n_sample=1)[0]
        array_ctr = array_ctr.reshape(3, 1) * room_size.reshape(3, 1)

    array = {}
    array['array_ctr'] = array_ctr
    array['mic_position'] = array['array_ctr'] + config['mic_positions']

    return array


def sample_source_position(config, room_size, array_center):
    """Sample the positions of the sound sources. """
    # sample the number of speakers
    n_spk = get_sample(config['num_spk'])
    n_spk = n_spk[0]

    if config['position_scheme'] == "random_coordinate":
        source_position = sample_source_position_by_random_coordinate(config, n_spk, room_size, array_center)
    else:
        raise Exception("Unknown speech source position scheme: %s" % config['position_scheme'])

    return source_position


def sample_source_position_by_random_coordinate(config, n_spk, room_size, array_center, forbidden_rect=None):
    """
    Sample positions of sound sources by uniformly sampling their 3D coordinates in the room.
    :param config: A configuration defined as in SoundSourceConfig
    :param n_spk: number of speakers
    :param room_size: 1D array containing room sizes
    :param array_center: 1D array containing xyz coordinates of microphone array center
    :param forbidden_rect: a region where we should not sample positions
    :return:
    """
    source_position = np.zeros((3, n_spk))

    d_from_wall = config['min_dist_from_wall'] if "min_dist_from_wall" in config.keys() else [0.0, 0.0] # minimum distance from wall
    d_from_array = config['min_dist_from_array'] if "min_dist_from_array" in config.keys() else 0.1     # minimum distnace from mic array
    d_from_other = config['min_dist_from_other'] if "min_dist_from_other" in config.keys() else 0.2     # minimum distance from other sources
    x_distribution = get_distribution_template("comment", min=d_from_wall[0],
                                                         max=room_size[0] - d_from_wall[0])
    y_distribution = get_distribution_template("comment", min=d_from_wall[0],
                                                         max=room_size[1] - d_from_wall[1])
    if "height" in config.keys():
        z_distribution = config['height']
    else:
        z_distribution = get_distribution_template("comment", min=0.0, max=room_size[2])

    for i in range(n_spk):
        cnt = 0
        while 1:
            cnt += 1
            x = get_sample(x_distribution)[0]
            y = get_sample(y_distribution)[0]
            z = get_sample(z_distribution)[0]
            curr_pos = np.asarray([x, y, z])
            if np.linalg.norm(curr_pos[:2]-array_center[:2]) >= d_from_array:
                if forbidden_rect is None or (np.prod(curr_pos[0] - forbidden_rect[0, :]) > 0 or np.prod(curr_pos[1] - forbidden_rect[1, :]) > 0):
                    if i == 0 or (np.linalg.norm(curr_pos[:2,np.newaxis]-source_position[:2, :i], axis=0) >= d_from_other).all():
                        source_position[:, i] = curr_pos[:]
                        break
            if cnt > 1000:
                raise Exception("Maximum number (1000) of trial finished but still not able to find acceptable position for speaker position. ")

    return source_position


# test
# coordinates = sample_source_position_by_random_coordinate({}, 5, np.asarray([3,3,3]), np.asarray([1.5,1.5,1.5]))
