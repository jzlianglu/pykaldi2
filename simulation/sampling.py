import numpy as np
import scipy
from time import gmtime, strftime
import re
import os
import sys
sys.path.append("../..")
import signal_graph as sig


class Sampler(object):
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


class UniformSampler(Sampler):
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


class UniformIntSampler(Sampler):
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


class GaussianSampler(Sampler):
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
                print("Simulator::sample_array_position: Warning: cannot find a suitable value after %d tries." % j)
        if n_valid_collected < n_sample:
            print("Simulator::sample_array_position: Warning: only %d valid samples collected after %d tries." %(n_valid_collected, j))
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

class DiscreteSampler(Sampler):
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
    def __init__(self, prob=0.5, description="Binary variable sampler"):
        super().__init__([0, 1], [1-prob, prob], description)


class AroundRectangleSampler2D(Sampler):
    def __init__(self, room_size, seat_circle, not_allowed_region, min_prob=0.001):
        self.room_size = room_size
        self.x_sampler = UniformSampler(room_size[0], 0)
        self.y_sampler = UniformSampler(room_size[1], 0)
        self.seat_circle = seat_circle
        self.not_allowed_region = not_allowed_region
        self.min_prob = min_prob

        # find the approximate max distance to the seat_circle
        self.find_max_dist2seat_circle()
        if 0:
            import matplotlib.pyplot as plt
            points = np.asarray(self.get_sample(10000))
            room = sig.signal.geometry.Rectangle([0, self.room_size[0]], [0, self.room_size[1]])
            room.draw()
            self.not_allowed_region.draw()
            self.seat_circle.draw()
            plt.plot(points[:,0], points[:, 1], '*')
            plt.show()
            self.show_acceptance_rate()

    def get_grid_dist(self, N=100):
        x = np.linspace(0, self.room_size[0], N)
        y = np.linspace(0, self.room_size[1], N)
        xv, yv = np.meshgrid(x, y)

        dist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                point = [xv[i, j], yv[i, j]]
                if not self.not_allowed_region.is_inside(point):
                    dist[i, j] = self.seat_circle.dist2point(point)
        return dist

    def find_max_dist2seat_circle(self):
        dist = self.get_grid_dist()
        self.max_dist = np.max(dist)
        self.scale = -np.log(self.min_prob) / self.max_dist

    def get_sample(self, n_sample=1):
        accepted_points = []
        while len(accepted_points)<n_sample:
            x = self.x_sampler.get_sample()[0]
            y = self.y_sampler.get_sample()[0]
            point = [x,y]
            if self.not_allowed_region.is_inside(point):
                continue
            #sig.signal.geometry.illustrate_dist2rectangle()
            dist2circle = self.seat_circle.dist2point(point)
            acceptance_rate =self.dist2acceptance(dist2circle)
            if acceptance_rate > np.random.uniform():
                accepted_points.append(point)

        return accepted_points

    def dist2acceptance(self, dist):
        return np.exp(-dist * self.scale)

    def show_acceptance_rate(self):
        dist = self.get_grid_dist()
        acceptance_rate = self.dist2acceptance(dist)

        import matplotlib.pyplot as plt
        plt.figure()
        sig.signal.utils.imagesc(dist, show_color_bar=True, title='distance', colormap='gray')
        plt.figure()
        sig.signal.utils.imagesc(acceptance_rate, show_color_bar=True, title='acceptance', colormap='gray')