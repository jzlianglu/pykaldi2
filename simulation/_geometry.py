import numpy as np
from . import _sampling


class SoundSourceConfig:
    def __init__(self):
        self.config = dict()
        self.config['num_spk'] = _sampling.get_distribution_template(
            'distribution of how many speech sources to use in the simulation', max=3,
            min=1, distribution='uniform_int')
        self.config['position_scheme'] = "random_coordinate"  # [minimum_angle|random_coordinate]
        self.config['between_source_angle'] = _sampling.get_distribution_template(
            'distribution of angles between sources in degrees',
            max=180, min=20, distribution='uniform')
        self.config['min_dist_from_wall'] = [0.5, 0.5]  # the minimum distance between speech source and a wall
        self.config['min_dist_from_array'] = [0.3]  # the minimum distance between speech source and array center
        self.config['min_dist_from_other'] = [0.5]  # the minimum distance between two speech sources
        self.config['height'] = _sampling.get_distribution_template('distribution of the height of speech sources in meters', max=2,
                                                                min=1, distribution='uniform')


class ArrayPositionConfig:
    def __init__(self, mic_positions):
        self.config = dict()
        self.config['mic_positions'] = mic_positions
        self.config['position_scheme'] = 'ratio'
        self.config['length_ratio'] = _sampling.get_distribution_template(
            'distribution of array position in length axis (percentage)',
            max=0.8, min=0.2, distribution='uniform')
        self.config['width_ratio'] = _sampling.get_distribution_template(
            'distribution of array position in width axis (percentage)',
            max=0.8, min=0.2, distribution='uniform')
        self.config['height_ratio'] = _sampling.get_distribution_template(
            'distribution of array position in height axis (percentage)',
            max=0.6, min=0.4, distribution='uniform')


class RoomConfig:
    def __init__(self):
        self.config = dict()
        self.config['length'] = _sampling.get_distribution_template('distribution of room length in meters',
                                                          max=20, min=2.5, distribution='uniform')
        self.config['width'] = _sampling.get_distribution_template('distribution of room width in meters',
                                                         max=20, min=2.5, distribution='uniform')
        self.config['height'] = _sampling.get_distribution_template('distribution of room height in meters',
                                                          max=5, min=2.5, distribution='uniform')


class Rectangle:
    """
    A class to model rectangular shape in a room, e.g. a table.
    Assume that the edges of the rectangle is parallel to the coordinate system, and the rectangular is in the x-y plane.
    """

    def __init__(self, x_range, y_range):
        """
        x_range: 1D array with two elements: minimum and maximum values of the x-coordiate of the rectangular
        y_range: similar to x_range, but for y-coordinate
        """
        self._x_range = x_range
        self._y_range = y_range

    def is_inside(self, point):
        """
        Decide whether the given point is within the rectangular. Z-coordinate of the point is ignored.
        point: a 1D array containing [x,y] or [x,y,z] coordinates of the point to be tested.
        """
        return not (np.prod(point[0] - self._x_range) > 0 or np.prod(point[1] - self._y_range) > 0)

    def dist2point(self, point):
        """
        Find the minimum distance from the edges of the rectangular to a given point.
        """
        if not self.is_inside(point):  # point outside of rectangle
            dx = np.max([self._x_range[0] - point[0], 0, point[0] - self._x_range[1]])
            dy = np.max([self._y_range[0] - point[1], 0, point[1] - self._y_range[1]])
            dist = np.linalg.norm(dx * dx + dy * dy)
        else:
            dx = np.min(np.abs(point[0]-self._x_range))
            dy = np.min(np.abs(point[1] - self._y_range))
            dist = np.minimum(dx, dy)

        return dist

    def draw(self, style='b--'):
        """
        Draw the rectangular
        """
        import matplotlib.pyplot as plt
        idx = [[0,0], [1, 0], [1, 1], [0, 1]]
        corners = np.asarray([[self._x_range[idx[i][0]], self._y_range[idx[i][1]]] for i in range(len(idx))])

        for i in range(0, len(corners)):
            plt.plot(corners[i:i + 2, 0], corners[i:i + 2, 1], style)
        plt.plot([corners[-1, 0], corners[0, 0]], [corners[-1, 1], corners[0, 1]], style)
        plt.show()


def _illustrate_dist2rectangle():
    L = 10
    W = 10
    N = 100
    x = np.linspace(0, L, N)
    y = np.linspace(0, W, N)
    xv, yv = np.meshgrid(x, y)

    rect = Rectangle([L/4, L/4*3], [W/4, W/4*3])

    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist[i, j] = rect.dist2point([xv[i, j], yv[i, j]])

    from .simulation import imagesc
    imagesc(dist, show_color_bar=True, title='distance', colormap='gray')
    imagesc(np.exp(-dist / np.max(dist) * 2.0), show_color_bar=True, title='probability', colormap='gray')
