import numpy as np
import scipy
from time import gmtime, strftime
import re
import os
import sys
sys.path.append("../..")
import signal_graph as sig


class Rectangle:
    def __init__(self, x_range, y_range):
        # assume that the edges of the rectangle is parallel to the coordinate system
        self.x_range = x_range
        self.y_range = y_range

    def is_inside(self, point):
        if np.prod(point[0] - self.x_range) > 0 or np.prod(point[1] - self.y_range) > 0:
            return False
        else:
            return True

    def dist2point(self, point):
        if not self.is_inside(point):  # point outside of rectangle
            dx = np.max([self.x_range[0] - point[0], 0, point[0] - self.x_range[1]])
            dy = np.max([self.y_range[0] - point[1], 0, point[1] - self.y_range[1]])
            dist = np.sqrt(dx * dx + dy * dy)
        else:
            dx = np.min(np.abs(point[0]-self.x_range))
            dy = np.min(np.abs(point[1] - self.y_range))
            dist = np.minimum(dx, dy)

        return dist

    def draw(self, style='b--'):
        import matplotlib.pyplot as plt
        idx = [[0,0], [1, 0], [1, 1], [0, 1]]
        corners = np.asarray([[self.x_range[idx[i][0]], self.y_range[idx[i][1]]] for i in range(len(idx))])

        for i in range(0, len(corners)):
            plt.plot(corners[i:i + 2, 0], corners[i:i + 2, 1], style)
        plt.plot([corners[-1, 0], corners[0, 0]], [corners[-1, 1], corners[0, 1]], style)


def illustrate_dist2rectangle():
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

    sig.signal.utils.imagesc(dist, show_color_bar=True, title='distance', colormap='gray')
    sig.signal.utils.imagesc(np.exp(-dist / np.max(dist) * 2.0), show_color_bar=True, title='probability', colormap='gray')
