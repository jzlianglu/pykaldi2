import numpy as np
from .simulation import imagesc


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

    imagesc(dist, show_color_bar=True, title='distance', colormap='gray')
    imagesc(np.exp(-dist / np.max(dist) * 2.0), show_color_bar=True, title='probability', colormap='gray')


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
            room = Rectangle([0, self.room_size[0]], [0, self.room_size[1]])
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
        imagesc(dist, show_color_bar=True, title='distance', colormap='gray')
        plt.figure()
        imagesc(acceptance_rate, show_color_bar=True, title='acceptance', colormap='gray')