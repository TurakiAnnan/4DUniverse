import numpy as np
from scipy.spatial import KDTree
import pandas as pd
from collections import defaultdict

# Simulation constants
VOLUME_SIZE = 1000
TIME_SPAN = 1000
GRAVITY_STRENGTH = 0.01
BLACK_HOLE_THRESHOLD = 50  # Number of collisions to form black hole
BLACK_HOLE_RADIUS = 5

def generate_streams(n_streams):
    streams = np.random.rand(n_streams, 4)
    streams[:, :3] *= VOLUME_SIZE
    streams[:, 3] *= TIME_SPAN
    return streams

def find_intersections(streams, radius=2, time_window=2):
    locked = []
    black_holes = []
    density_map = defaultdict(int)
    sorted_idx = np.argsort(streams[:, 3])
    sorted_streams = streams[sorted_idx]

    for i in range(len(sorted_streams)):
        t_i = sorted_streams[i, 3]
        j = i + 1
        local_indices = []
        local_points = []

        while j < len(sorted_streams) and (sorted_streams[j, 3] - t_i) < time_window:
            local_indices.append(j)
            local_points.append(sorted_streams[j, :3])
            j += 1

        if local_points:
            local_points = np.array(local_points)
            tree = KDTree(local_points)
            neighbors = tree.query_ball_point(sorted_streams[i, :3], radius)

            for n in neighbors:
                midpoint = 0.5 * (sorted_streams[i, :3] + local_points[n])
                locked.append(midpoint)

                # Track density
                region = tuple((midpoint // BLACK_HOLE_RADIUS).astype(int))
                density_map[region] += 1

                # Black hole gravity boost
                if density_map[region] > BLACK_HOLE_THRESHOLD:
                    black_holes.append(midpoint)
                    boost = GRAVITY_STRENGTH * 10
                else:
                    boost = GRAVITY_STRENGTH

                # Gravitational pull
                neighbor_idx = local_indices[n]
                direction = midpoint - sorted_streams[neighbor_idx, :3]
                sorted_streams[neighbor_idx, :3] += boost * direction

    return np.array(locked), np.array(black_holes)
