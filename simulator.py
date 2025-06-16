import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict

# Simulation constants
VOLUME_SIZE = 1000
TIME_SPAN = 1000
GRAVITY_STRENGTH = 0.01
BLACK_HOLE_THRESHOLD = 50       # Threshold for black hole formation
BLACK_HOLE_RADIUS = 5
CENTER_BIAS_STRENGTH = 0.2      # Controls how much streams are pulled toward center

def generate_streams(n_streams):
    streams = np.random.rand(n_streams, 4)
    streams[:, :3] *= VOLUME_SIZE
    streams[:, 3] *= TIME_SPAN

    # Bias stream positions toward center
    center = np.array([VOLUME_SIZE/2]*3)
    directions = center - streams[:, :3]
    streams[:, :3] += CENTER_BIAS_STRENGTH * directions * (np.linalg.norm(directions, axis=1)/ (VOLUME_SIZE/2))[:, np.newaxis]

    return streams

def find_intersections(streams, radius=4.5, time_window=3):
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

                region = tuple((midpoint // BLACK_HOLE_RADIUS).astype(int))
                density_map[region] += 1

                boost = GRAVITY_STRENGTH * (10 if density_map[region] > BLACK_HOLE_THRESHOLD else 1)

                neighbor_idx = local_indices[n]
                direction = midpoint - sorted_streams[neighbor_idx, :3]
                sorted_streams[neighbor_idx, :3] += boost * direction

                if density_map[region] == BLACK_HOLE_THRESHOLD + 1:
                    black_holes.append(midpoint)

    return np.array(locked), np.array(black_holes)
