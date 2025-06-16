import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict

# Simulation constants
VOLUME_SIZE = 1000
TIME_SPAN = 1000
GRAVITY_STRENGTH = 0.01
BLACK_HOLE_THRESHOLD = 50       # Threshold for black hole formation
BLACK_HOLE_RADIUS = 5
CENTER_BIAS_STRENGTH = 0.2      # Pull toward center of universe
CENTER_POSITION = np.array([0.0, 0.0, 0.0])  # Explicit central black hole

def generate_streams(n_streams):
    streams = np.random.rand(n_streams, 4)
    streams[:, :3] *= VOLUME_SIZE
    streams[:, 3] *= TIME_SPAN

    # Pull streams toward the center
    directions = CENTER_POSITION - streams[:, :3]
    norms = np.linalg.norm(directions, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero
    unit_directions = directions / norms[:, np.newaxis]
    streams[:, :3] += CENTER_BIAS_STRENGTH * unit_directions

    return streams

def find_intersections(streams, radius=4.5, time_window=3):
    locked = []
    black_holes = [CENTER_POSITION]  # Start with central black hole
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

                # Increase local gravity if more visible matter accumulates
                local_boost = GRAVITY_STRENGTH * (1 + density_map[region] / 10)

                neighbor_idx = local_indices[n]
                direction = midpoint - sorted_streams[neighbor_idx, :3]
                sorted_streams[neighbor_idx, :3] += local_boost * direction

                if density_map[region] == BLACK_HOLE_THRESHOLD + 1:
                    black_holes.append(midpoint)

    return np.array(locked), np.array(black_holes)
