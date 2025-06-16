from joblib import Parallel, delayed
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict

# Constants
VOLUME_SIZE = 1000
TIME_SPAN = 1000
GRAVITY_STRENGTH = 0.01
BLACK_HOLE_THRESHOLD = 50
BLACK_HOLE_RADIUS = 5
CENTER_BIAS_STRENGTH = 0.2
CENTER = np.array([VOLUME_SIZE / 2] * 3)

def generate_streams(n_streams):
    streams = np.random.rand(n_streams, 4)
    streams[:, :3] *= VOLUME_SIZE
    streams[:, 3] *= TIME_SPAN

    # Bias toward center
    directions = CENTER - streams[:, :3]
    scale = np.linalg.norm(directions, axis=1) / (VOLUME_SIZE / 2)
    streams[:, :3] += CENTER_BIAS_STRENGTH * directions * scale[:, np.newaxis]
    return streams

def process_chunk(chunk_idx, streams, time_window, radius):
    local_locked = []
    local_density = defaultdict(int)
    local_black_holes = []

    for i in chunk_idx:
        t_i = streams[i, 3]
        j = i + 1
        local_indices = []
        local_points = []

        while j < len(streams) and (streams[j, 3] - t_i) < time_window:
            local_indices.append(j)
            local_points.append(streams[j, :3])
            j += 1

        if local_points:
            local_points = np.array(local_points)
            tree = KDTree(local_points)
            neighbors = tree.query_ball_point(streams[i, :3], radius)

            for n in neighbors:
                midpoint = 0.5 * (streams[i, :3] + local_points[n])
                local_locked.append(midpoint)

                region = tuple((midpoint // BLACK_HOLE_RADIUS).astype(int))
                local_density[region] += 1

                if local_density[region] == BLACK_HOLE_THRESHOLD + 1:
                    local_black_holes.append(midpoint.tolist())

    return local_locked, local_black_holes

def find_intersections_parallel(streams, radius=4.5, time_window=3, n_jobs=2):
    n = len(streams)
    chunk_size = max(n // 8, 1)
    indices = [range(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk, streams, time_window, radius)
        for chunk in indices
    )

    all_locked = []
    all_black_holes = []

    for locked, blackholes in results:
        all_locked.extend(locked)
        all_black_holes.extend(blackholes)

    return np.array(all_locked), np.array(all_black_holes)
