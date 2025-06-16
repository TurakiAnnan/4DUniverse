# simulator.py

import numpy as np
from config import num_streams, volume_size, time_span, intersection_radius, time_window

def generate_streams():
    streams = np.random.rand(num_streams, 4)
    streams[:, :3] *= volume_size  # x, y, z
    streams[:, 3] *= time_span     # time
    return streams

def find_intersections(streams):
    locked = []
    for i in range(len(streams)):
        for j in range(i + 1, len(streams)):
            t_diff = abs(streams[i, 3] - streams[j, 3])
            if t_diff < time_window:
                dist = np.linalg.norm(streams[i, :3] - streams[j, :3])
                if dist < intersection_radius:
                    mid = 0.5 * (streams[i, :3] + streams[j, :3])
                    locked.append(mid)
    return np.array(locked)
