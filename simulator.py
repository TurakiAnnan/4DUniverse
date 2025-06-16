# simulator.py
import numpy as np

# Constants
VOLUME_SIZE = 100
TIME_SPAN = 100
INTERSECTION_RADIUS = 2
TIME_WINDOW = 2
BLACKHOLE_CENTER = np.array([50, 50, 50])
BLACKHOLE_RADIUS = 30
BLACKHOLE_STRENGTH = 0.1
GRAVITY_PULL_FACTOR = 0.02

def generate_streams(n):
    streams = np.random.rand(n, 4)
    streams[:, :3] *= VOLUME_SIZE
    streams[:, 3] *= TIME_SPAN
    return streams

def apply_blackhole_gravity(streams):
    for i in range(len(streams)):
        dist = np.linalg.norm(streams[i, :3] - BLACKHOLE_CENTER)
        if dist < BLACKHOLE_RADIUS:
            direction = (BLACKHOLE_CENTER - streams[i, :3]) / (dist + 1e-5)
            streams[i, :3] += BLACKHOLE_STRENGTH * direction
    return streams

def apply_local_gravity(streams, visible_matter):
    for i in range(len(streams)):
        dists = np.linalg.norm(visible_matter - streams[i, :3], axis=1)
        close = visible_matter[dists < 10]
        if len(close) > 0:
            center = close.mean(axis=0)
            direction = center - streams[i, :3]
            streams[i, :3] += GRAVITY_PULL_FACTOR * direction
    return streams

def find_intersections(streams):
    locked = []
    for i in range(len(streams)):
        for j in range(i+1, len(streams)):
            if abs(streams[i, 3] - streams[j, 3]) < TIME_WINDOW:
                dist = np.linalg.norm(streams[i, :3] - streams[j, :3])
                if dist < INTERSECTION_RADIUS:
                    midpoint = 0.5 * (streams[i, :3] + streams[j, :3])
                    locked.append(midpoint)
    return np.array(locked)
