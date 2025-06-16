# visualizer.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def show_simulation(streams, locked):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Plot all 4D streams (as faint dots)
    ax.scatter(*streams[:, :3].T, s=2, alpha=0.05, label="Streams")

    # Plot locked (visible) matter in red
    if len(locked) > 0:
        ax.scatter(*locked.T, color='red', s=12, label="Visible Matter")

    ax.set_title("4D Stream Intersection Simulator")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()
