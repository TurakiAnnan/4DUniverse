def show_simulation(streams, visible_matter, black_holes=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # Plot all streams (faint)
    ax.scatter(*streams[:, :3].T, s=1, alpha=0.05, label="Streams")

    # Plot visible matter (red)
    if len(visible_matter) > 0:
        ax.scatter(*visible_matter.T, color='red', s=10, label="Visible Matter")

    # Plot black holes (black)
    if black_holes is not None and len(black_holes) > 0:
        ax.scatter(*black_holes.T, color='black', s=30, label="Black Holes")

    ax.set_title("4D Stream Intersection Simulation")
    ax.legend()
    plt.show()
