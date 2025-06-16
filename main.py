# main.py

from simulator import generate_streams, find_intersections
from visualizer import show_simulation
from exporter import export_locked_points
from config import run_label
import os

def main():
    print(f"🔭 Running simulation: {run_label}")

    # Generate 4D streams
    streams = generate_streams()

    # Find 4D collisions that become 'visible matter'
    locked = find_intersections(streams)

    print(f"✅ Locked (visible) matter count: {len(locked)}")

    # Save output to a data folder
    os.makedirs("data", exist_ok=True)
    export_file = f"data/visible_matter_{run_label}.csv"
    export_locked_points(locked, filename=export_file)
    print(f"📁 Saved visible matter to {export_file}")

    # Launch 3D visualization
    show_simulation(streams, locked)

if __name__ == "__main__":
    main()
