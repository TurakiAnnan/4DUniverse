from simulator import (
    generate_streams,
    find_intersections,
)
from visualizer import show_simulation
import pandas as pd

# Step 1: Create 100,000 4D streams
streams = generate_streams(200000)

# Step 2: First pass to detect visible matter and black holes
visible_matter, black_holes = find_intersections(streams)

# Step 3: Show results
print(f"✅ Locked (visible) matter count: {len(visible_matter)}")
print(f"🌀 Black hole count: {len(black_holes)}")

# Step 4: Visualize the simulation
show_simulation(streams, visible_matter, black_holes)

# Step 5: Save to CSV
pd.DataFrame(visible_matter, columns=["x", "y", "z"]).to_csv("data/visible_matter.csv", index=False)
if len(black_holes) > 0:
    pd.DataFrame(black_holes, columns=["x", "y", "z"]).to_csv("data/black_holes.csv", index=False)
