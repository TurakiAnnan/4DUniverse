from simulator import generate_streams, find_intersections_parallel
from visualizer import show_simulation
import pandas as pd
from pathlib import Path

# Step 0: Ensure data directory exists
Path("data").mkdir(exist_ok=True)

# Step 1: Generate streams
print("🔄 Generating 100,000 4D streams...")
streams = generate_streams(100000)

# Step 2: Detect intersections and black holes in parallel
print("🔍 Finding intersections and black holes...")
visible_matter, black_holes = find_intersections_parallel(streams, n_jobs=2)

# Step 3: Report findings
print(f"✅ Locked (visible) matter count: {len(visible_matter)}")
print(f"🌀 Black hole count: {len(black_holes)}")

# Step 4: Show visualization (optional)
show_simulation(streams, visible_matter)

# Step 5: Save to CSV
pd.DataFrame(visible_matter, columns=["x", "y", "z"]).to_csv("data/visible_matter.csv", index=False)
if len(black_holes) > 0:
    pd.DataFrame(black_holes, columns=["x", "y", "z"]).to_csv("data/black_holes.csv", index=False)
else:
    pd.DataFrame(columns=["x", "y", "z"]).to_csv("data/black_holes.csv", index=False)
