from simulator import generate_streams, find_intersections_parallel
from pathlib import Path
import pandas as pd
import numpy as np

# Constants
VOLUME_SIZE = 2000  # Match the value from simulator
CENTER_POSITION = np.array([[VOLUME_SIZE / 2, VOLUME_SIZE / 2, VOLUME_SIZE / 2]])
DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)

# Step 1: Generate streams
print("🔄 Generating 1,000,000 4D streams...")
streams = generate_streams(1_000_000)

# Step 2: Detect visible matter and dynamic black holes
print("🔍 Finding intersections and black holes...")
visible_matter, black_holes = find_intersections_parallel(streams, n_jobs=2)

# Step 3: Add the fixed central black hole
black_holes = np.vstack([CENTER_POSITION, black_holes]) if black_holes.size else CENTER_POSITION

# Step 4: Report results
print(f"✅ Locked (visible) matter count: {len(visible_matter)}")
print(f"🌀 Black hole count: {len(black_holes)} (including central core at {CENTER_POSITION[0]})")

# Step 5: Save results to CSV
pd.DataFrame(visible_matter, columns=["x", "y", "z"]).to_csv(DATA_PATH / "visible_matter.csv", index=False)
pd.DataFrame(black_holes, columns=["x", "y", "z"]).to_csv(DATA_PATH / "black_holes.csv", index=False)
