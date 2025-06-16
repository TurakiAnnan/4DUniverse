from simulator import (
    generate_streams,
    apply_blackhole_gravity,
    apply_local_gravity,
    find_intersections,
)
from visualizer import show_simulation
import pandas as pd

# Step 1: Create 5000 4D streams
streams = generate_streams(5000)

# Step 2: Apply blackhole influence
streams = apply_blackhole_gravity(streams)

# Step 3: First pass of visible matter
visible_matter = find_intersections(streams)

# Step 4: Pull lines toward detected visible matter
if len(visible_matter) > 0:
    streams = apply_local_gravity(streams, visible_matter)

# Step 5: Final pass to get updated visible matter
final_visible = find_intersections(streams)

# Step 6: Show results
print(f"✅ Locked (visible) matter count: {len(final_visible)}")
show_simulation(streams, final_visible)

# Optional: Save visible matter to CSV
df = pd.DataFrame(final_visible, columns=["x", "y", "z"])
df.to_csv("data/visible_matter.csv", index=False)
