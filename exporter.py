# exporter.py

import csv

def export_locked_points(locked, filename="locked_matter.csv"):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])
        for pt in locked:
            writer.writerow(pt)
