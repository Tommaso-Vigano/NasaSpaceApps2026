from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.adapters.implementations.map_adapter import MapAdapter


def main():
    path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "resources"
            / "M20_JezeroCrater_CTXDEM_20m.tif"
    )

    print("starting to adapt!")

    terrain = MapAdapter().adapt(path)

    print("Shape:", terrain.shape)
    print("Granularity:", terrain.granularity)
    print("Coordinate type:", terrain.value_type)
    print("Top-left cell:", terrain.top_left)
    print("Bottom-right cell:", terrain.bottom_right)

    valid = np.isfinite(terrain.elevation)
    print("Valid cells:", np.count_nonzero(valid))
    print("Missing cells:", terrain.elevation.size - np.count_nonzero(valid))

    if not valid.any():
        raise ValueError("The map contains no valid elevations.")

    print("Minimum elevation:", np.nanmin(terrain.elevation))
    print("Maximum elevation:", np.nanmax(terrain.elevation))

    # Try retrieving the centre cell's coordinates and elevation.
    rows, columns = terrain.shape
    row, column = rows // 2, columns // 2

    print("Centre cell (x, y, z):", terrain.get_point(row, column))

    # Show the elevation matrix.
    plt.figure(figsize=(9, 7))
    plt.imshow(
        terrain.elevation,
        cmap="terrain",
        origin="upper",
        interpolation="nearest",
    )
    plt.colorbar(label="Elevation (m)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.title("Jezero crater — elevation")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()