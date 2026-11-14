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

    print("Shape:", terrain.values.shape)
    print("Granularity:", terrain.granularity)
    print("Value type:", terrain.value_type)
    print("Top-left cell:", terrain.top_left)
    print("Bottom-right cell:", terrain.bottom_right)

    valid = np.isfinite(terrain.values)
    print("Valid cells:", np.count_nonzero(valid))
    print("Missing cells:", terrain.values.size - np.count_nonzero(valid))

    if not valid.any():
        raise ValueError("The map contains no valid elevations.")

    print("Minimum elevation:", np.nanmin(terrain.values))
    print("Maximum elevation:", np.nanmax(terrain.values))

    # Try retrieving the centre cell's coordinates and elevation.
    rows, columns = terrain.values.shape
    row, column = rows // 2, columns // 2

    longitude = terrain.top_left[0] + column * terrain.granularity[0]
    latitude = terrain.top_left[1] - row * terrain.granularity[1]
    elevation = terrain.values[row, column]

    print("Longitude:", longitude)
    print("Latitude:", latitude)
    print("Elevation:", elevation)
    # Show the elevation matrix.
    plt.figure(figsize=(9, 7))
    plt.imshow(
        terrain.values,
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