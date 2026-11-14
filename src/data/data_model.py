from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.basic import Coordinate

class WorldModel:
    """ Collection of all maps and data related to the world """


    def make_slice(self, top_left: Coordinate, bot_right: Coordinate) -> None:
        """ Extract a slice from the current collection of maps and data """



class SamplingUtils:

    @staticmethod
    def resample_to_window(
        m: Map,
        top_left: tuple[float, float],
        bottom_right: tuple[float, float],
        new_stride: float,
    ) -> Map:
        x_min, y_min = top_left
        x_max, y_max = bottom_right

        # Target grid in world coordinates
        xs = np.arange(x_min, x_max, new_stride)
        ys = np.arange(y_min, y_max, new_stride)

        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        # Original grid in world coordinates
        h, w = m.data.shape[:2]

        old_x = m.top_left[0] + np.arange(w) * m.stride
        old_y = m.top_left[1] + np.arange(h) * m.stride

        # Interpolate, filling outside regions with NaN
        interpolator = RegularGridInterpolator(
            (old_y, old_x),
            m.data.astype(float),
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        new_data = interpolator((yy, xx))

        return Map(
            data=new_data,
            stride=new_stride,
            top_left=top_left,
        )