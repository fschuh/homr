from dataclasses import dataclass

import numpy as np

from homr.staff_dewarping import StaffDewarping

# Newton steps for inverting the canvas mapping. The dewarp is a mild piecewise affine
# correction around an affine scaling, so a handful of steps reaches sub-pixel accuracy.
_MAX_INVERSE_STEPS = 12
_INVERSE_TOLERANCE = 0.05


@dataclass(frozen=True)
class StaffCanvasTransform:
    """The point mapping between prediction space and the canvas TrOMR reads for one staff.

    ``prepare_staff_image`` moves a staff through a region crop and scale, a dewarp, a
    second crop, and a canvas resize. It applies those steps to staffs and notes one
    point at a time; this records the same steps so that a point the transformer reports
    on its canvas, such as an attention coordinate, can be carried back to the page.
    """

    region_top_left: tuple[float, float]
    region_scaling: float
    dewarp: StaffDewarping
    crop_top_left: tuple[float, float]
    canvas_scaling: tuple[float, float]
    canvas_y_offset: float

    def to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        x = (point[0] - self.region_top_left[0]) * self.region_scaling
        y = (point[1] - self.region_top_left[1]) * self.region_scaling
        x, y = self.dewarp.dewarp_point((x - self.crop_top_left[0], y - self.crop_top_left[1]))
        return (
            x * self.canvas_scaling[0],
            y * self.canvas_scaling[1] + self.canvas_y_offset,
        )

    def to_prediction(self, point: tuple[float, float]) -> tuple[float, float] | None:
        """Invert ``to_canvas``; None when no prediction point maps onto ``point``."""
        scale_x = self.region_scaling * self.canvas_scaling[0]
        scale_y = self.region_scaling * self.canvas_scaling[1]
        if scale_x <= 0 or scale_y <= 0:
            return None
        target = np.array(point, dtype=np.float64)
        # The affine part alone gives the starting point; the dewarp is then corrected
        # with its residual, using the affine scale as the Jacobian estimate.
        estimate = np.array(
            [
                (target[0] / self.canvas_scaling[0] + self.crop_top_left[0]) / self.region_scaling
                + self.region_top_left[0],
                (
                    (target[1] - self.canvas_y_offset) / self.canvas_scaling[1]
                    + self.crop_top_left[1]
                )
                / self.region_scaling
                + self.region_top_left[1],
            ]
        )
        for _ in range(_MAX_INVERSE_STEPS):
            residual = target - np.array(self.to_canvas((estimate[0], estimate[1])))
            if np.all(np.abs(residual) < _INVERSE_TOLERANCE):
                return float(estimate[0]), float(estimate[1])
            estimate = estimate + residual / np.array([scale_x, scale_y])
        residual = target - np.array(self.to_canvas((estimate[0], estimate[1])))
        # A dewarp seam can leave no exact preimage; accept the closest point within a pixel.
        if np.all(np.abs(residual) < 1.0):
            return float(estimate[0]), float(estimate[1])
        return None
