from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PredictionCoordinateTransform:
    source_image_size: tuple[int, int]
    autocrop_box: tuple[int, int, int, int]
    cropped_size: tuple[int, int]
    resized_size: tuple[int, int]
    resize_scale: tuple[float, float]
    prediction_size: tuple[int, int]

    def prediction_point_to_source(self, point: tuple[float, float]) -> tuple[float, float]:
        pred_w, pred_h = self.prediction_size
        resized_w, resized_h = self.resized_size
        crop_x, crop_y, _crop_w, _crop_h = self.autocrop_box
        x = point[0] * resized_w / pred_w
        y = point[1] * resized_h / pred_h
        return (x / self.resize_scale[0] + crop_x, y / self.resize_scale[1] + crop_y)

    def source_point_to_prediction(self, point: tuple[float, float]) -> tuple[float, float]:
        pred_w, pred_h = self.prediction_size
        resized_w, resized_h = self.resized_size
        crop_x, crop_y, _crop_w, _crop_h = self.autocrop_box
        x = (point[0] - crop_x) * self.resize_scale[0]
        y = (point[1] - crop_y) * self.resize_scale[1]
        return (x * pred_w / resized_w, y * pred_h / resized_h)

    def prediction_contour_to_source(self, contour: Any) -> list[list[float]]:
        points = np.asarray(contour).reshape(-1, 2)
        return [
            [round(x, 3), round(y, 3)]
            for x, y in (
                self.prediction_point_to_source((float(p[0]), float(p[1]))) for p in points
            )
        ]

    def _ellipse_to_json(self, ellipse: Any) -> dict[str, Any]:
        center, size, angle = ellipse
        width = float(size[0])
        height = float(size[1])
        if width >= height:
            rx = width / 2
            ry = height / 2
            svg_angle = float(angle)
        else:
            rx = height / 2
            ry = width / 2
            svg_angle = float(angle) + 90
        while svg_angle > 90:
            svg_angle -= 180
        while svg_angle <= -90:
            svg_angle += 180
        return {
            "center": [round(float(center[0]), 3), round(float(center[1]), 3)],
            "rx": round(rx, 3),
            "ry": round(ry, 3),
            "angle": round(svg_angle, 3),
        }

    def prediction_ellipse_to_source(self, ellipse: Any) -> dict[str, Any]:
        center, size, angle = ellipse
        pred_w, pred_h = self.prediction_size
        resized_w, resized_h = self.resized_size
        source_center = self.prediction_point_to_source((float(center[0]), float(center[1])))
        source_width = float(size[0]) * resized_w / pred_w / self.resize_scale[0]
        source_height = float(size[1]) * resized_h / pred_h / self.resize_scale[1]
        return self._ellipse_to_json((source_center, (source_width, source_height), angle))

    def prediction_contour_ellipse_to_source(
        self, contour: Any, fallback_ellipse: Any
    ) -> dict[str, Any]:
        source_points = np.asarray(self.prediction_contour_to_source(contour), dtype=np.float32)
        if len(source_points) >= 5:
            source_contour = source_points.reshape(-1, 1, 2)
            return self._ellipse_to_json(cv2.fitEllipse(source_contour))
        return self.prediction_ellipse_to_source(fallback_ellipse)
