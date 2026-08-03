from typing import Any

import cv2
import numpy as np

from homr.model import Note
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform

STRETCHED_NOTEHEAD_ASPECT_RATIO = 2.0


class NoteheadGeometry:
    def __init__(
        self,
        coordinate_transform: PredictionCoordinateTransform,
        notehead_mask: Any | None,
        source_image: Any | None,
    ) -> None:
        self.coordinate_transform = coordinate_transform
        self.notehead_mask = notehead_mask
        self.source_image = source_image

    def detected_notehead_contour(self, note: Note) -> list[list[float]]:
        return self._detected_notehead_contour(note)

    def ellipse_from_source_contour(self, contour: list[list[float]]) -> dict[str, Any]:
        return self._ellipse_from_source_contour(contour)

    def is_stretched_notehead(self, note: Note) -> bool:
        return self._is_stretched_notehead(note)

    def is_hollow_notehead(self, note: Note) -> bool:
        return self._is_hollow_notehead(note)

    def refined_notehead_contour(
        self, note: Note, neighboring_notes: list[Note]
    ) -> list[list[float]] | None:
        return self._refined_notehead_contour(note, neighboring_notes)

    def notehead_ellipse(self, note: Note) -> dict[str, Any]:
        return self._notehead_ellipse_for_visual_sidecar(note)

    def _detected_notehead_contour(self, note: Note) -> list[list[float]]:
        """Return the segmentation contour while preserving the legacy polygon separately."""
        mask_contour = self._notehead_mask_contour(note)
        contour = mask_contour if mask_contour is not None else note.box.contours
        return self.coordinate_transform.prediction_contour_to_source(contour)

    def _ellipse_from_source_contour(self, contour: list[list[float]]) -> dict[str, Any]:
        points = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
        ellipse = self.coordinate_transform._ellipse_to_json(cv2.fitEllipse(points))
        ellipse["_fit_source"] = "recovered"
        return ellipse

    @staticmethod
    def _is_stretched_notehead(note: Note) -> bool:
        height = max(float(note.box.size[1]), 1.0)
        return float(note.box.size[0]) / height > STRETCHED_NOTEHEAD_ASPECT_RATIO

    def _is_hollow_notehead(self, note: Note) -> bool:
        if self.source_image is None:
            return False
        image = self.source_image
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        polygon = np.asarray(note.box.polygon, dtype=np.int32).reshape(-1, 2)
        left = max(0, int(np.min(polygon[:, 0])))
        top = max(0, int(np.min(polygon[:, 1])))
        right = min(image.shape[1], int(np.max(polygon[:, 0])) + 1)
        bottom = min(image.shape[0], int(np.max(polygon[:, 1])) + 1)
        if right <= left or bottom <= top:
            return False
        local_polygon = polygon - np.array([left, top])
        mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
        cv2.fillPoly(mask, [local_polygon.reshape(-1, 1, 2)], 1)
        pixels = image[top:bottom, left:right][mask > 0]
        if len(pixels) == 0:
            return False
        return float(np.mean(pixels < 160)) < 0.7

    def _refined_notehead_contour(
        self, note: Note, neighboring_notes: list[Note]
    ) -> list[list[float]] | None:
        """Fit the outer notehead boundary to source-image darkness and contrast."""
        if self.source_image is None:
            return None
        image = self.source_image
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        expected_height = max(float(note.box.size[1]), 4.0)
        cx, cy = note.center
        close_neighbors = []
        for other in neighboring_notes:
            if other is note:
                continue
            delta_x = float(other.center[0] - cx)
            delta_y = float(other.center[1] - cy)
            distance = float(np.hypot(delta_x, delta_y))
            if 0 < distance <= expected_height * 2.2:
                close_neighbors.append((delta_x / distance, delta_y / distance, distance))
        radius_x = int(np.ceil(expected_height * 2.0))
        radius_y = int(np.ceil(expected_height * 1.5))
        left = max(0, int(round(cx)) - radius_x)
        right = min(image.shape[1], int(round(cx)) + radius_x + 1)
        top = max(0, int(round(cy)) - radius_y)
        bottom = min(image.shape[0], int(round(cy)) + radius_y + 1)
        if right - left < 7 or bottom - top < 7:
            return None

        darkness = 1.0 - image[top:bottom, left:right].astype(np.float32) / 255.0
        sample_angles = np.linspace(0, 2 * np.pi, 96, endpoint=False, dtype=np.float32)
        unit_x = np.cos(sample_angles)
        unit_y = np.sin(sample_angles)

        def sample(values_x: np.ndarray, values_y: np.ndarray) -> np.ndarray:
            ix = np.clip(np.rint(values_x - left).astype(np.int32), 0, darkness.shape[1] - 1)
            iy = np.clip(np.rint(values_y - top).astype(np.int32), 0, darkness.shape[0] - 1)
            return darkness[iy, ix]

        fill_angles = np.linspace(0, 2 * np.pi, 32, endpoint=False, dtype=np.float32)
        fill_unit_x = np.cos(fill_angles)
        fill_unit_y = np.sin(fill_angles)
        center_evidence = np.concatenate(
            [
                sample(
                    cx + expected_height * radius * fill_unit_x,
                    cy + expected_height * radius * fill_unit_y,
                )
                for radius in (0.08, 0.18, 0.28)
            ]
        )
        is_filled_notehead = float(np.mean(center_evidence)) >= 0.62

        def score(
            params: tuple[float, float, float, float, float],
            center_anchor: tuple[float, float] | None,
        ) -> float:
            center_x, center_y, radius_major, radius_minor, angle = params
            theta = np.deg2rad(angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            base_x = radius_major * unit_x
            base_y = radius_minor * unit_y

            def ring(scale: float) -> np.ndarray:
                return sample(
                    center_x + scale * (cos_t * base_x - sin_t * base_y),
                    center_y + scale * (sin_t * base_x + cos_t * base_y),
                )

            boundary = np.maximum.reduce([ring(0.94), ring(1.0), ring(1.06)])
            outside = np.minimum.reduce([ring(1.18), ring(1.26)])
            support = boundary * (1.0 - 0.65 * outside)
            # A staff or stem may occlude a small arc. A trimmed mean rewards the
            # supported outer boundary without letting those straight runs dominate.
            supported = np.sort(support)[10:]
            sectors = support.reshape(12, -1).max(axis=1)
            coverage = float(np.mean(sectors > 0.35))
            center_penalty = 0.0
            if center_anchor is not None:
                center_penalty = 0.025 * (
                    abs(center_x - center_anchor[0]) / expected_height
                    + abs(center_y - center_anchor[1]) / expected_height
                )
            neighbor_penalty = 0.0
            for direction_x, direction_y, distance in close_neighbors:
                local_direction_x = cos_t * direction_x + sin_t * direction_y
                local_direction_y = -sin_t * direction_x + cos_t * direction_y
                extent = np.sqrt(
                    (radius_major * local_direction_x) ** 2
                    + (radius_minor * local_direction_y) ** 2
                )
                center_shift = (center_x - cx) * direction_x + (center_y - cy) * direction_y
                # Adjacent chord heads may touch, but neither fitted boundary may
                # travel materially beyond the midpoint toward the other center.
                overflow = center_shift + extent - distance * 0.52
                neighbor_penalty += 1.5 * max(0.0, overflow / expected_height)
            interior_score = 0.0
            if is_filled_notehead:
                interior = np.concatenate([ring(radius) for radius in (0.12, 0.3, 0.48, 0.66)])
                interior_score = 0.32 * float(np.mean(interior))
            return float(
                np.mean(supported)
                + 0.22 * coverage
                + interior_score
                - center_penalty
                - neighbor_penalty
            )

        initial_angles = (-35.0, -20.0, -5.0, 10.0)
        detection_is_stretched = self._is_stretched_notehead(note)
        initial_major_ratio = 0.62 if detection_is_stretched else 0.72
        initial_minor_ratio = 0.46 if detection_is_stretched else 0.50
        anchored = max(
            (
                (
                    cx,
                    cy,
                    expected_height * initial_major_ratio,
                    expected_height * initial_minor_ratio,
                    angle,
                )
                for angle in initial_angles
            ),
            key=lambda item: score(item, None),
        )
        recovered = anchored
        if detection_is_stretched:
            center_candidates = []
            for offset_x in np.linspace(-0.7, 0.7, 8):
                for offset_y in np.linspace(-0.6, 0.6, 7):
                    for angle in initial_angles:
                        center_candidates.append(
                            (
                                cx + float(offset_x) * expected_height,
                                cy + float(offset_y) * expected_height,
                                expected_height * 0.62,
                                expected_height * 0.46,
                                angle,
                            )
                        )
            recovered_candidate = max(center_candidates, key=lambda item: score(item, None))
            if score(recovered_candidate, None) >= score(anchored, None) + 0.045:
                recovered = recovered_candidate
        recovered_center = (recovered[0], recovered[1])

        candidates = [
            (
                recovered_center[0],
                recovered_center[1],
                expected_height * initial_major_ratio,
                expected_height * initial_minor_ratio,
                angle,
            )
            for angle in initial_angles
        ]
        params = max(candidates, key=lambda item: score(item, recovered_center))
        center_limit_x = 0.28 if detection_is_stretched else 0.75
        center_limit_y = 0.28 if detection_is_stretched else 0.65
        major_min = 0.50 if detection_is_stretched else 0.52
        major_max = 0.84 if detection_is_stretched else 0.95
        steps = ((0.24, 12.0), (0.12, 6.0), (0.06, 3.0), (0.03, 1.5))
        for relative_step, angle_step in steps:
            center_step = expected_height * relative_step
            dimensions = (
                (
                    0,
                    center_step,
                    recovered_center[0] - expected_height * center_limit_x,
                    recovered_center[0] + expected_height * center_limit_x,
                ),
                (
                    1,
                    center_step,
                    recovered_center[1] - expected_height * center_limit_y,
                    recovered_center[1] + expected_height * center_limit_y,
                ),
                (2, center_step, expected_height * major_min, expected_height * major_max),
                (3, center_step, expected_height * 0.36, expected_height * 0.68),
                (4, angle_step, -55.0, 25.0),
            )
            for index, amount, minimum, maximum in dimensions:
                options = []
                for offset in (-2, -1, 0, 1, 2):
                    option = list(params)
                    option[index] = min(max(option[index] + offset * amount, minimum), maximum)
                    if option[3] * 1.05 <= option[2] <= option[3] * 1.85:
                        options.append(tuple(option))
                params = max(options, key=lambda item: score(item, recovered_center))

        if score(params, recovered_center) < 0.34:
            return None
        center = (params[0], params[1])
        size = (params[2] * 2, params[3] * 2)
        angle = params[4]
        contour = cv2.ellipse2Poly(
            (int(round(center[0])), int(round(center[1]))),
            (max(1, int(round(size[0] / 2))), max(1, int(round(size[1] / 2)))),
            int(round(angle)),
            0,
            360,
            3,
        ).reshape(-1, 1, 2)
        return self.coordinate_transform.prediction_contour_to_source(contour)

    def _notehead_ellipse_for_visual_sidecar(self, note: Note) -> dict[str, Any]:
        points = np.asarray(note.box.contours).reshape(-1, 2)
        if len(points) < 5:
            ellipse = self.coordinate_transform.prediction_ellipse_to_source(note.box.box)
            ellipse["_fit_source"] = "fallback"
            return ellipse

        contour = self._notehead_mask_contour(note)
        if contour is not None:
            ellipse = self.coordinate_transform.prediction_contour_ellipse_to_source(
                contour, note.box.box
            )
            ellipse["_fit_source"] = "mask"
            return ellipse
        ellipse = self.coordinate_transform.prediction_contour_ellipse_to_source(
            note.box.contours, note.box.box
        )
        ellipse["_fit_source"] = "contour"
        return ellipse

    def _notehead_mask_contour(self, note: Note) -> Any | None:
        if self.notehead_mask is None:
            return None

        height, width = self.notehead_mask.shape[:2]
        left = max(0, int(np.floor(note.box.top_left[0])) - 1)
        top = max(0, int(np.floor(note.box.top_left[1])) - 1)
        right = min(width, int(np.ceil(note.box.bottom_right[0])) + 1)
        bottom = min(height, int(np.ceil(note.box.bottom_right[1])) + 1)
        if right <= left or bottom <= top:
            return None

        region = self.notehead_mask[top:bottom, left:right]
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            return None
        return contour + np.array([[[left, top]]])
