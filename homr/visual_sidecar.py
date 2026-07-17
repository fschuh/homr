import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr import constants
from homr.model import Note, Staff
from homr.note_detection import split_clumps_of_noteheads
from homr.transformer.vocabulary import EncodedSymbol


STRETCHED_NOTEHEAD_ASPECT_RATIO = 2.0
HORIZONTAL_HOLLOW_NOTEHEAD_ASPECT_RATIO = 1.8
MAX_RECONSTRUCTED_STEM_DISTANCE_IN_NOTEHEADS = 8.0


class StemRepairDirection(Enum):
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class PreprocessingMetadata:
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


@dataclass
class VisualGroup:
    visual_id: str
    staff_index: int
    staff_position: int
    prediction_center: tuple[float, float]
    transformer_center: tuple[float, float] | None
    notehead_ellipses: list[dict[str, Any]]
    notehead_contours: list[list[list[float]]]
    detected_notehead_contours: list[list[list[float]]]
    refined_notehead_contours: list[list[list[float]]]
    detected_stem_contours: list[list[list[float]]]
    stem_contours: list[list[list[float]]]
    owned_stem_component_ids: list[str]
    is_hollow_notehead: bool
    duration: str | None = None
    linked_musicxml_ids: list[str] = field(default_factory=list)

    @property
    def bbox(self) -> list[float]:
        points = [
            point for contour in self.notehead_contours + self.stem_contours for point in contour
        ]
        if not points:
            return []
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]


@dataclass
class MusicXmlNoteRecord:
    musicxml_id: str
    part: int
    measure: int
    staff: int
    voice: int
    pitch: str | None
    duration: str
    match_confidence: float
    visual_group_id: str | None


@dataclass
class VisualMatch:
    symbol: EncodedSymbol
    visual_id: str | None
    confidence: float


@dataclass
class StemOwnershipCache:
    component_by_fragment_id: dict[int, int]
    owner_note_ids_by_component: dict[int, set[int]]


class VisualSidecar:
    def __init__(
        self,
        metadata: PreprocessingMetadata,
        stem_fragments: list[RotatedBoundingBox] | None = None,
        notehead_mask: Any | None = None,
        notehead_candidates: list[Any] | None = None,
        source_image: Any | None = None,
    ) -> None:
        self.metadata = metadata
        self.stem_fragments = stem_fragments or []
        self.notehead_mask = notehead_mask
        self.notehead_candidates = notehead_candidates or []
        self.source_image = source_image
        self._recovery_notes_by_staff_id: dict[int, list[Note]] = {}
        self._stem_ownership_cache: StemOwnershipCache | None = None
        self.visual_groups: dict[str, VisualGroup] = {}
        self.matches_by_symbol_id: dict[int, VisualMatch] = {}
        self.musicxml_notes: list[MusicXmlNoteRecord] = []
        self.unmatched_visual_notes: set[str] = set()
        self._next_musicxml_note_id = 1
        self._next_recovered_visual_id = 1

    def prepare_recovery_notes(self, staffs: list[Staff]) -> None:
        """Assign real, inference-excluded candidates to their nearest staff.

        These notes are used exclusively for sidecar geometry. They are never added to
        ``staff.symbols`` and therefore cannot affect TrOMR inference or MusicXML output.
        """
        self._recovery_notes_by_staff_id = {id(staff): [] for staff in staffs}
        existing_notes = [note for staff in staffs for note in staff.get_notes()]
        for candidate in self.notehead_candidates:
            notehead = candidate.notehead
            eligible: list[Staff] = []
            for staff in staffs:
                if not staff.get_notes():
                    continue
                if not (
                    staff.min_x - constants.staff_position_tolerance
                    <= notehead.center[0]
                    <= staff.max_x + constants.staff_position_tolerance
                ):
                    continue
                sidecar_tolerance = (
                    constants.max_number_of_ledger_lines + 1
                ) * staff.average_unit_size
                if (
                    notehead.center[1] >= staff.min_y - sidecar_tolerance
                    and notehead.center[1] <= staff.max_y + sidecar_tolerance
                ):
                    eligible.append(staff)
            if not eligible:
                continue
            staff = min(
                eligible,
                key=lambda item: min(
                    np.linalg.norm(np.subtract(note.center, notehead.center))
                    for note in item.get_notes()
                ),
            )
            split_candidates = (
                split_clumps_of_noteheads(candidate, self.notehead_mask, staff)
                if self.notehead_mask is not None
                else [candidate]
            )
            for split_candidate in split_candidates:
                split_notehead = split_candidate.notehead
                if any(split_notehead.is_overlapping(note.box) for note in existing_notes):
                    continue
                point = staff.get_at(split_notehead.center[0])
                unit_size = point.average_unit_size if point is not None else staff.average_unit_size
                if (
                    split_notehead.size[0] < 0.45 * unit_size
                    # Keep the inference filter untouched. Sidecar recovery gets a
                    # small allowance for mask contours that include ledger-line ink.
                    or split_notehead.size[0] > 3.25 * unit_size
                    or split_notehead.size[1] < 0.45 * unit_size
                    or split_notehead.size[1] > 2 * unit_size
                ):
                    continue
                if point is not None:
                    position = point.find_position_in_unit_sizes(split_notehead)
                else:
                    nearest_point = min(
                        staff.grid, key=lambda item: abs(item.x - split_notehead.center[0])
                    )
                    position = nearest_point.find_position_in_unit_sizes(split_notehead)
                visual_id = f"vnote-recovered-{self._next_recovered_visual_id}"
                self._next_recovered_visual_id += 1
                self._recovery_notes_by_staff_id[id(staff)].append(
                    Note(
                        split_notehead,
                        position,
                        split_candidate.stem,
                        split_candidate.stem_direction,
                        visual_id,
                    )
                )
        all_visual_notes = [note for staff in staffs for note in staff.get_notes()]
        all_visual_notes.extend(
            note
            for recovered in self._recovery_notes_by_staff_id.values()
            for note in recovered
        )
        self._stem_ownership_cache = self._build_stem_ownership_cache(all_visual_notes)

    def recovery_notes_for_staff(self, staff: Staff) -> list[Note]:
        return self._recovery_notes_by_staff_id.get(id(staff), [])

    def add_staff_visual_notes(
        self, staff_index: int, original_notes: list[Note], transformed_notes: list[Note]
    ) -> None:
        stem_ownership = self._stem_ownership_cache or self._build_stem_ownership_cache(
            original_notes
        )
        for original, transformed in zip(original_notes, transformed_notes, strict=False):
            if original.visual_id is None:
                continue
            notehead_contour = self.metadata.prediction_contour_to_source(original.box.polygon)
            detected_notehead_contour = self._detected_notehead_contour(original)
            refined_notehead_contour = self._refined_notehead_contour(original, original_notes)
            recovered_stretched_notehead = (
                self._is_stretched_notehead(original) and refined_notehead_contour is not None
            )
            is_hollow_notehead = self._is_hollow_notehead(original)
            if recovered_stretched_notehead:
                notehead_contour = refined_notehead_contour
                notehead_ellipse = self._ellipse_from_source_contour(refined_notehead_contour)
            else:
                notehead_ellipse = self._notehead_ellipse_for_visual_sidecar(original)
            notehead_ellipse["_is_hollow"] = is_hollow_notehead
            detected_stem_contours = []
            if original.stem is not None:
                detected_stem_contours.append(
                    self.metadata.prediction_contour_to_source(original.stem.contours)
                )
            stem_contours = []
            stem = self._visual_sidecar_stem_for_note(original, stem_ownership)
            if stem is not None:
                stem_contours.append(
                    self.metadata.prediction_contour_to_source(stem.polygon)
                )
            owned_stem_component_ids = sorted(
                f"staff-{staff_index}-stem-{component}"
                for component, owner_note_ids in stem_ownership.owner_note_ids_by_component.items()
                if id(original) in owner_note_ids and len(owner_note_ids) > 1
            )
            self.visual_groups[original.visual_id] = VisualGroup(
                visual_id=original.visual_id,
                staff_index=staff_index,
                staff_position=original.position,
                prediction_center=original.center,
                transformer_center=transformed.center,
                notehead_ellipses=[notehead_ellipse],
                notehead_contours=[notehead_contour],
                detected_notehead_contours=[detected_notehead_contour],
                refined_notehead_contours=(
                    [refined_notehead_contour] if refined_notehead_contour is not None else []
                ),
                detected_stem_contours=detected_stem_contours,
                stem_contours=stem_contours,
                owned_stem_component_ids=owned_stem_component_ids,
                is_hollow_notehead=is_hollow_notehead,
            )
            self.unmatched_visual_notes.add(original.visual_id)

    def _detected_notehead_contour(self, note: Note) -> list[list[float]]:
        """Return the segmentation contour while preserving the legacy polygon separately."""
        mask_contour = self._notehead_mask_contour(note)
        contour = mask_contour if mask_contour is not None else note.box.contours
        return self.metadata.prediction_contour_to_source(contour)

    def _ellipse_from_source_contour(self, contour: list[list[float]]) -> dict[str, Any]:
        points = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
        ellipse = self.metadata._ellipse_to_json(cv2.fitEllipse(points))
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
                interior = np.concatenate(
                    [
                        ring(radius)
                        for radius in (0.12, 0.3, 0.48, 0.66)
                    ]
                )
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
        return self.metadata.prediction_contour_to_source(contour)

    def _raw_stem_contours_for_output(self) -> list[dict[str, Any]]:
        return [
            {
                "debug_id": stem.debug_id,
                "contour": self.metadata.prediction_contour_to_source(stem.contours),
                "bbox": self.metadata.prediction_contour_to_source(stem.polygon),
            }
            for stem in self.stem_fragments
        ]

    def add_staff_matches(self, symbols: list[EncodedSymbol], staff_index: int) -> None:
        visual_groups = [
            group for group in self.visual_groups.values() if group.staff_index == staff_index
        ]
        note_symbols = [symbol for symbol in symbols if symbol.rhythm.startswith("note")]

        def valid_point(point: tuple[float, float] | None) -> bool:
            return point is not None and bool(np.all(np.isfinite(point)))

        candidates: list[tuple[float, int, int]] = []
        for symbol_index, symbol in enumerate(note_symbols):
            if not valid_point(symbol.coordinates):
                continue
            assert symbol.coordinates is not None
            for group_index, group in enumerate(visual_groups):
                if not valid_point(group.transformer_center):
                    continue
                assert group.transformer_center is not None
                distance = float(
                    np.linalg.norm(np.subtract(symbol.coordinates, group.transformer_center))
                )
                candidates.append((distance, symbol_index, group_index))

        assigned_symbols: set[int] = set()
        assigned_groups: set[int] = set()
        assignments: list[tuple[int, int]] = []
        for _, symbol_index, group_index in sorted(candidates):
            if symbol_index in assigned_symbols or group_index in assigned_groups:
                continue
            assigned_symbols.add(symbol_index)
            assigned_groups.add(group_index)
            assignments.append((symbol_index, group_index))

        unmatched_symbol_indices = [
            index for index in range(len(note_symbols)) if index not in assigned_symbols
        ]
        unmatched_group_indices = [
            index for index in range(len(visual_groups)) if index not in assigned_groups
        ]
        unmatched_group_indices.sort(
            key=lambda index: visual_groups[index].transformer_center or (0.0, 0.0)
        )
        for symbol_index, group_index in zip(
            unmatched_symbol_indices, unmatched_group_indices, strict=False
        ):
            assignments.append((symbol_index, group_index))
            assigned_symbols.add(symbol_index)
            assigned_groups.add(group_index)

        for symbol_index, group_index in assignments:
            symbol = note_symbols[symbol_index]
            visual_group = visual_groups[group_index]
            confidence = self._score_match(symbol, visual_group)
            visual_group.duration = symbol.rhythm
            self.matches_by_symbol_id[id(symbol)] = VisualMatch(
                symbol=symbol,
                visual_id=visual_group.visual_id,
                confidence=confidence,
            )
            self.unmatched_visual_notes.discard(visual_group.visual_id)

        for symbol_index, symbol in enumerate(note_symbols):
            if symbol_index not in assigned_symbols:
                self.matches_by_symbol_id[id(symbol)] = VisualMatch(
                    symbol=symbol,
                    visual_id=None,
                    confidence=0.0,
                )

    def _stem_component_ids_for_output(self, group: VisualGroup) -> list[str]:
        if group.duration is None:
            return []
        result = []
        for component_id in group.owned_stem_component_ids:
            if any(
                candidate.visual_id != group.visual_id
                and candidate.duration == group.duration
                and component_id in candidate.owned_stem_component_ids
                for candidate in self.visual_groups.values()
            ):
                result.append(f"{component_id}-duration-{group.duration}")
        return result

    def create_musicxml_id(self) -> str:
        musicxml_id = f"homr-note-{self._next_musicxml_note_id}"
        self._next_musicxml_note_id += 1
        return musicxml_id

    def record_musicxml_note(
        self,
        musicxml_id: str,
        part: int,
        measure: int,
        staff: int,
        voice: int,
        symbol: EncodedSymbol,
    ) -> None:
        match = self.matches_by_symbol_id.get(id(symbol))
        visual_id = match.visual_id if match is not None else None
        confidence = match.confidence if match is not None else 0.0
        pitch = symbol.pitch if symbol.pitch not in ("_", ".") else None
        if visual_id is not None and visual_id in self.visual_groups:
            self.visual_groups[visual_id].linked_musicxml_ids.append(musicxml_id)
        self.musicxml_notes.append(
            MusicXmlNoteRecord(
                musicxml_id=musicxml_id,
                part=part,
                measure=measure,
                staff=staff,
                voice=voice,
                pitch=pitch,
                duration=symbol.rhythm,
                match_confidence=confidence,
                visual_group_id=visual_id,
            )
        )

    def unmatched_musicxml_notes(self) -> list[str]:
        return [note.musicxml_id for note in self.musicxml_notes if note.visual_group_id is None]

    def _notehead_ellipse_for_visual_sidecar(self, note: Note) -> dict[str, Any]:
        points = np.asarray(note.box.contours).reshape(-1, 2)
        if len(points) < 5:
            ellipse = self.metadata.prediction_ellipse_to_source(note.box.box)
            ellipse["_fit_source"] = "fallback"
            return ellipse

        contour = self._notehead_mask_contour(note)
        if contour is not None:
            ellipse = self.metadata.prediction_contour_ellipse_to_source(contour, note.box.box)
            ellipse["_fit_source"] = "mask"
            return ellipse
        ellipse = self.metadata.prediction_contour_ellipse_to_source(
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

    def _visual_sidecar_stem_for_note(
        self, note: Note, stem_ownership: StemOwnershipCache
    ) -> RotatedBoundingBox | None:
        stem_fragments = self._available_stem_fragments_for_note(note, stem_ownership)
        seed = self._best_visual_sidecar_stem_seed(note, stem_fragments)
        stem = (
            self._merge_visual_sidecar_stem_fragments(note, seed, stem_fragments)
            if seed is not None
            else None
        )
        if stem is not None and stem.center[0] >= note.center[0]:
            stem = self._repair_upward_visual_sidecar_stem(note, stem, stem_fragments)
            return self._repair_downward_visual_sidecar_stem(note, stem, stem_fragments)
        stem = self._repair_downward_visual_sidecar_stem(note, stem, stem_fragments)
        return self._repair_upward_visual_sidecar_stem(note, stem, stem_fragments)

    def _available_stem_fragments_for_note(
        self, note: Note, stem_ownership: StemOwnershipCache
    ) -> list[RotatedBoundingBox]:
        note_id = id(note)
        x_radius = max(20.0, float(note.box.size[0]) * 1.25)
        return [
            stem
            for stem in self.stem_fragments
            if abs(stem.center[0] - note.center[0]) <= x_radius
            if not (
                owners := stem_ownership.owner_note_ids_by_component.get(
                    stem_ownership.component_by_fragment_id[id(stem)], set()
                )
            )
            or note_id in owners
        ]

    def _build_stem_ownership_cache(
        self, notes: list[Note]
    ) -> StemOwnershipCache:
        if not self.stem_fragments:
            return StemOwnershipCache({}, {})

        widths = [float(note.box.size[0]) for note in notes]
        heights = [float(note.box.size[1]) for note in notes]
        x_tolerance = max(4.0, float(np.median(widths)) * 0.6) if widths else 4.0
        max_vertical_gap = max(4.0, float(np.median(heights))) if heights else 4.0

        parent = list(range(len(self.stem_fragments)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        sorted_indices = sorted(
            range(len(self.stem_fragments)),
            key=lambda index: self.stem_fragments[index].center[0],
        )
        for position, first_index in enumerate(sorted_indices):
            first = self.stem_fragments[first_index]
            next_position = position + 1
            while next_position < len(sorted_indices):
                second_index = sorted_indices[next_position]
                second = self.stem_fragments[second_index]
                if second.center[0] - first.center[0] > x_tolerance:
                    break
                if self._is_collinear_stem_fragment(
                    second, [first], x_tolerance, max_vertical_gap
                ):
                    union(first_index, second_index)
                next_position += 1

        component_by_fragment_id = {
            id(stem): find(index) for index, stem in enumerate(self.stem_fragments)
        }
        owner_note_ids_by_component: dict[int, set[int]] = {}
        notes_by_x = sorted(notes, key=lambda note: note.center[0])
        note_xs = [note.center[0] for note in notes_by_x]
        max_note_half_width = max(
            (float(note.box.size[0]) / 2 for note in notes_by_x), default=0.0
        )
        for index, stem in enumerate(self.stem_fragments):
            component = find(index)
            owners = owner_note_ids_by_component.setdefault(component, set())
            stem_bounds = self._stem_bounds(stem)
            x_padding = max_note_half_width + 2.0
            first_note = bisect_left(note_xs, stem_bounds[0] - x_padding)
            last_note = bisect_right(note_xs, stem_bounds[1] + x_padding)
            for note in notes_by_x[first_note:last_note]:
                if self._stem_bounds_touch_notehead(stem_bounds, note) or (
                    note.stem is not None and self._same_stem_fragment(stem, note.stem)
                ):
                    owners.add(id(note))

        return StemOwnershipCache(component_by_fragment_id, owner_note_ids_by_component)

    def _same_stem_fragment(
        self, first: RotatedBoundingBox, second: RotatedBoundingBox
    ) -> bool:
        return first is second or (
            np.allclose(first.center, second.center, atol=1.0)
            and np.allclose(first.size, second.size, atol=1.0)
        )

    def _stem_touches_notehead(self, stem: RotatedBoundingBox, note: Note) -> bool:
        return self._stem_bounds_touch_notehead(self._stem_bounds(stem), note)

    def _stem_bounds(
        self, stem: RotatedBoundingBox
    ) -> tuple[float, float, float, float]:
        points = np.asarray(stem.polygon, dtype=np.float32).reshape(-1, 2)
        return (
            float(np.min(points[:, 0])),
            float(np.max(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 1])),
        )

    def _stem_bounds_touch_notehead(
        self, stem_bounds: tuple[float, float, float, float], note: Note
    ) -> bool:
        stem_left, stem_right, stem_top, stem_bottom = stem_bounds
        padding = 2.0
        note_left = min(note.box.top_left[0], note.box.bottom_left[0])
        note_right = max(note.box.top_right[0], note.box.bottom_right[0])
        note_top = min(note.box.top_left[1], note.box.top_right[1])
        note_bottom = max(note.box.bottom_left[1], note.box.bottom_right[1])
        return (
            stem_left <= note_right + padding
            and stem_right >= note_left - padding
            and stem_top <= note_bottom + padding
            and stem_bottom >= note_top - padding
        )

    def _best_visual_sidecar_stem_seed(
        self, note: Note, stem_fragments: list[RotatedBoundingBox]
    ) -> RotatedBoundingBox | None:
        if len(stem_fragments) == 0:
            return note.stem

        candidates = [
            stem
            for stem in stem_fragments
            if self._is_stem_seed_candidate(stem, note)
            and stem.is_overlapping(note.box.make_box_thicker(20))
        ]
        if note.stem is not None and self._is_stem_seed_candidate(note.stem, note):
            candidates.append(note.stem)
        if not candidates:
            return note.stem

        def score(stem: RotatedBoundingBox) -> float:
            note_left = min(note.box.top_left[0], note.box.bottom_left[0])
            note_right = max(note.box.top_right[0], note.box.bottom_right[0])
            note_center_x, note_center_y = note.center
            stem_top = min(stem.top_left[1], stem.top_right[1])
            stem_bottom = max(stem.bottom_left[1], stem.bottom_right[1])
            stem_x = stem.center[0]
            if stem_x >= note_center_x:
                attachment_error = abs(stem_x - note_right)
                extension = note_center_y - stem_top
            else:
                attachment_error = abs(stem_x - note_left)
                extension = stem_bottom - note_center_y
            height_bonus = max(stem_bottom - stem_top, 0) * 0.15
            return float(extension + height_bonus - 2.0 * attachment_error)

        current_score = score(note.stem) if note.stem is not None else float("-inf")
        best = max(candidates, key=score)
        improvement_needed = max(float(note.box.size[1]) * 0.5, 4.0)
        if best is not note.stem and score(best) < current_score + improvement_needed:
            return note.stem
        return best if score(best) > 0 else note.stem

    def _merge_visual_sidecar_stem_fragments(
        self,
        note: Note,
        seed: RotatedBoundingBox,
        stem_fragments: list[RotatedBoundingBox],
    ) -> RotatedBoundingBox:
        if len(stem_fragments) == 0:
            return seed

        notehead_height = max(float(note.box.size[1]), 1.0)
        x_tolerance = max(4.0, float(note.box.size[0]) * 0.6)
        max_vertical_gap = notehead_height * 3
        fragments = [seed]
        changed = True
        while changed:
            changed = False
            for candidate in stem_fragments:
                if candidate in fragments:
                    continue
                if not self._is_stem_like_fragment(candidate, note):
                    continue
                if not self._is_collinear_stem_fragment(
                    candidate, fragments, x_tolerance, max_vertical_gap
                ):
                    continue
                fragments.append(candidate)
                changed = True

        if len(fragments) == 1:
            return seed

        contour = np.concatenate([fragment.polygon.reshape(-1, 1, 2) for fragment in fragments])
        merged = RotatedBoundingBox(cv2.minAreaRect(contour), contour, seed.debug_id)
        if not self._is_stem_like_fragment(merged, note):
            return seed
        return merged

    def _repair_downward_visual_sidecar_stem(
        self,
        note: Note,
        stem: RotatedBoundingBox | None,
        stem_fragments: list[RotatedBoundingBox],
    ) -> RotatedBoundingBox | None:
        return self._repair_visual_sidecar_stem(
            note, stem, StemRepairDirection.DOWN, stem_fragments
        )

    def _repair_upward_visual_sidecar_stem(
        self,
        note: Note,
        stem: RotatedBoundingBox | None,
        stem_fragments: list[RotatedBoundingBox],
    ) -> RotatedBoundingBox | None:
        return self._repair_visual_sidecar_stem(
            note, stem, StemRepairDirection.UP, stem_fragments
        )

    def _repair_visual_sidecar_stem(
        self,
        note: Note,
        stem: RotatedBoundingBox | None,
        direction: StemRepairDirection,
        stem_fragments: list[RotatedBoundingBox],
    ) -> RotatedBoundingBox | None:
        if len(stem_fragments) == 0:
            return stem
        if not self._needs_stem_repair(note, stem, direction):
            return stem

        note_left = min(note.box.top_left[0], note.box.bottom_left[0])
        note_right = max(note.box.top_right[0], note.box.bottom_right[0])
        notehead_height = max(float(note.box.size[1]), 1.0)
        x_tolerance = max(4.0, float(note.box.size[0]) * 0.6)
        max_vertical_gap = notehead_height * 5

        candidates = [
            candidate
            for candidate in stem_fragments
            if self._is_stem_seed_candidate(candidate, note)
            and self._is_stem_repair_seed(
                candidate, note, x_tolerance, max_vertical_gap, direction
            )
        ]
        if not candidates:
            return stem

        def chain_from(seed: RotatedBoundingBox) -> list[RotatedBoundingBox]:
            fragments = [seed]
            changed = True
            while changed:
                changed = False
                for candidate in stem_fragments:
                    if candidate in fragments:
                        continue
                    if not self._is_stem_seed_candidate(candidate, note):
                        continue
                    if not self._is_collinear_stem_fragment(
                        candidate, fragments, x_tolerance, max_vertical_gap
                    ):
                        continue
                    fragments.append(candidate)
                    changed = True
            return fragments

        def score(seed: RotatedBoundingBox) -> float:
            fragments = chain_from(seed)
            points = np.concatenate([fragment.polygon.reshape(-1, 2) for fragment in fragments])
            x_center = float(np.mean(points[:, 0]))
            y_min = float(np.min(points[:, 1]))
            y_max = float(np.max(points[:, 1]))
            attachment_x = note_left if x_center < note.center[0] else note_right
            attachment_error = abs(x_center - attachment_x)
            if direction == StemRepairDirection.UP:
                gap_from_note = max(note.center[1] - y_max, 0.0)
                extension = note.center[1] - y_min
            else:
                gap_from_note = max(y_min - note.center[1], 0.0)
                extension = y_max - note.center[1]
            return extension - attachment_error * 2.0 - gap_from_note

        best_seed = max(candidates, key=score)
        if score(best_seed) <= max(notehead_height, 10.0):
            return stem

        fragments = chain_from(best_seed)
        points = np.concatenate([fragment.polygon.reshape(-1, 2) for fragment in fragments])
        x_min = float(np.min(points[:, 0]))
        x_max = float(np.max(points[:, 0]))
        width = max(x_max - x_min, 1.0)
        x_center = float(np.mean(points[:, 0]))
        half_width = min(max(width / 2, 1.0), max(float(note.box.size[0]) * 0.25, 3.0))
        if direction == StemRepairDirection.UP:
            y_min = float(np.min(points[:, 1]))
            y_max = max(float(np.max(points[:, 1])), float(note.center[1]))
        else:
            y_min = min(float(np.min(points[:, 1])), float(note.center[1]))
            y_max = float(np.max(points[:, 1]))
        contour = np.array(
            [
                [[x_center - half_width, y_min]],
                [[x_center + half_width, y_min]],
                [[x_center + half_width, y_max]],
                [[x_center - half_width, y_max]],
            ],
            dtype=np.float32,
        )
        repaired = RotatedBoundingBox(
            cv2.minAreaRect(contour),
            contour,
            stem.debug_id if stem is not None else best_seed.debug_id,
        )
        if not self._is_stem_like_fragment(repaired, note):
            return stem
        if stem is not None and not self._is_repaired_stem_better(note, stem, repaired):
            return stem
        return repaired

    def _needs_downward_stem_repair(
        self, note: Note, stem: RotatedBoundingBox | None
    ) -> bool:
        return self._needs_stem_repair(note, stem, StemRepairDirection.DOWN)

    def _needs_upward_stem_repair(
        self, note: Note, stem: RotatedBoundingBox | None
    ) -> bool:
        return self._needs_stem_repair(note, stem, StemRepairDirection.UP)

    def _needs_stem_repair(
        self,
        note: Note,
        stem: RotatedBoundingBox | None,
        direction: StemRepairDirection,
    ) -> bool:
        if stem is None:
            return True
        points = np.asarray(stem.polygon, dtype=np.float32).reshape(-1, 2)
        height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        top = float(np.min(points[:, 1]))
        bottom = float(np.max(points[:, 1]))
        notehead_height = max(float(note.box.size[1]), 1.0)
        if direction == StemRepairDirection.UP:
            stem_looks_right_way = stem.center[0] >= note.center[0] or top < note.center[1]
            has_bad_attachment = bottom < note.center[1] - notehead_height * 0.35
        else:
            stem_looks_right_way = stem.center[0] < note.center[0] or bottom > note.center[1]
            has_bad_attachment = top > note.center[1] + notehead_height * 0.35
        return stem_looks_right_way and (
            height < max(1.5 * notehead_height, 18.0) or has_bad_attachment
        )

    def _is_downward_repair_seed(
        self,
        stem: RotatedBoundingBox,
        note: Note,
        x_tolerance: float,
        max_vertical_gap: float,
    ) -> bool:
        return self._is_stem_repair_seed(
            stem, note, x_tolerance, max_vertical_gap, StemRepairDirection.DOWN
        )

    def _is_upward_repair_seed(
        self,
        stem: RotatedBoundingBox,
        note: Note,
        x_tolerance: float,
        max_vertical_gap: float,
    ) -> bool:
        return self._is_stem_repair_seed(
            stem, note, x_tolerance, max_vertical_gap, StemRepairDirection.UP
        )

    def _is_stem_repair_seed(
        self,
        stem: RotatedBoundingBox,
        note: Note,
        x_tolerance: float,
        max_vertical_gap: float,
        direction: StemRepairDirection,
    ) -> bool:
        stem_top = min(stem.top_left[1], stem.top_right[1])
        stem_bottom = max(stem.bottom_left[1], stem.bottom_right[1])
        if direction == StemRepairDirection.UP:
            if stem_top >= note.center[1]:
                return False
            if stem_bottom < note.center[1] - max_vertical_gap:
                return False
        else:
            if stem_bottom <= note.center[1]:
                return False
            if stem_top > note.center[1] + max_vertical_gap:
                return False
        note_left = min(note.box.top_left[0], note.box.bottom_left[0])
        note_right = max(note.box.top_right[0], note.box.bottom_right[0])
        attachment_x = note_left if stem.center[0] < note.center[0] else note_right
        return abs(stem.center[0] - attachment_x) <= x_tolerance

    def _is_repaired_stem_better(
        self, note: Note, current: RotatedBoundingBox, repaired: RotatedBoundingBox
    ) -> bool:
        current_points = np.asarray(current.polygon, dtype=np.float32).reshape(-1, 2)
        repaired_points = np.asarray(repaired.polygon, dtype=np.float32).reshape(-1, 2)
        current_height = float(np.max(current_points[:, 1]) - np.min(current_points[:, 1]))
        repaired_height = float(np.max(repaired_points[:, 1]) - np.min(repaired_points[:, 1]))
        return repaired_height >= current_height + max(float(note.box.size[1]) * 0.5, 4.0)

    def _is_stem_seed_candidate(self, stem: RotatedBoundingBox, note: Note) -> bool:
        points = np.asarray(stem.polygon, dtype=np.float32).reshape(-1, 2)
        width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        notehead_width = max(float(note.box.size[0]), 1.0)
        notehead_height = max(float(note.box.size[1]), 1.0)
        max_width = max(8.0, notehead_width * 0.75)
        return (
            self._is_stem_fragment_near_note(points, note, notehead_height)
            and width <= max_width
            and height >= max(2.0 * max(width, 1.0), notehead_height * 0.45)
        )

    def _is_stem_like_fragment(self, stem: RotatedBoundingBox, note: Note) -> bool:
        points = np.asarray(stem.polygon, dtype=np.float32).reshape(-1, 2)
        width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
        height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
        notehead_width = max(float(note.box.size[0]), 1.0)
        notehead_height = max(float(note.box.size[1]), 1.0)
        max_width = max(8.0, notehead_width * 0.75)
        return (
            self._is_stem_fragment_near_note(points, note, notehead_height)
            and width <= max_width
            and height >= max(2.0 * max(width, 1.0), notehead_height * 0.75)
        )

    @staticmethod
    def _is_stem_fragment_near_note(
        points: Any, note: Note, notehead_height: float
    ) -> bool:
        """Keep stem recovery from crossing into unrelated vertically aligned notation."""
        max_distance = notehead_height * MAX_RECONSTRUCTED_STEM_DISTANCE_IN_NOTEHEADS
        top = float(np.min(points[:, 1]))
        bottom = float(np.max(points[:, 1]))
        return (
            top >= note.center[1] - max_distance
            and bottom <= note.center[1] + max_distance
        )

    def _is_collinear_stem_fragment(
        self,
        candidate: RotatedBoundingBox,
        fragments: list[RotatedBoundingBox],
        x_tolerance: float,
        max_vertical_gap: float,
    ) -> bool:
        if abs(candidate.center[0] - fragments[0].center[0]) > x_tolerance:
            return False
        for fragment in fragments:
            if abs(candidate.center[0] - fragment.center[0]) > x_tolerance:
                continue
            candidate_top = min(candidate.top_left[1], candidate.top_right[1])
            candidate_bottom = max(candidate.bottom_left[1], candidate.bottom_right[1])
            fragment_top = min(fragment.top_left[1], fragment.top_right[1])
            fragment_bottom = max(fragment.bottom_left[1], fragment.bottom_right[1])
            vertical_gap = max(candidate_top - fragment_bottom, fragment_top - candidate_bottom, 0)
            if vertical_gap <= max_vertical_gap:
                return True
        return False

    def _typical_notehead_angles_by_staff(self) -> dict[int, float]:
        angles_by_staff: dict[int, list[float]] = {}
        for group in self.visual_groups.values():
            for ellipse in group.notehead_ellipses:
                if ellipse.get("_fit_source") == "fallback":
                    continue
                if abs(float(ellipse["angle"])) < 6:
                    continue
                if float(ellipse["rx"]) <= float(ellipse["ry"]):
                    continue
                angles_by_staff.setdefault(group.staff_index, []).append(float(ellipse["angle"]))

        all_angles = [angle for angles in angles_by_staff.values() for angle in angles]
        global_angle = float(np.median(all_angles)) if all_angles else None
        result = {}
        for staff_index, angles in angles_by_staff.items():
            result[staff_index] = float(np.median(angles))
        if global_angle is not None:
            for group in self.visual_groups.values():
                result.setdefault(group.staff_index, global_angle)
        return result

    def _notehead_ellipses_for_output(
        self, group: VisualGroup, typical_angle: float | None
    ) -> list[dict[str, Any]]:
        ellipses = []
        for ellipse in group.notehead_ellipses:
            output = {key: value for key, value in ellipse.items() if not key.startswith("_")}
            fit_source = ellipse.get("_fit_source")
            aspect_ratio = float(output["rx"]) / max(float(output["ry"]), 1e-6)
            mask_angle_needs_fallback = (
                fit_source == "mask"
                and abs(float(output["angle"])) < 6
                and aspect_ratio < HORIZONTAL_HOLLOW_NOTEHEAD_ASPECT_RATIO
                and not bool(ellipse.get("_is_hollow", False))
            )
            if typical_angle is not None and (
                fit_source == "fallback"
                or mask_angle_needs_fallback
            ):
                output["angle"] = round(typical_angle, 3)
            ellipses.append(output)
        return ellipses

    def to_json_dict(self) -> dict[str, Any]:
        typical_angles_by_staff = self._typical_notehead_angles_by_staff()
        return {
            "version": 1,
            "source_image_size": list(self.metadata.source_image_size),
            "preprocessing": {
                "autocrop_box": list(self.metadata.autocrop_box),
                "cropped_size": list(self.metadata.cropped_size),
                "resized_size": list(self.metadata.resized_size),
                "resize_scale": [
                    round(self.metadata.resize_scale[0], 8),
                    round(self.metadata.resize_scale[1], 8),
                ],
                "prediction_size": list(self.metadata.prediction_size),
            },
            "notes": [record.__dict__ for record in self.musicxml_notes],
            "raw_stem_contours": self._raw_stem_contours_for_output(),
            "visual_groups": [
                {
                    "visual_group_id": group.visual_id,
                    "staff_index": group.staff_index,
                    "staff_position": group.staff_position,
                    "center": [
                        round(
                            self.metadata.prediction_point_to_source(group.prediction_center)[0], 3
                        ),
                        round(
                            self.metadata.prediction_point_to_source(group.prediction_center)[1], 3
                        ),
                    ],
                    "bbox": group.bbox,
                    "notehead_ellipses": self._notehead_ellipses_for_output(
                        group, typical_angles_by_staff.get(group.staff_index)
                    ),
                    "notehead_contours": group.notehead_contours,
                    "detected_notehead_contours": group.detected_notehead_contours,
                    "refined_notehead_contours": group.refined_notehead_contours,
                    "detected_stem_contours": group.detected_stem_contours,
                    "stem_contours": group.stem_contours,
                    "stem_component_ids": self._stem_component_ids_for_output(group),
                    "is_hollow_notehead": group.is_hollow_notehead,
                    "musicxml_ids": group.linked_musicxml_ids,
                }
                for group in sorted(self.visual_groups.values(), key=lambda g: g.visual_id)
            ],
            "unmatched_musicxml_notes": self.unmatched_musicxml_notes(),
            "unmatched_visual_notes": sorted(self.unmatched_visual_notes),
        }

    def _score_match(self, symbol: EncodedSymbol, visual_group: VisualGroup) -> float:
        score = 0.65
        if visual_group.transformer_center is not None and symbol.coordinates is not None:
            try:
                coords = np.asarray(symbol.coordinates).reshape(-1)
                if len(coords) >= 2:
                    dx = abs(float(coords[0]) - visual_group.transformer_center[0])
                    dy = abs(float(coords[1]) - visual_group.transformer_center[1])
                    score += max(0.0, 0.25 - (dx + dy) / 1000.0)
            except (TypeError, ValueError):
                pass
        return min(round(score, 3), 1.0)


def write_visual_sidecar(path: str, collector: VisualSidecar) -> None:
    visual_sidecar_path = Path(path)
    visual_sidecar_path.write_text(json.dumps(collector.to_json_dict(), indent=2), encoding="utf-8")
