from bisect import bisect_left, bisect_right
from enum import Enum
from typing import Any

import cv2
import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Note
from homr.visual_sidecar.models import StemOwnershipCache

MAX_RECONSTRUCTED_STEM_DISTANCE_IN_NOTEHEADS = 8.0
MAX_STEM_COMPONENT_GAP_IN_NOTEHEADS = 1.5


class StemRepairDirection(Enum):
    UP = "up"
    DOWN = "down"


class StemGeometry:
    def __init__(self, stem_fragments: list[RotatedBoundingBox]) -> None:
        self.stem_fragments = stem_fragments
        self._stem_ownership_cache: StemOwnershipCache | None = None
        self._stem_component_bounds_cache: dict[int, tuple[float, float, float, float]] | None = (
            None
        )

    def build_ownership_cache(self, notes: list[Note]) -> StemOwnershipCache:
        cache = self._build_stem_ownership_cache(notes)
        self._stem_ownership_cache = cache
        self._stem_component_bounds_cache = None
        return cache

    def stem_for_note(self, note: Note, ownership: StemOwnershipCache) -> RotatedBoundingBox | None:
        return self._visual_sidecar_stem_for_note(note, ownership)

    def same_fragment(self, first: RotatedBoundingBox, second: RotatedBoundingBox) -> bool:
        return self._same_stem_fragment(first, second)

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

    def _build_stem_ownership_cache(self, notes: list[Note]) -> StemOwnershipCache:
        if not self.stem_fragments:
            return StemOwnershipCache({}, {})

        widths = [float(note.box.size[0]) for note in notes]
        heights = [float(note.box.size[1]) for note in notes]
        x_tolerance = (
            self._stem_fragment_x_tolerance(float(np.median(widths)), float(np.median(heights)))
            if widths and heights
            else 4.0
        )
        max_vertical_gap = (
            max(
                4.0,
                float(np.median(heights)) * MAX_STEM_COMPONENT_GAP_IN_NOTEHEADS,
            )
            if heights
            else 4.0
        )

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
                if self._is_collinear_stem_fragment(second, [first], x_tolerance, max_vertical_gap):
                    union(first_index, second_index)
                next_position += 1

        component_by_fragment_id = {
            id(stem): find(index) for index, stem in enumerate(self.stem_fragments)
        }
        owner_note_ids_by_component: dict[int, set[int]] = {}
        notes_by_x = sorted(notes, key=lambda note: note.center[0])
        note_xs = [note.center[0] for note in notes_by_x]
        max_note_half_width = max((float(note.box.size[0]) / 2 for note in notes_by_x), default=0.0)
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

    def _same_stem_fragment(self, first: RotatedBoundingBox, second: RotatedBoundingBox) -> bool:
        return first is second or (
            np.allclose(first.center, second.center, atol=1.0)
            and np.allclose(first.size, second.size, atol=1.0)
        )

    def _stem_touches_notehead(self, stem: RotatedBoundingBox, note: Note) -> bool:
        return self._stem_bounds_touch_notehead(self._stem_bounds(stem), note)

    def _stem_bounds(self, stem: RotatedBoundingBox) -> tuple[float, float, float, float]:
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
        x_tolerance = self._stem_fragment_x_tolerance(
            float(note.box.size[0]), float(note.box.size[1])
        )
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
        return self._repair_visual_sidecar_stem(note, stem, StemRepairDirection.UP, stem_fragments)

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
        x_tolerance = self._stem_fragment_x_tolerance(
            float(note.box.size[0]), float(note.box.size[1])
        )
        max_vertical_gap = notehead_height * 5

        candidates = [
            candidate
            for candidate in stem_fragments
            if self._is_stem_seed_candidate(candidate, note)
            and self._is_stem_repair_seed(candidate, note, x_tolerance, max_vertical_gap, direction)
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
        if not self._is_stem_like_fragment(repaired, note) or (
            stem is not None and not self._is_repaired_stem_better(note, stem, repaired)
        ):
            return stem
        return repaired

    def _needs_downward_stem_repair(self, note: Note, stem: RotatedBoundingBox | None) -> bool:
        return self._needs_stem_repair(note, stem, StemRepairDirection.DOWN)

    def _needs_upward_stem_repair(self, note: Note, stem: RotatedBoundingBox | None) -> bool:
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
    def _is_stem_fragment_near_note(points: Any, note: Note, notehead_height: float) -> bool:
        """Keep stem recovery from crossing into unrelated vertically aligned notation."""
        max_distance = notehead_height * MAX_RECONSTRUCTED_STEM_DISTANCE_IN_NOTEHEADS
        top = float(np.min(points[:, 1]))
        bottom = float(np.max(points[:, 1]))
        return top >= note.center[1] - max_distance and bottom <= note.center[1] + max_distance

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

    @staticmethod
    def _stem_fragment_x_tolerance(notehead_width: float, notehead_height: float) -> float:
        """Allow segmentation drift without joining neighboring opposing stems."""
        return max(4.0, min(notehead_width, notehead_height) * 0.4)
