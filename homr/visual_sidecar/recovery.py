from typing import Any

import cv2
import numpy as np

from homr import constants
from homr.model import Note, Staff
from homr.note_detection import split_clumps_of_noteheads
from homr.visual_sidecar.models import SidecarState, StemOwnershipCache
from homr.visual_sidecar.stems import StemGeometry


class RecoveryManager:
    def __init__(
        self,
        state: SidecarState,
        notehead_candidates: list[Any],
        notehead_mask: Any | None,
        stems: StemGeometry,
        source_image: Any | None,
    ) -> None:
        self.state = state
        self.notehead_candidates = notehead_candidates
        self.notehead_mask = notehead_mask
        self.stems = stems
        self.source_image = (
            cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
            if source_image is not None and source_image.ndim == 3
            else source_image
        )

    @property
    def _recovery_notes_by_staff_id(self) -> dict[int, list[Note]]:
        return self.state.recovery_notes_by_staff_id

    @_recovery_notes_by_staff_id.setter
    def _recovery_notes_by_staff_id(self, value: dict[int, list[Note]]) -> None:
        self.state.recovery_notes_by_staff_id = value

    @property
    def _staff_index_by_visual_id(self) -> dict[str, int]:
        return self.state.staff_index_by_visual_id

    @_staff_index_by_visual_id.setter
    def _staff_index_by_visual_id(self, value: dict[str, int]) -> None:
        self.state.staff_index_by_visual_id = value

    @property
    def _staff_position_by_visual_id(self) -> dict[str, int]:
        return self.state.staff_position_by_visual_id

    @_staff_position_by_visual_id.setter
    def _staff_position_by_visual_id(self, value: dict[str, int]) -> None:
        self.state.staff_position_by_visual_id = value

    @property
    def _stem_ownership_cache(self) -> StemOwnershipCache | None:
        return self.state.stem_ownership_cache

    @_stem_ownership_cache.setter
    def _stem_ownership_cache(self, value: StemOwnershipCache | None) -> None:
        self.state.stem_ownership_cache = value

    @property
    def _next_recovered_visual_id(self) -> int:
        return self.state.next_recovered_visual_id

    @_next_recovered_visual_id.setter
    def _next_recovered_visual_id(self, value: int) -> None:
        self.state.next_recovered_visual_id = value

    def prepare(self, staffs: list[Staff]) -> None:
        self.prepare_recovery_notes(staffs)

    def for_staff(self, staff: Staff) -> list[Note]:
        return self.recovery_notes_for_staff(staff)

    def prepare_recovery_notes(self, staffs: list[Staff]) -> None:
        """Assign real, inference-excluded candidates to their nearest staff.

        These notes are used exclusively for sidecar geometry. They are never added to
        ``staff.symbols`` and therefore cannot affect TrOMR inference or MusicXML output.
        """
        self._recovery_notes_by_staff_id = {id(staff): [] for staff in staffs}
        self._staff_index_by_visual_id = {}
        self._staff_position_by_visual_id = {}
        for staff in staffs:
            for note in staff.get_notes():
                if note.visual_id is None:
                    continue
                staff_index = self._staff_index_for_note(staff, note)
                self._staff_index_by_visual_id[note.visual_id] = staff_index
                self._staff_position_by_visual_id[note.visual_id] = self._staff_position_for_center(
                    staff, note.center, staff_index
                )
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
                unit_size = (
                    point.average_unit_size if point is not None else staff.average_unit_size
                )
                if (
                    split_notehead.size[0] < 0.45 * unit_size
                    # Keep the inference filter untouched. Sidecar recovery gets a
                    # small allowance for mask contours that include ledger-line ink.
                    or split_notehead.size[0] > 3.25 * unit_size
                    or split_notehead.size[1] < 0.45 * unit_size
                    or split_notehead.size[1] > 2 * unit_size
                ):
                    continue
                visual_id = f"vnote-recovered-{self._next_recovered_visual_id}"
                self._next_recovered_visual_id += 1
                staff_index = self._staff_index_for_center(staff, split_notehead.center)
                position = self._staff_position_for_center(
                    staff, split_notehead.center, staff_index
                )
                self._staff_index_by_visual_id[visual_id] = staff_index
                self._staff_position_by_visual_id[visual_id] = position
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
            note for recovered in self._recovery_notes_by_staff_id.values() for note in recovered
        )
        self._stem_ownership_cache = self.stems.build_ownership_cache(all_visual_notes)

    def recovery_notes_for_staff(self, staff: Staff) -> list[Note]:
        return self._recovery_notes_by_staff_id.get(id(staff), [])

    @staticmethod
    def _grid_staff_lines_at_x(staff: Staff, x: float, staff_index: int) -> list[float]:
        """Estimate staff lines from a robust local fit instead of one grid sample.

        Staff-line segmentation is commonly interrupted by noteheads, stems, and
        accidentals. A single resampled grid point can consequently jump by half a
        staff space at exactly the x-coordinate where a note must be classified.
        Fitting several neighboring points makes the result invariant to those local
        outliers while retaining genuine staff slope and curvature.
        """
        lines_per_staff = constants.number_of_lines_on_a_staff
        start = staff_index * lines_per_staff
        stop = start + lines_per_staff
        points = sorted(
            (point for point in staff.grid if len(point.y) >= stop),
            key=lambda point: point.x,
        )
        if not points:
            raise ValueError(f"Physical staff {staff_index} has no complete grid points")

        unit_sizes = [
            float(np.median(np.diff(point.y[start:stop])))
            for point in points
            if np.all(np.diff(point.y[start:stop]) > 0)
        ]
        if not unit_sizes:
            raise ValueError("Staff lines must be ordered from top to bottom")
        x_steps = [
            abs(second.x - first.x)
            for first, second in zip(points, points[1:], strict=False)
            if abs(second.x - first.x) > 1e-6
        ]
        unit_size = float(np.median(unit_sizes))
        grid_step = float(np.median(x_steps)) if x_steps else unit_size
        radius = max(4 * unit_size, 4 * grid_step)
        local_points = [point for point in points if abs(point.x - x) <= radius]
        minimum_points = min(7, len(points))
        if len(local_points) < minimum_points:
            local_points = sorted(points, key=lambda point: abs(point.x - x))[:minimum_points]

        def line_at_x(line_index: int) -> float:
            samples = [(point.x, point.y[start + line_index]) for point in local_points]
            slopes = [
                (second_y - first_y) / (second_x - first_x)
                for sample_index, (first_x, first_y) in enumerate(samples)
                for second_x, second_y in samples[sample_index + 1 :]
                if abs(second_x - first_x) > 1e-6
            ]
            if not slopes:
                return float(np.median([sample_y for _sample_x, sample_y in samples]))
            slope = float(np.median(slopes))
            intercept = float(
                np.median([sample_y - slope * sample_x for sample_x, sample_y in samples])
            )
            return slope * x + intercept

        lines = [line_at_x(line_index) for line_index in range(lines_per_staff)]
        if not np.all(np.diff(lines) > 0):
            raise ValueError("Robustly fitted staff lines must be ordered from top to bottom")
        return lines

    def _physical_staff_lines_at_x(self, staff: Staff, x: float, staff_index: int) -> list[float]:
        """Locate the printed staff lines, using the fitted grid as a search guide.

        Resampling can follow a notehead, stem, or broken segmentation fragment for
        several neighboring grid points. That makes a geometric fit alone repeatable
        but not necessarily correct. The source-image pass searches for horizontal
        ink on both sides of the note and accepts it only when the five resulting
        lines retain plausible staff spacing.
        """
        fitted_lines = self._grid_staff_lines_at_x(staff, x, staff_index)
        if self.source_image is None:
            return fitted_lines

        image = self.source_image
        height, width = image.shape[:2]

        unit_size = float(np.median(np.diff(fitted_lines)))
        center_x = int(round(x))
        search_radius = max(2, int(round(0.45 * unit_size)))
        outer_radius = max(12, int(round(6 * unit_size)))
        inner_radius = max(4, int(round(1.5 * unit_size)))
        left_x = np.arange(max(0, center_x - outer_radius), max(0, center_x - inner_radius))
        right_x = np.arange(
            min(width, center_x + inner_radius + 1),
            min(width, center_x + outer_radius + 1),
        )
        sample_x = np.concatenate((left_x, right_x))
        if min(len(left_x), len(right_x)) < max(4, int(round(unit_size))):
            return fitted_lines

        refined_lines: list[float] = []
        for fitted_y in fitted_lines:
            candidates: list[tuple[float, int, float, float, int]] = []
            first_y = max(0, int(round(fitted_y)) - search_radius)
            last_y = min(height - 1, int(round(fitted_y)) + search_radius)
            for candidate_y in range(first_y, last_y + 1):
                pixels = image[candidate_y, sample_x].astype(np.float32)
                left_dark_count = int(np.count_nonzero(image[candidate_y, left_x] < 160))
                right_dark_count = int(np.count_nonzero(image[candidate_y, right_x] < 160))
                bilateral_support = min(
                    left_dark_count / len(left_x), right_dark_count / len(right_x)
                )
                dark_count = left_dark_count + right_dark_count
                darkness = float(np.sum(255 - pixels))
                fitted_distance = -abs(candidate_y - fitted_y)
                candidates.append(
                    (bilateral_support, dark_count, darkness, fitted_distance, candidate_y)
                )
            best_support, _best_count, _best_darkness, _best_distance, best_y = max(candidates)
            if best_support < 0.35:
                return fitted_lines
            refined_lines.append(float(best_y))

        spacings = np.diff(refined_lines)
        if (
            not np.all(spacings > 0)
            or np.any(spacings < 0.6 * unit_size)
            or np.any(spacings > 1.4 * unit_size)
        ):
            return fitted_lines
        return refined_lines

    def _staff_index_for_center(self, staff: Staff, center: tuple[float, float]) -> int:
        lines_per_staff = constants.number_of_lines_on_a_staff
        staff_count = max(len(point.y) for point in staff.grid) // lines_per_staff
        return min(
            range(staff_count),
            key=lambda index: min(
                abs(line_y - center[1])
                for line_y in self._physical_staff_lines_at_x(staff, center[0], index)
            ),
        )

    def _staff_index_for_note(self, staff: Staff, note: Note) -> int:
        """Recover the physical staff that originally supplied a grand-staff note.

        Ledger zones between two staves overlap, so the same segmentation candidate
        can be admitted by both source staves. Its ``position`` was calculated on
        the source staff before the staffs were merged and therefore preserves
        ownership even when the notehead center is closer to the other staff.
        """
        lines_per_staff = constants.number_of_lines_on_a_staff
        staff_count = max(len(point.y) for point in staff.grid) // lines_per_staff
        if staff_count <= 1:
            return 0

        def position_error(staff_index: int) -> tuple[int, float]:
            lines = self._physical_staff_lines_at_x(staff, note.center[0], staff_index)
            expected_position = self._staff_position_from_lines(lines, note.center[1])
            geometric_distance = min(abs(line_y - note.center[1]) for line_y in lines)
            return (abs(expected_position - note.position), geometric_distance)

        return min(range(staff_count), key=position_error)

    @staticmethod
    def _staff_position_from_lines(lines: list[float], center_y: float) -> int:
        closest_line_index = int(np.argmin(np.abs(np.subtract(lines, center_y))))
        unit_size = float(np.mean(np.diff(lines)))
        if unit_size <= 0:
            raise ValueError("Staff lines must be ordered from top to bottom")
        distance = lines[closest_line_index] - center_y
        return 2 * (len(lines) - closest_line_index) + round(2 * distance / unit_size) - 1

    def _staff_position_for_center(
        self, staff: Staff, center: tuple[float, float], staff_index: int
    ) -> int:
        """Return a line/space position local to one physical five-line staff."""
        lines = self._physical_staff_lines_at_x(staff, center[0], staff_index)
        return self._staff_position_from_lines(lines, center[1])
