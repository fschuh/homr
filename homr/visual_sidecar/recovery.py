from typing import Any

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
    ) -> None:
        self.state = state
        self.notehead_candidates = notehead_candidates
        self.notehead_mask = notehead_mask
        self.stems = stems

    @property
    def _recovery_notes_by_staff_id(self) -> dict[int, list[Note]]:
        return self.state.recovery_notes_by_staff_id

    @_recovery_notes_by_staff_id.setter
    def _recovery_notes_by_staff_id(self, value: dict[int, list[Note]]) -> None:
        self.state.recovery_notes_by_staff_id = value

    @property
    def _stave_index_by_visual_id(self) -> dict[str, int]:
        return self.state.stave_index_by_visual_id

    @_stave_index_by_visual_id.setter
    def _stave_index_by_visual_id(self, value: dict[str, int]) -> None:
        self.state.stave_index_by_visual_id = value

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
        self._stave_index_by_visual_id = {
            note.visual_id: self._stave_index_for_note(staff, note)
            for staff in staffs
            for note in staff.get_notes()
            if note.visual_id is not None
        }
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
                if point is not None:
                    position = point.find_position_in_unit_sizes(split_notehead)
                else:
                    nearest_point = min(
                        staff.grid, key=lambda item: abs(item.x - split_notehead.center[0])
                    )
                    position = nearest_point.find_position_in_unit_sizes(split_notehead)
                visual_id = f"vnote-recovered-{self._next_recovered_visual_id}"
                self._next_recovered_visual_id += 1
                self._stave_index_by_visual_id[visual_id] = self._stave_index_for_center(
                    staff, split_notehead.center
                )
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
    def _stave_index_for_center(staff: Staff, center: tuple[float, float]) -> int:
        point = staff.get_at(center[0])
        if point is None:
            point = min(staff.grid, key=lambda candidate: abs(candidate.x - center[0]))
        lines_per_stave = constants.number_of_lines_on_a_staff
        line_groups = [
            point.y[index : index + lines_per_stave]
            for index in range(0, len(point.y), lines_per_stave)
        ]
        return min(
            range(len(line_groups)),
            key=lambda index: min(abs(line_y - center[1]) for line_y in line_groups[index]),
        )

    @classmethod
    def _stave_index_for_note(cls, staff: Staff, note: Note) -> int:
        """Recover the stave that originally supplied a grand-staff note.

        Ledger zones between two staves overlap, so the same segmentation candidate
        can be admitted by both source staves. Its ``position`` was calculated on
        the source stave before the staves were merged and therefore preserves
        ownership even when the notehead center is closer to the other stave.
        """
        point = staff.get_at(note.center[0])
        if point is None:
            point = min(staff.grid, key=lambda candidate: abs(candidate.x - note.center[0]))
        lines_per_stave = constants.number_of_lines_on_a_staff
        line_groups = [
            point.y[index : index + lines_per_stave]
            for index in range(0, len(point.y), lines_per_stave)
        ]
        if len(line_groups) <= 1:
            return 0

        def position_error(stave_index: int) -> tuple[int, float]:
            lines = line_groups[stave_index]
            closest_line_index = int(np.argmin(np.abs(np.subtract(lines, note.center[1]))))
            unit_size = float(np.mean(np.diff(lines)))
            if unit_size <= 0:
                return (abs(note.position), float("inf"))
            distance = lines[closest_line_index] - note.center[1]
            expected_position = (
                2 * (len(lines) - closest_line_index) + round(2 * distance / unit_size) - 1
            )
            geometric_distance = min(abs(line_y - note.center[1]) for line_y in lines)
            return (abs(expected_position - note.position), geometric_distance)

        return min(range(len(line_groups)), key=position_error)
