from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from homr.model import Staff
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar.matching_utils import token_moments
from homr.visual_sidecar.models import SidecarState, VisualGroup
from homr.visual_sidecar.noteheads import NoteheadGeometry, NoteheadPixelFit

PhysicalStaffLines = Callable[[Staff, float, int], list[float]]

MAX_POSITION_ERROR = 2
MIN_FIT_CONFIDENCE = 0.46
MIN_CORE_SUPPORT = 0.16
MIN_BOUNDARY_SUPPORT = 0.15
WEAK_CURRENT_FIT = 0.72
MIN_ASSIGNMENT_MARGIN = 0.075
MAX_JOINT_MEMBERS = 6
MAX_DUPLICATE_CENTER_DISTANCE_RATIO = 0.8


@dataclass(frozen=True)
class _Member:
    symbol: EncodedSymbol
    group: VisualGroup
    expected_position: int
    staff_lines: list[float]
    unit_size: float
    expected_y: float
    current_core: frozenset[int]
    current_confidence: float

    @property
    def needs_position_repair(self) -> bool:
        return self.group.staff_position != self.expected_position


class NoteheadRefitter:
    """Conservatively refit matched noteheads from independent pixel evidence.

    Recognition supplies an expected staff slot and therefore bounds the search;
    it never supplies the result.  Every accepted center, ellipse, and final staff
    position comes from source/mask pixels and the physical staff grid.
    """

    def __init__(self, state: SidecarState, noteheads: NoteheadGeometry) -> None:
        self.state = state
        self.noteheads = noteheads
        self.visual_groups = state.visual_groups
        self.matches_by_symbol_id = state.matches_by_symbol_id

    def refit(
        self,
        symbols: list[EncodedSymbol],
        staff_group_index: int,
        *,
        source_staff: Staff | None,
        expected_staff_positions: dict[int, int],
        physical_staff_lines: PhysicalStaffLines,
    ) -> None:
        if source_staff is None or self.noteheads.source_image is None:
            return

        all_members = self._matched_members(
            symbols,
            staff_group_index,
            source_staff,
            expected_staff_positions,
            physical_staff_lines,
        )
        if not all_members:
            return
        members_by_symbol_id = {member.symbol.visual_match_id: member for member in all_members}
        matched_visual_ids = {member.group.visual_id for member in all_members}
        handled_target_ids: set[str] = set()

        for moment in token_moments(symbols):
            moment_members = [
                members_by_symbol_id[symbol.visual_match_id]
                for symbol in moment
                if symbol.visual_match_id in members_by_symbol_id
            ]
            for staff_index in sorted({member.group.staff_index for member in moment_members}):
                physical_members = [
                    member for member in moment_members if member.group.staff_index == staff_index
                ]
                targets = [
                    member
                    for member in physical_members
                    if member.group.visual_id not in handled_target_ids and self._eligible(member)
                ]
                if not targets:
                    continue

                # A split component is a single visual fact even if sequence repair
                # placed one of its members outside the token moment.  Pull every
                # matched member of that component into the same atomic fit.
                clump_ids = {
                    target.group.split_clump_id
                    for target in targets
                    if target.group.split_clump_id is not None
                }
                subset_by_visual_id = {
                    member.group.visual_id: member for member in physical_members
                }
                for member in all_members:
                    if (
                        member.group.staff_index == staff_index
                        and member.group.split_clump_id in clump_ids
                    ):
                        subset_by_visual_id[member.group.visual_id] = member
                subset = sorted(
                    subset_by_visual_id.values(),
                    key=lambda member: (
                        -member.expected_position,
                        member.group.prediction_center[0],
                        member.group.visual_id,
                    ),
                )
                target_ids = {target.group.visual_id for target in targets}
                handled_target_ids.update(target_ids)
                if not subset or len(subset) > MAX_JOINT_MEMBERS:
                    continue

                owned_pixels = self._owned_neighbor_pixels(
                    source_staff,
                    physical_staff_lines,
                    matched_visual_ids,
                    {member.group.visual_id for member in subset},
                )
                assignment = self._joint_assignment(
                    subset,
                    owned_pixels,
                    source_staff,
                    physical_staff_lines,
                )
                if assignment is None or not self._meaningfully_improves(
                    subset, assignment, target_ids
                ):
                    continue
                self._apply_atomically(
                    subset,
                    assignment,
                    target_ids,
                    source_staff,
                    physical_staff_lines,
                )

    def _matched_members(
        self,
        symbols: list[EncodedSymbol],
        staff_group_index: int,
        source_staff: Staff,
        expected_staff_positions: dict[int, int],
        physical_staff_lines: PhysicalStaffLines,
    ) -> list[_Member]:
        result: list[_Member] = []
        for symbol in symbols:
            if not symbol.rhythm.startswith("note"):
                continue
            expected_position = expected_staff_positions.get(symbol.visual_match_id)
            match = self.matches_by_symbol_id.get(symbol.visual_match_id)
            if expected_position is None or match is None or match.visual_id is None:
                continue
            group = self.visual_groups.get(match.visual_id)
            if group is None or group.staff_group_index != staff_group_index:
                continue
            try:
                lines = physical_staff_lines(
                    source_staff,
                    group.prediction_center[0],
                    group.staff_index,
                )
            except (IndexError, ValueError):
                continue
            unit_size = self._reliable_staff_unit(lines)
            if unit_size is None:
                continue
            current = self.noteheads.score_prediction_geometry(
                group.prediction_center,
                group.prediction_notehead_size,
                unit_size,
                lines,
            )
            current_core, current_confidence = (
                current if current is not None else (frozenset(), 0.0)
            )
            result.append(
                _Member(
                    symbol=symbol,
                    group=group,
                    expected_position=expected_position,
                    staff_lines=lines,
                    unit_size=unit_size,
                    expected_y=self._staff_position_y(lines, expected_position),
                    current_core=current_core,
                    current_confidence=current_confidence,
                )
            )
        return result

    @staticmethod
    def _reliable_staff_unit(lines: list[float]) -> float | None:
        if len(lines) != 5:
            return None
        spacings = np.asarray(np.diff(lines), dtype=float)
        if len(spacings) != 4 or np.any(spacings <= 0):
            return None
        unit_size = float(np.median(spacings))
        if unit_size < 4 or float(np.std(spacings)) > 0.25 * unit_size:
            return None
        return unit_size

    @staticmethod
    def _staff_position_y(lines: list[float], staff_position: int) -> float:
        unit_size = float(np.median(np.diff(lines)))
        return float(lines[-1] - (staff_position - 1) * unit_size / 2)

    @staticmethod
    def _staff_position_from_lines(lines: list[float], center_y: float) -> int:
        closest_line_index = int(np.argmin(np.abs(np.subtract(lines, center_y))))
        unit_size = float(np.median(np.diff(lines)))
        distance = lines[closest_line_index] - center_y
        return 2 * (len(lines) - closest_line_index) + round(2 * distance / unit_size) - 1

    @staticmethod
    def _eligible(member: _Member) -> bool:
        position_error = abs(member.group.staff_position - member.expected_position)
        if position_error == 0 or position_error > MAX_POSITION_ERROR:
            return False
        return (
            member.group.split_clump_id is not None or member.current_confidence < WEAK_CURRENT_FIT
        )

    def _owned_neighbor_pixels(
        self,
        source_staff: Staff,
        physical_staff_lines: PhysicalStaffLines,
        matched_visual_ids: set[str],
        subset_visual_ids: set[str],
    ) -> frozenset[int]:
        owned: set[int] = set()
        for visual_id in sorted(matched_visual_ids - subset_visual_ids):
            group = self.visual_groups.get(visual_id)
            if group is None:
                continue
            if group.notehead_core_pixels:
                owned.update(group.notehead_core_pixels)
                continue
            try:
                lines = physical_staff_lines(
                    source_staff,
                    group.prediction_center[0],
                    group.staff_index,
                )
            except (IndexError, ValueError):
                continue
            unit_size = self._reliable_staff_unit(lines)
            if unit_size is None:
                continue
            scored = self.noteheads.score_prediction_geometry(
                group.prediction_center,
                group.prediction_notehead_size,
                unit_size,
                lines,
            )
            if scored is None:
                continue
            group.notehead_core_pixels = scored[0]
            owned.update(scored[0])
        return frozenset(owned)

    def _joint_assignment(
        self,
        members: list[_Member],
        owned_pixels: frozenset[int],
        source_staff: Staff,
        physical_staff_lines: PhysicalStaffLines,
    ) -> list[NoteheadPixelFit] | None:
        hypotheses = self._hypotheses(members, owned_pixels)
        if len(hypotheses) < len(members):
            return None

        options: list[list[tuple[int, float]]] = []
        for member in members:
            member_options: list[tuple[int, float]] = []
            for hypothesis_index, hypothesis in enumerate(hypotheses):
                try:
                    lines = physical_staff_lines(
                        source_staff,
                        hypothesis.prediction_center[0],
                        member.group.staff_index,
                    )
                except (IndexError, ValueError):
                    continue
                if self._reliable_staff_unit(lines) is None:
                    continue
                fitted_position = self._staff_position_from_lines(
                    lines, hypothesis.prediction_center[1]
                )
                if fitted_position != member.expected_position:
                    continue
                y_distance = abs(hypothesis.prediction_center[1] - member.expected_y)
                if y_distance > 0.62 * member.unit_size:
                    continue
                x_distance = abs(
                    hypothesis.prediction_center[0] - member.group.prediction_center[0]
                )
                if not self._within_horizontal_window(member, hypothesis, x_distance):
                    continue
                cost = (
                    1.0
                    - hypothesis.confidence
                    + 0.16 * y_distance / member.unit_size
                    + 0.035 * x_distance / member.unit_size
                )
                member_options.append((hypothesis_index, cost))
            member_options.sort(key=lambda option: (option[1], option[0]))
            if not member_options:
                return None
            options.append(member_options)

        assignments: list[tuple[float, tuple[int, ...]]] = []

        def visit(
            member_index: int,
            selected: list[int],
            used_hypotheses: set[int],
            used_core: set[int],
            total_cost: float,
        ) -> None:
            if member_index == len(members):
                assignments.append((total_cost, tuple(selected)))
                return
            for hypothesis_index, cost in options[member_index]:
                hypothesis = hypotheses[hypothesis_index]
                if hypothesis_index in used_hypotheses or not used_core.isdisjoint(
                    hypothesis.core_pixels
                ):
                    continue
                visit(
                    member_index + 1,
                    [*selected, hypothesis_index],
                    {*used_hypotheses, hypothesis_index},
                    used_core.union(hypothesis.core_pixels),
                    total_cost + cost,
                )

        visit(0, [], set(), set(), 0.0)
        if not assignments:
            return None
        assignments.sort(key=lambda assignment: (assignment[0], assignment[1]))
        best_cost, best_indices = assignments[0]
        if len(assignments) > 1:
            second_cost = assignments[1][0]
            if second_cost - best_cost < MIN_ASSIGNMENT_MARGIN * len(members):
                return None
        return [hypotheses[index] for index in best_indices]

    @staticmethod
    def _within_horizontal_window(
        member: _Member,
        hypothesis: NoteheadPixelFit,
        x_distance: float,
    ) -> bool:
        bounds = member.group.split_clump_bounds
        if bounds is not None:
            margin = 0.35 * member.unit_size
            return bounds[0] - margin <= hypothesis.prediction_center[0] <= bounds[2] + margin
        return x_distance <= 1.2 * member.unit_size

    def _hypotheses(
        self,
        members: list[_Member],
        owned_pixels: frozenset[int],
    ) -> list[NoteheadPixelFit]:
        hypotheses: list[NoteheadPixelFit] = []
        shared_x = float(np.median([member.group.prediction_center[0] for member in members]))
        for member in members:
            x_values = {
                round(member.group.prediction_center[0], 3),
                round(shared_x, 3),
                round(member.group.prediction_center[0] - 0.35 * member.unit_size, 3),
                round(member.group.prediction_center[0] + 0.35 * member.unit_size, 3),
            }
            bounds = member.group.split_clump_bounds
            if bounds is not None:
                x_values.update(
                    {
                        round((bounds[0] + bounds[2]) / 2, 3),
                        round(bounds[0] + 0.3 * (bounds[2] - bounds[0]), 3),
                        round(bounds[0] + 0.7 * (bounds[2] - bounds[0]), 3),
                    }
                )
            for x_value in sorted(x_values):
                fit = self.noteheads.fit_notehead_hypothesis(
                    (float(x_value), member.expected_y),
                    member.unit_size,
                    member.staff_lines,
                    member.group.visual_id,
                )
                if (
                    fit is None
                    or fit.confidence < MIN_FIT_CONFIDENCE
                    or fit.core_support < MIN_CORE_SUPPORT
                    or fit.boundary_support < MIN_BOUNDARY_SUPPORT
                    or not owned_pixels.isdisjoint(fit.core_pixels)
                ):
                    continue
                self._insert_unique_hypothesis(
                    hypotheses,
                    fit,
                    member.unit_size,
                    member.staff_lines,
                )
        hypotheses.sort(
            key=lambda fit: (
                round(fit.prediction_center[1], 5),
                round(fit.prediction_center[0], 5),
                -fit.confidence,
            )
        )
        return hypotheses

    @staticmethod
    def _insert_unique_hypothesis(
        hypotheses: list[NoteheadPixelFit],
        candidate: NoteheadPixelFit,
        unit_size: float,
        staff_lines: list[float],
    ) -> None:
        for index, existing in enumerate(hypotheses):
            center_distance = float(
                np.linalg.norm(np.subtract(existing.prediction_center, candidate.prediction_center))
            )
            union = existing.core_pixels | candidate.core_pixels
            overlap = (
                len(existing.core_pixels & candidate.core_pixels) / len(union) if union else 0.0
            )
            existing_position = NoteheadRefitter._staff_position_from_lines(
                staff_lines, existing.prediction_center[1]
            )
            candidate_position = NoteheadRefitter._staff_position_from_lines(
                staff_lines, candidate.prediction_center[1]
            )
            same_staff_slot = existing_position == candidate_position
            if same_staff_slot and (
                center_distance <= MAX_DUPLICATE_CENTER_DISTANCE_RATIO * unit_size
                or overlap >= 0.55
            ):
                if (
                    candidate.confidence,
                    candidate.core_support,
                    candidate.boundary_support,
                ) > (
                    existing.confidence,
                    existing.core_support,
                    existing.boundary_support,
                ):
                    hypotheses[index] = candidate
                return
        hypotheses.append(candidate)

    @staticmethod
    def _meaningfully_improves(
        members: list[_Member],
        assignment: list[NoteheadPixelFit],
        target_ids: set[str],
    ) -> bool:
        for member, fitted in zip(members, assignment, strict=True):
            if member.group.visual_id not in target_ids:
                if fitted.confidence + 0.08 < member.current_confidence:
                    return False
                continue
            improvement = fitted.confidence - member.current_confidence
            shift = (
                float(
                    np.linalg.norm(
                        np.subtract(fitted.prediction_center, member.group.prediction_center)
                    )
                )
                / member.unit_size
            )
            minimum_improvement = 0.025 if member.group.split_clump_id is not None else 0.08
            if improvement < minimum_improvement:
                if not (
                    member.group.split_clump_id is not None
                    and shift >= 0.18
                    and fitted.confidence >= 0.68
                ):
                    return False
        return True

    @staticmethod
    def _apply_atomically(
        members: list[_Member],
        assignment: list[NoteheadPixelFit],
        target_ids: set[str],
        source_staff: Staff,
        physical_staff_lines: PhysicalStaffLines,
    ) -> None:
        applications: list[tuple[_Member, NoteheadPixelFit, int]] = []
        for member, fitted in zip(members, assignment, strict=True):
            lines = physical_staff_lines(
                source_staff,
                fitted.prediction_center[0],
                member.group.staff_index,
            )
            fitted_position = NoteheadRefitter._staff_position_from_lines(
                lines, fitted.prediction_center[1]
            )
            if fitted_position != member.expected_position:
                return
            applications.append((member, fitted, fitted_position))

        for member, fitted, fitted_position in applications:
            group = member.group
            group.notehead_core_pixels = fitted.core_pixels
            if group.visual_id not in target_ids:
                continue
            group.staff_position = fitted_position
            group.prediction_center = fitted.prediction_center
            group.prediction_notehead_size = fitted.prediction_size
            group.notehead_ellipses = [dict(fitted.source_ellipse)]
            group.notehead_contours = [fitted.source_contour]
            group.refined_notehead_contours = [fitted.source_contour]
            group.is_hollow_notehead = fitted.is_hollow
            group.visual_status = "fallback"
            for action in ("joint_notehead_refit", "pixel_staff_position_repaired"):
                if action not in group.repair_actions:
                    group.repair_actions.append(action)
