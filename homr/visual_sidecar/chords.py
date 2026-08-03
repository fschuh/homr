from typing import Any

import cv2
import numpy as np

from homr import constants
from homr.bounding_boxes import BoundingEllipse
from homr.model import Note, Staff
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform
from homr.visual_sidecar.matching_utils import (
    diatonic_pitch_index,
    local_staff_unit,
    noteheads_can_share_chord_stem,
    token_moments,
)
from homr.visual_sidecar.models import SidecarState, StemOwnershipCache, VisualGroup
from homr.visual_sidecar.noteheads import NoteheadGeometry
from homr.visual_sidecar.stems import StemGeometry

VISUAL_MOMENT_NOTEHEAD_WIDTH_RATIO = 0.65


class ChordResolver:
    def __init__(
        self,
        state: SidecarState,
        coordinate_transform: PredictionCoordinateTransform,
        source_image: Any | None,
        noteheads: NoteheadGeometry,
        stems: StemGeometry,
    ) -> None:
        self.state = state
        self.coordinate_transform = coordinate_transform
        self.source_image = source_image
        self.noteheads = noteheads
        self.stems = stems
        self.visual_groups = state.visual_groups
        self.matches_by_symbol_id = state.matches_by_symbol_id
        self.stem_fragments = stems.stem_fragments
        self._stem_component_bounds_cache: dict[int, tuple[float, float, float, float]] | None = (
            None
        )

    @property
    def _stem_ownership_cache(self) -> StemOwnershipCache | None:
        return self.state.stem_ownership_cache

    @property
    def _next_transformer_recovered_visual_id(self) -> int:
        return self.state.next_transformer_recovered_visual_id

    @_next_transformer_recovered_visual_id.setter
    def _next_transformer_recovered_visual_id(self, value: int) -> None:
        self.state.next_transformer_recovered_visual_id = value

    def recover_transformer_notehead(
        self,
        symbol: EncodedSymbol,
        staff_index: int,
        *,
        source_staff: Staff | None,
        neighboring_groups: list[VisualGroup],
        chord_mates: list[tuple[EncodedSymbol, VisualGroup]],
        available_groups: list[VisualGroup],
    ) -> VisualGroup | None:
        return self._recover_transformer_chord_notehead(
            symbol,
            staff_index,
            source_staff,
            neighboring_groups,
            chord_mates,
            available_groups=available_groups,
        )

    def assign_physical_chord_ids(self, symbols: list[EncodedSymbol], staff_index: int) -> None:
        self._assign_physical_chord_ids(symbols, staff_index)

    def stem_component_ids_for_output(self, group: VisualGroup) -> list[str]:
        return self._stem_component_ids_for_output(group)

    def _recover_transformer_chord_notehead(
        self,
        symbol: EncodedSymbol,
        staff_index: int,
        source_staff: Staff | None,
        neighboring_groups: list[VisualGroup],
        chord_mates: list[tuple[EncodedSymbol, VisualGroup]],
        *,
        available_groups: list[VisualGroup],
    ) -> VisualGroup | None:
        """Recover a chord head that segmentation missed but TrOMR recognized.

        A matched same-stave chord member supplies an exact source anchor. Apply the
        recognized diatonic interval using the local five-line stave spacing, then
        require the source-image contour fitter to find supporting ink. Chord context
        and pixel evidence together keep hallucinated MusicXML notes unmatched.
        """
        if (
            self.source_image is None
            or source_staff is None
            or not chord_mates
            or not symbol.rhythm.startswith("note")
            or symbol.coordinates is None
            or not bool(np.all(np.isfinite(symbol.coordinates)))
        ):
            return None
        prediction_center: tuple[float, float] | None = None
        recovered_staff_position: int | None = None
        recovered_stave_index: int | None = None
        symbol_pitch_index = diatonic_pitch_index(symbol.pitch)
        for mate_symbol, mate_group in chord_mates:
            mate_pitch_index = diatonic_pitch_index(mate_symbol.pitch)
            if (
                mate_symbol.position != symbol.position
                or symbol_pitch_index is None
                or mate_pitch_index is None
            ):
                continue
            pitch_steps = symbol_pitch_index - mate_pitch_index
            mate_source_point = source_staff.get_at(mate_group.prediction_center[0])
            if mate_source_point is None:
                mate_source_point = min(
                    source_staff.grid,
                    key=lambda point: abs(point.x - mate_group.prediction_center[0]),
                )
            mate_line_index = int(
                np.argmin(
                    np.abs(np.asarray(mate_source_point.y) - float(mate_group.prediction_center[1]))
                )
            )
            local_unit = local_staff_unit(mate_source_point, mate_line_index)
            prediction_center = (
                float(mate_group.prediction_center[0]),
                float(mate_group.prediction_center[1]) - pitch_steps * local_unit / 2,
            )
            recovered_staff_position = mate_group.staff_position + pitch_steps
            recovered_stave_index = mate_group.stave_index
            break
        if (
            prediction_center is None
            or recovered_staff_position is None
            or recovered_stave_index is None
        ):
            return None
        source_point = source_staff.get_at(prediction_center[0])
        if source_point is None:
            source_point = min(
                source_staff.grid, key=lambda point: abs(point.x - prediction_center[0])
            )
        source_line_index = int(
            np.argmin(np.abs(np.asarray(source_point.y) - float(prediction_center[1])))
        )
        unit_size = max(local_staff_unit(source_point, source_line_index), 4.0)
        width = constants.NOTEHEAD_SIZE_RATIO * unit_size
        height = unit_size
        center = (float(prediction_center[0]), float(prediction_center[1]))
        existing_candidates: list[VisualGroup] = []
        for candidate in available_groups:
            if (
                candidate.staff_index != staff_index
                or candidate.stave_index != recovered_stave_index
                or abs(candidate.staff_position - recovered_staff_position) > 2
            ):
                continue
            horizontal_distance = abs(candidate.prediction_center[0] - center[0])
            vertical_distance = abs(candidate.prediction_center[1] - center[1])
            candidate_width = max(candidate.prediction_notehead_size[0], 1.0)
            if (
                horizontal_distance > max(width, candidate_width) * 1.5
                or vertical_distance > unit_size * 0.85
            ):
                continue
            existing_candidates.append(candidate)
        if len(existing_candidates) == 1:
            return existing_candidates[0]

        axes = (max(2, int(round(width / 2))), max(2, int(round(height / 2))))
        contour = cv2.ellipse2Poly(
            (int(round(center[0])), int(round(center[1]))),
            axes,
            -20,
            0,
            360,
            5,
        ).reshape(-1, 1, 2)
        box = BoundingEllipse((center, (width, height), -20), contour)
        visual_id = f"vnote-transformer-recovered-{self._next_transformer_recovered_visual_id}"
        guessed_note = Note(
            box,
            recovered_staff_position,
            None,
            None,
            visual_id,
        )
        neighboring_group_notes = [
            (
                group,
                Note(
                    BoundingEllipse(
                        (
                            group.prediction_center,
                            (width, height),
                            -20,
                        ),
                        contour,
                    ),
                    group.staff_position,
                    None,
                    None,
                    group.visual_id,
                ),
            )
            for group in neighboring_groups
            if group.staff_index == staff_index
        ]
        neighboring_notes = [note for _group, note in neighboring_group_notes]
        refined_contour = self.noteheads.refined_notehead_contour(
            guessed_note, [guessed_note, *neighboring_notes]
        )
        relaxed_dense_chord_fit = False
        if refined_contour is None:
            chord_mate_visual_ids = {
                mate_group.visual_id for _mate_symbol, mate_group in chord_mates
            }
            non_mate_neighbors = [
                note
                for group, note in neighboring_group_notes
                if group.visual_id not in chord_mate_visual_ids
            ]
            refined_contour = self.noteheads.refined_notehead_contour(
                guessed_note, [guessed_note, *non_mate_neighbors]
            )
            relaxed_dense_chord_fit = refined_contour is not None
        if refined_contour is None:
            return None
        self._next_transformer_recovered_visual_id += 1
        notehead_ellipse = self.noteheads.ellipse_from_source_contour(refined_contour)
        notehead_ellipse["_is_hollow"] = self.noteheads.is_hollow_notehead(guessed_note)
        return VisualGroup(
            visual_id=visual_id,
            staff_index=staff_index,
            stave_index=recovered_stave_index,
            staff_position=guessed_note.position,
            prediction_center=center,
            prediction_notehead_size=(float(width), float(height)),
            transformer_center=(float(symbol.coordinates[0]), float(symbol.coordinates[1])),
            transformer_notehead_size=None,
            notehead_ellipses=[notehead_ellipse],
            notehead_contours=[refined_contour],
            detected_notehead_contours=[],
            refined_notehead_contours=[refined_contour],
            detected_stem_contours=[],
            stem_contours=[],
            owned_stem_component_ids=[],
            is_hollow_notehead=self.noteheads.is_hollow_notehead(guessed_note),
            visual_status="fallback",
            provenance="transformer_recovered",
            repair_actions=[
                "transformer_notehead_recovered",
                *(["dense_chord_notehead_recovered"] if relaxed_dense_chord_fit else []),
            ],
            duration=symbol.rhythm,
        )

    def _stem_component_ids_for_output(self, group: VisualGroup) -> list[str]:
        if group.duration is None:
            return []
        duration_class = group.duration.rstrip(".")
        result = []
        for component_id in group.owned_stem_component_ids:
            if any(
                candidate.visual_id != group.visual_id
                and candidate.duration is not None
                and candidate.duration.rstrip(".") == duration_class
                and component_id in candidate.owned_stem_component_ids
                and noteheads_can_share_chord_stem(group, candidate)
                for candidate in self.visual_groups.values()
            ):
                result.append(f"{component_id}-duration-{duration_class}")
        return result

    def _stem_component_bounds(self) -> dict[int, tuple[float, float, float, float]]:
        if self._stem_component_bounds_cache is not None:
            return self._stem_component_bounds_cache
        stem_ownership = (
            self._stem_ownership_cache
            if self._stem_ownership_cache is not None
            else self.stems.build_ownership_cache([])
        )

        bounds_by_component: dict[int, tuple[float, float, float, float]] = {}
        for stem in self.stem_fragments:
            component = stem_ownership.component_by_fragment_id.get(id(stem))
            if component is None:
                continue
            left, right, top, bottom = self.stems._stem_bounds(stem)
            previous = bounds_by_component.get(component)
            if previous is not None:
                left = min(left, previous[0])
                right = max(right, previous[1])
                top = min(top, previous[2])
                bottom = max(bottom, previous[3])
            bounds_by_component[component] = (left, right, top, bottom)
        self._stem_component_bounds_cache = bounds_by_component
        return bounds_by_component

    def _have_opposed_independent_stems(self, first: VisualGroup, second: VisualGroup) -> bool:
        """Detect close simultaneous voices whose stems leave opposite sides.

        Adjacent heads can make the lower, downward stem touch both segmentation
        candidates. That shared ownership is not chord proof when a separate upward
        stem also leaves the upper head on the other side.
        """
        upper, lower = sorted((first, second), key=lambda group: group.prediction_center[1])
        center_x = float(np.median([upper.prediction_center[0], lower.prediction_center[0]]))
        notehead_width = max(
            min(
                float(upper.prediction_notehead_size[0]),
                float(lower.prediction_notehead_size[0]),
            ),
            1.0,
        )
        notehead_height = max(
            min(
                float(upper.prediction_notehead_size[1]),
                float(lower.prediction_notehead_size[1]),
            ),
            1.0,
        )
        side_offset = max(1.0, notehead_width * 0.18)
        horizontal_limit = max(6.0, notehead_width * 0.9)
        minimum_extension = notehead_height * 0.55

        upward_components: set[int] = set()
        downward_components: set[int] = set()
        for component, (left, right, top, bottom) in self._stem_component_bounds().items():
            component_x = (left + right) / 2
            if abs(component_x - center_x) > horizontal_limit:
                continue
            if (
                component_x >= center_x + side_offset
                and top <= upper.prediction_center[1] - minimum_extension
                and upper.prediction_center[1] - notehead_height * 0.9
                <= bottom
                <= upper.prediction_center[1] + notehead_height * 0.25
            ):
                upward_components.add(component)
            if (
                component_x <= center_x - side_offset
                and bottom >= lower.prediction_center[1] + minimum_extension
                and lower.prediction_center[1] - notehead_height * 0.25
                <= top
                <= lower.prediction_center[1] + notehead_height * 0.9
            ):
                downward_components.add(component)
        return any(
            upward != downward for upward in upward_components for downward in downward_components
        )

    def _assign_physical_chord_ids(self, symbols: list[EncodedSymbol], staff_index: int) -> None:
        """Assign chord identity only when same-stave geometry proves it."""
        chord_index = 1
        for token_moment in token_moments(symbols):
            matched_by_stave: dict[int, list[tuple[EncodedSymbol, VisualGroup]]] = {}
            for symbol in token_moment:
                match = self.matches_by_symbol_id.get(symbol.visual_match_id)
                if match is None or match.visual_id is None:
                    continue
                group = self.visual_groups.get(match.visual_id)
                if (
                    group is None
                    or group.staff_index != staff_index
                    or group.visual_status == "diagnostic"
                ):
                    continue
                matched_by_stave.setdefault(group.stave_index, []).append((symbol, group))
            for stave_index, members in sorted(matched_by_stave.items()):
                if len(members) < 2:
                    continue
                members_by_duration: dict[str, list[tuple[EncodedSymbol, VisualGroup]]] = {}
                for symbol, group in members:
                    members_by_duration.setdefault(symbol.rhythm.rstrip("."), []).append(
                        (symbol, group)
                    )
                if len(members_by_duration) > 1:
                    for _symbol, group in members:
                        if "mixed_duration_stems_separated" not in group.repair_actions:
                            group.repair_actions.append("mixed_duration_stems_separated")

                for duration_members in members_by_duration.values():
                    if len(duration_members) < 2:
                        continue
                    chord_assigned = self._assign_physical_chord_id_to_members(
                        duration_members,
                        staff_index,
                        stave_index,
                        chord_index,
                    )
                    if chord_assigned:
                        chord_index += 1

    def _assign_physical_chord_id_to_members(
        self,
        members: list[tuple[EncodedSymbol, VisualGroup]],
        staff_index: int,
        stave_index: int,
        chord_index: int,
    ) -> bool:
        groups = [group for _symbol, group in members]
        opposed_stems = any(
            self._have_opposed_independent_stems(first, second)
            for group_index, first in enumerate(groups)
            for second in groups[group_index + 1 :]
        )
        if opposed_stems:
            for group in groups:
                if "opposed_stems_separated" not in group.repair_actions:
                    group.repair_actions.append("opposed_stems_separated")
            return False

        common_stem_components = set(groups[0].owned_stem_component_ids)
        for group in groups[1:]:
            common_stem_components.intersection_update(group.owned_stem_component_ids)
        stem_proven = bool(common_stem_components) and all(
            noteheads_can_share_chord_stem(first, second)
            for first, second in zip(groups, groups[1:], strict=False)
        )
        compact_chord_geometry = all(
            noteheads_can_share_chord_stem(first, second)
            for group_index, first in enumerate(groups)
            for second in groups[group_index + 1 :]
        )
        structural_chord_proven = (
            compact_chord_geometry
            and len({group.moment_id for group in groups}) == 1
            and groups[0].moment_id is not None
            and all(
                self.matches_by_symbol_id[symbol.visual_match_id].alignment_method == "structural"
                for symbol, _group in members
            )
        )
        widths = [max(group.prediction_notehead_size[0], 1.0) for group in groups]
        whole_note_proven = (
            all(group.is_hollow_notehead for group in groups)
            and all(
                group.duration is not None and group.duration.rstrip(".") == "note_1"
                for group in groups
            )
            and bool(widths)
            and max(group.prediction_center[0] for group in groups)
            - min(group.prediction_center[0] for group in groups)
            <= float(np.median(widths)) * VISUAL_MOMENT_NOTEHEAD_WIDTH_RATIO
        )
        transformer_chord_recovered = (
            any(group.provenance == "transformer_recovered" for group in groups)
            and compact_chord_geometry
        )
        if (
            not stem_proven
            and not structural_chord_proven
            and not whole_note_proven
            and not transformer_chord_recovered
        ):
            return False

        moment_id = next(
            (group.moment_id for group in groups if group.moment_id),
            f"moment-{staff_index + 1}-repair-{chord_index}",
        )
        chord_id = f"chord-{staff_index + 1}-{moment_id}-" f"{stave_index + 1}-{chord_index}"
        for symbol, group in members:
            group.chord_id = chord_id
            group.moment_id = moment_id
            if stem_proven and "shared_stem_proven" not in group.repair_actions:
                group.repair_actions.append("shared_stem_proven")
            if structural_chord_proven and "structural_chord_proven" not in group.repair_actions:
                group.repair_actions.append("structural_chord_proven")
            if (
                transformer_chord_recovered
                and "transformer_chord_recovered" not in group.repair_actions
            ):
                group.repair_actions.append("transformer_chord_recovered")
            match = self.matches_by_symbol_id[symbol.visual_match_id]
            if match.alignment_method == "attention" and group.provenance not in (
                "recovered_candidate",
                "transformer_recovered",
            ):
                match.alignment_method = "stem_repair"
                group.visual_status = "canonical"
        return True
