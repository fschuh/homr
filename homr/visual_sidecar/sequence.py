import itertools

import numpy as np

from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords
from homr.visual_sidecar.matching_utils import (
    diatonic_pitch_index,
    noteheads_can_share_chord_stem,
    symbol_group_distance,
)
from homr.visual_sidecar.models import VisualGroup

ATTENTION_MATCH_NOTEHEAD_WIDTH_RATIO = 1.5
ATTENTION_UNIQUENESS_NOTEHEAD_WIDTH_RATIO = 0.25
VISUAL_MOMENT_X_TOLERANCE = 6.0


class SequenceMatcher:
    def assign(
        self,
        symbols: list[EncodedSymbol],
        visual_groups: list[VisualGroup],
        locked_assignments: list[tuple[int, int]],
        reserved_group_indices: set[int],
    ) -> list[tuple[int, int]]:
        return self._assign_around_locked_matches(
            symbols, visual_groups, locked_assignments, reserved_group_indices
        )

    def release_split_moment_outliers(
        self,
        symbols: list[EncodedSymbol],
        note_symbols: list[EncodedSymbol],
        visual_groups: list[VisualGroup],
        assignments: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        return self._release_split_moment_outliers(
            symbols, note_symbols, visual_groups, assignments
        )

    @classmethod
    def _assign_around_locked_matches(
        cls,
        note_symbols: list[EncodedSymbol],
        visual_groups: list[VisualGroup],
        locked_assignments: list[tuple[int, int]],
        reserved_group_indices: set[int] | None = None,
    ) -> list[tuple[int, int]]:
        """Add only unique, local, order-compatible attention assignments."""
        assignments = list(locked_assignments)
        assigned_symbols = {symbol_index for symbol_index, _ in assignments}
        assigned_groups = {group_index for _, group_index in assignments}
        assigned_groups.update(reserved_group_indices or set())
        has_multiple_staves = any(group.stave_index == 1 for group in visual_groups)
        widths = [
            max(
                (group.transformer_notehead_size or group.prediction_notehead_size)[0],
                1.0,
            )
            for group in visual_groups
        ]
        typical_width = float(np.median(widths)) if widths else 10.0
        max_distance = max(6.0, typical_width * ATTENTION_MATCH_NOTEHEAD_WIDTH_RATIO)
        uniqueness_margin = max(1.0, typical_width * ATTENTION_UNIQUENESS_NOTEHEAD_WIDTH_RATIO)
        distances: dict[tuple[int, int], float] = {}
        for symbol_index, symbol in enumerate(note_symbols):
            if symbol_index in assigned_symbols or symbol.coordinates is None:
                continue
            if not bool(np.all(np.isfinite(symbol.coordinates))):
                continue
            for group_index, group in enumerate(visual_groups):
                if group_index in assigned_groups or group.transformer_center is None:
                    continue
                expected_stave_index = 1 if symbol.position == "lower" else 0
                if has_multiple_staves and group.stave_index != expected_stave_index:
                    continue
                if not bool(np.all(np.isfinite(group.transformer_center))):
                    continue
                distances[(symbol_index, group_index)] = float(
                    np.linalg.norm(np.subtract(symbol.coordinates, group.transformer_center))
                )

        def unique_nearest(values: list[tuple[float, int]]) -> tuple[float, int] | None:
            if not values:
                return None
            ordered = sorted(values)
            if ordered[0][0] > max_distance:
                return None
            if len(ordered) > 1 and ordered[1][0] - ordered[0][0] < uniqueness_margin:
                return None
            return ordered[0]

        nearest_group_by_symbol: dict[int, int] = {}
        for symbol_index in range(len(note_symbols)):
            nearest = unique_nearest(
                [
                    (distance, group_index)
                    for (candidate_symbol, group_index), distance in distances.items()
                    if candidate_symbol == symbol_index
                ]
            )
            if nearest is not None:
                nearest_group_by_symbol[symbol_index] = nearest[1]
        nearest_symbol_by_group: dict[int, int] = {}
        for group_index in range(len(visual_groups)):
            nearest = unique_nearest(
                [
                    (distance, symbol_index)
                    for (symbol_index, candidate_group), distance in distances.items()
                    if candidate_group == group_index
                ]
            )
            if nearest is not None:
                nearest_symbol_by_group[group_index] = nearest[1]

        candidates = sorted(
            (
                distances[(symbol_index, group_index)],
                symbol_index,
                group_index,
            )
            for symbol_index, group_index in nearest_group_by_symbol.items()
            if nearest_symbol_by_group.get(group_index) == symbol_index
        )
        for _, symbol_index, group_index in candidates:
            if symbol_index in assigned_symbols or group_index in assigned_groups:
                continue
            expected_stave_index = 1 if note_symbols[symbol_index].position == "lower" else 0
            group_x = visual_groups[group_index].prediction_center[0]
            crosses = False
            for assigned_symbol_index, assigned_group_index in assignments:
                assigned_stave_index = (
                    1 if note_symbols[assigned_symbol_index].position == "lower" else 0
                )
                if assigned_stave_index != expected_stave_index:
                    continue
                assigned_x = visual_groups[assigned_group_index].prediction_center[0]
                if (symbol_index - assigned_symbol_index) * (group_x - assigned_x) < 0:
                    group = visual_groups[group_index]
                    assigned_group = visual_groups[assigned_group_index]
                    shares_physical_stem = bool(
                        set(group.owned_stem_component_ids).intersection(
                            assigned_group.owned_stem_component_ids
                        )
                    ) and noteheads_can_share_chord_stem(group, assigned_group)
                    if not shares_physical_stem:
                        crosses = True
                        break
            if crosses:
                continue
            assignments.append((symbol_index, group_index))
            assigned_symbols.add(symbol_index)
            assigned_groups.add(group_index)

        # When attention is absent, a single remaining symbol and a single
        # compatible candidate form a unique sequence repair. Do not generalize
        # this into a cursor fill: repeated or surplus candidates stay unmatched.
        remaining_symbols = [
            index for index in range(len(note_symbols)) if index not in assigned_symbols
        ]
        remaining_groups = [
            index for index in range(len(visual_groups)) if index not in assigned_groups
        ]
        if len(remaining_symbols) == 1 and len(remaining_groups) == 1:
            symbol_index = remaining_symbols[0]
            group_index = remaining_groups[0]
            expected_stave_index = 1 if note_symbols[symbol_index].position == "lower" else 0
            if (
                not has_multiple_staves
                or visual_groups[group_index].stave_index == expected_stave_index
            ):
                assignments.append((symbol_index, group_index))
        return sorted(assignments)

    def _repair_chord_assignments(
        self,
        symbols: list[EncodedSymbol],
        note_symbols: list[EncodedSymbol],
        visual_groups: list[VisualGroup],
        assignments: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Keep recognized chord members on their shared physical stem.

        Transformer attention can exchange a lower chord member with an adjacent
        single note even when the top member is positioned correctly. Shared stem
        components provide stronger chord identity than those individual attention
        coordinates. Reassign the recognized chord by pitch height and give any
        displaced neighboring symbols the visual groups that the chord vacated.
        """
        group_by_symbol_index = dict(assignments)
        symbol_index_by_match_id = {
            symbol.visual_match_id: index for index, symbol in enumerate(note_symbols)
        }

        for chord in sort_token_chords(symbols):
            chord_symbol_indices = [
                symbol_index_by_match_id[symbol.visual_match_id]
                for symbol in chord
                if (
                    symbol.rhythm.startswith("note")
                    and symbol.visual_match_id in symbol_index_by_match_id
                )
            ]
            symbols_by_stave: dict[str, list[int]] = {}
            for symbol_index in chord_symbol_indices:
                position = note_symbols[symbol_index].position
                symbols_by_stave.setdefault(position, []).append(symbol_index)

            for stave_symbol_indices in symbols_by_stave.values():
                if len(stave_symbol_indices) < 2 or any(
                    index not in group_by_symbol_index for index in stave_symbol_indices
                ):
                    continue
                current_group_indices = [
                    group_by_symbol_index[index] for index in stave_symbol_indices
                ]
                component_candidates: list[list[int]] = []
                for current_group_index in current_group_indices:
                    current_group = visual_groups[current_group_index]
                    for component_id in current_group.owned_stem_component_ids:
                        candidates = [
                            group_index
                            for group_index, group in enumerate(visual_groups)
                            if (
                                group.stave_index == current_group.stave_index
                                and component_id in group.owned_stem_component_ids
                            )
                        ]
                        if len(candidates) == len(stave_symbol_indices):
                            component_candidates.append(candidates)
                if not component_candidates:
                    continue
                desired_group_indices = max(
                    component_candidates,
                    key=lambda candidates: len(set(candidates) & set(current_group_indices)),
                )

                def pitch_height(symbol_index: int) -> int:
                    pitch_index = diatonic_pitch_index(note_symbols[symbol_index].pitch)
                    return pitch_index if pitch_index is not None else -1

                symbol_order = sorted(
                    stave_symbol_indices,
                    key=pitch_height,
                    reverse=True,
                )
                group_order = sorted(
                    desired_group_indices,
                    key=lambda index: visual_groups[index].prediction_center[1],
                )
                desired_by_symbol = dict(zip(symbol_order, group_order, strict=True))
                if all(
                    group_by_symbol_index[symbol_index] == desired_group_index
                    for symbol_index, desired_group_index in desired_by_symbol.items()
                ):
                    continue

                symbol_by_group_index = {
                    group_index: symbol_index
                    for symbol_index, group_index in group_by_symbol_index.items()
                }
                entering_group_indices = [
                    index for index in desired_group_indices if index not in current_group_indices
                ]
                leaving_group_indices = [
                    index for index in current_group_indices if index not in desired_group_indices
                ]
                displaced_symbol_indices = [
                    symbol_by_group_index[index]
                    for index in entering_group_indices
                    if index in symbol_by_group_index
                ]
                if len(displaced_symbol_indices) != len(leaving_group_indices):
                    continue

                group_by_symbol_index.update(desired_by_symbol)
                if displaced_symbol_indices:
                    replacement_order = min(
                        itertools.permutations(leaving_group_indices),
                        key=lambda candidate_order: sum(
                            symbol_group_distance(
                                note_symbols[symbol_index], visual_groups[group_index]
                            )
                            for symbol_index, group_index in zip(
                                displaced_symbol_indices, candidate_order, strict=True
                            )
                        ),
                    )
                    group_by_symbol_index.update(
                        zip(displaced_symbol_indices, replacement_order, strict=True)
                    )

        return sorted(group_by_symbol_index.items())

    def _repair_adjacent_sequence_inversions(
        self,
        symbols: list[EncodedSymbol],
        note_symbols: list[EncodedSymbol],
        visual_groups: list[VisualGroup],
        assignments: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Undo crossed attention matches between adjacent same-stave notes.

        Attention occasionally exchanges two neighboring notes in a scalar run.
        Musical order then points right-to-left while the assigned staff positions
        also contradict the pitches. Requiring both inversions makes the swap safe
        around chords, repeated pitches, and deliberately displaced noteheads.
        """
        group_by_symbol_index = dict(assignments)
        symbol_index_by_match_id = {
            symbol.visual_match_id: index for index, symbol in enumerate(note_symbols)
        }
        moments = [
            [
                symbol_index_by_match_id[symbol.visual_match_id]
                for symbol in chord
                if symbol.visual_match_id in symbol_index_by_match_id
            ]
            for chord in sort_token_chords(symbols)
        ]
        moments = [moment for moment in moments if moment]

        for _ in range(len(moments)):
            changed = False
            for first_moment, second_moment in zip(moments, moments[1:], strict=False):
                positions = {note_symbols[index].position for index in first_moment + second_moment}
                for position in positions:
                    first_indices = [
                        index for index in first_moment if note_symbols[index].position == position
                    ]
                    second_indices = [
                        index for index in second_moment if note_symbols[index].position == position
                    ]
                    if len(first_indices) != 1 or len(second_indices) != 1:
                        continue
                    first_symbol_index = first_indices[0]
                    second_symbol_index = second_indices[0]
                    if (
                        first_symbol_index not in group_by_symbol_index
                        or second_symbol_index not in group_by_symbol_index
                    ):
                        continue
                    first_symbol = note_symbols[first_symbol_index]
                    second_symbol = note_symbols[second_symbol_index]
                    if first_symbol.rhythm.rstrip(".") != second_symbol.rhythm.rstrip("."):
                        continue
                    first_pitch = diatonic_pitch_index(first_symbol.pitch)
                    second_pitch = diatonic_pitch_index(second_symbol.pitch)
                    if first_pitch is None or second_pitch is None or first_pitch == second_pitch:
                        continue
                    first_group_index = group_by_symbol_index[first_symbol_index]
                    second_group_index = group_by_symbol_index[second_symbol_index]
                    first_group = visual_groups[first_group_index]
                    second_group = visual_groups[second_group_index]
                    if first_group.stave_index != second_group.stave_index:
                        continue
                    if (
                        first_group.prediction_center[0]
                        <= second_group.prediction_center[0] + VISUAL_MOMENT_X_TOLERANCE
                    ):
                        continue
                    pitch_difference = first_pitch - second_pitch
                    position_difference = first_group.staff_position - second_group.staff_position
                    if pitch_difference * position_difference >= 0:
                        continue
                    group_by_symbol_index[first_symbol_index] = second_group_index
                    group_by_symbol_index[second_symbol_index] = first_group_index
                    changed = True
            if not changed:
                break

        return sorted(group_by_symbol_index.items())

    def _release_split_moment_outliers(
        self,
        symbols: list[EncodedSymbol],
        note_symbols: list[EncodedSymbol],
        visual_groups: list[VisualGroup],
        assignments: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Unassign chord members pulled outside their neighboring moments.

        When one chord head is missing from segmentation, greedy attention can
        attach that member to an unrelated earlier notehead. If another member is
        correctly anchored between the previous and next musical moments, release
        only the outlier. The normal pixel-backed chord recovery can then recreate
        it at the anchor x-position.
        """
        group_by_symbol_index = dict(assignments)
        symbol_index_by_match_id = {
            symbol.visual_match_id: index for index, symbol in enumerate(note_symbols)
        }
        moments = [
            [
                symbol_index_by_match_id[symbol.visual_match_id]
                for symbol in chord
                if symbol.visual_match_id in symbol_index_by_match_id
            ]
            for chord in sort_token_chords(symbols)
        ]
        moments = [moment for moment in moments if moment]

        def assigned_center(moment: list[int]) -> float | None:
            centers = [
                visual_groups[group_by_symbol_index[index]].prediction_center[0]
                for index in moment
                if index in group_by_symbol_index
            ]
            return float(np.median(centers)) if centers else None

        moment_centers = [assigned_center(moment) for moment in moments]
        for moment_index, moment in enumerate(moments):
            assigned_indices = [index for index in moment if index in group_by_symbol_index]
            if len(assigned_indices) < 2:
                continue
            previous_center = next(
                (
                    moment_centers[index]
                    for index in range(moment_index - 1, -1, -1)
                    if moment_centers[index] is not None
                ),
                None,
            )
            next_center = next(
                (
                    moment_centers[index]
                    for index in range(moment_index + 1, len(moments))
                    if moment_centers[index] is not None
                ),
                None,
            )
            in_order: list[int] = []
            outliers: list[int] = []
            for symbol_index in assigned_indices:
                center_x = visual_groups[group_by_symbol_index[symbol_index]].prediction_center[0]
                follows_previous = (
                    previous_center is None
                    or center_x >= previous_center - VISUAL_MOMENT_X_TOLERANCE
                )
                precedes_next = (
                    next_center is None or center_x <= next_center + VISUAL_MOMENT_X_TOLERANCE
                )
                if follows_previous and precedes_next:
                    in_order.append(symbol_index)
                else:
                    outliers.append(symbol_index)
            if in_order:
                for symbol_index in outliers:
                    del group_by_symbol_index[symbol_index]

        return sorted(group_by_symbol_index.items())
