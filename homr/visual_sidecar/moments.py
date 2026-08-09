import itertools

import numpy as np

from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar.matching_utils import (
    diatonic_pitch_index,
    noteheads_can_share_chord_stem,
    token_moments,
)
from homr.visual_sidecar.models import (
    SidecarState,
    StructuralMatchPlan,
    StructuralMomentCompatibility,
    VisualGroup,
)

VISUAL_MOMENT_X_TOLERANCE = 6.0
VISUAL_MOMENT_NOTEHEAD_WIDTH_RATIO = 0.65
ATTENTION_MATCH_NOTEHEAD_WIDTH_RATIO = 1.5
ATTENTION_UNIQUENESS_NOTEHEAD_WIDTH_RATIO = 0.25
DISPLACED_CHORD_MAX_VERTICAL_NOTEHEAD_RATIO = 1.25
CROSS_STAFF_SLOT_NOTEHEAD_WIDTH_RATIO = 1.5
CLEF_REFERENCE_PITCHES = {"G": "G4", "F": "F3", "C": "C4"}


class MomentMatcher:
    def __init__(self, state: SidecarState) -> None:
        self.state = state
        self._duplicate_staff_positions_by_visual_id = state.duplicate_staff_positions_by_visual_id
        self._moment_id_by_symbol_id = state.moment_id_by_symbol_id

    def structural_assignments(
        self,
        symbols: list[EncodedSymbol],
        note_symbols: list[EncodedSymbol],
        visual_groups: list[VisualGroup],
    ) -> StructuralMatchPlan | None:
        return self._structural_moment_assignments(symbols, note_symbols, visual_groups)

    @staticmethod
    def token_moments(symbols: list[EncodedSymbol]) -> list[list[EncodedSymbol]]:
        return token_moments(symbols)

    def cross_staff_assignments(
        self,
        symbols: list[EncodedSymbol],
        pending_symbols: list[EncodedSymbol],
        available_groups: list[VisualGroup],
        assigned_groups: list[VisualGroup],
    ) -> list[tuple[EncodedSymbol, VisualGroup]]:
        """Link unique other-staff candidates in one missing structural slot.

        This repairs a transformer staff-branch error, not a missing note. Pixel
        geometry must already provide the notehead, and its local staff position
        must encode the recognized step and octave under that physical staff's
        active clef. Accidentals deliberately do not participate because they are
        not independently associated with noteheads in sidecar v3.
        """
        if not pending_symbols or not available_groups:
            return []

        moment_ids = self._ordered_note_moment_ids(symbols)
        moment_index_by_id = {
            moment_id: moment_index for moment_index, moment_id in enumerate(moment_ids)
        }
        groups_by_moment: dict[str, list[VisualGroup]] = {}
        for group in assigned_groups:
            if group.moment_id is not None:
                groups_by_moment.setdefault(group.moment_id, []).append(group)
        clefs_by_symbol_id = self._active_clefs_by_symbol_id(symbols)

        slot_cache: dict[str, list[VisualGroup]] = {}
        possible_groups_by_symbol_id: dict[int, list[VisualGroup]] = {}
        for symbol in pending_symbols:
            if not symbol.rhythm.startswith("note"):
                continue
            moment_id = self._moment_id_by_symbol_id.get(symbol.visual_match_id)
            if moment_id is None or moment_id not in moment_index_by_id:
                continue
            if moment_id not in slot_cache:
                slot_cache[moment_id] = self._cross_staff_slot_groups(
                    moment_id,
                    moment_ids,
                    moment_index_by_id,
                    available_groups,
                    groups_by_moment,
                )
            slot_groups = slot_cache[moment_id]
            if not slot_groups:
                continue

            expected_staff_index = 1 if symbol.position == "lower" else 0
            clefs = clefs_by_symbol_id.get(symbol.visual_match_id, {})
            same_staff_matches = [
                group
                for group in slot_groups
                if group.staff_index == expected_staff_index
                and self._group_encodes_symbol_pitch(group, symbol, clefs)
            ]
            if same_staff_matches:
                # Ordinary staff ownership remains preferable even if another
                # candidate happens to encode the same diatonic pitch.
                continue
            possible_groups_by_symbol_id[symbol.visual_match_id] = [
                group
                for group in slot_groups
                if group.staff_index != expected_staff_index
                and self._group_encodes_symbol_pitch(group, symbol, clefs)
            ]

        claiming_symbols_by_visual_id: dict[str, list[int]] = {}
        for symbol_id, groups in possible_groups_by_symbol_id.items():
            for group in groups:
                claiming_symbols_by_visual_id.setdefault(group.visual_id, []).append(symbol_id)

        result = []
        for symbol in pending_symbols:
            groups = possible_groups_by_symbol_id.get(symbol.visual_match_id, [])
            if len(groups) != 1:
                continue
            group = groups[0]
            claimants = claiming_symbols_by_visual_id.get(group.visual_id, [])
            if len(claimants) != 1 or claimants[0] != symbol.visual_match_id:
                continue
            result.append((symbol, group))
        return result

    def _ordered_note_moment_ids(self, symbols: list[EncodedSymbol]) -> list[str]:
        result = []
        for moment in token_moments(symbols):
            occupied = [symbol for symbol in moment if self._symbol_occupies_visual_moment(symbol)]
            if not occupied:
                continue
            moment_id = self._moment_id_by_symbol_id.get(occupied[0].visual_match_id)
            if moment_id is not None:
                result.append(moment_id)
        return result

    @staticmethod
    def _active_clefs_by_symbol_id(
        symbols: list[EncodedSymbol],
    ) -> dict[int, dict[int, tuple[str, int]]]:
        active_clefs: dict[int, tuple[str, int]] = {}
        result: dict[int, dict[int, tuple[str, int]]] = {}
        for symbol in symbols:
            if symbol.rhythm.startswith("clef_"):
                definition = symbol.rhythm.removeprefix("clef_")
                if len(definition) >= 2 and definition[0] in CLEF_REFERENCE_PITCHES:
                    try:
                        line = int(definition[1:])
                    except ValueError:
                        pass
                    else:
                        if 1 <= line <= 5:
                            staff_index = 1 if symbol.position == "lower" else 0
                            active_clefs[staff_index] = (definition[0], line)
            if symbol.rhythm.startswith(("note", "rest")):
                result[symbol.visual_match_id] = dict(active_clefs)
        return result

    @classmethod
    def _cross_staff_slot_groups(
        cls,
        moment_id: str,
        moment_ids: list[str],
        moment_index_by_id: dict[str, int],
        available_groups: list[VisualGroup],
        groups_by_moment: dict[str, list[VisualGroup]],
    ) -> list[VisualGroup]:
        target_groups = groups_by_moment.get(moment_id, [])
        if target_groups:
            target_x = float(np.median([group.prediction_center[0] for group in target_groups]))
            widths = [
                max(group.prediction_notehead_size[0], 1.0)
                for group in [*target_groups, *available_groups]
            ]
            tolerance = max(
                VISUAL_MOMENT_X_TOLERANCE,
                float(np.median(widths)) * CROSS_STAFF_SLOT_NOTEHEAD_WIDTH_RATIO,
            )
            candidates = [
                group
                for group in available_groups
                if abs(group.prediction_center[0] - target_x) <= tolerance
            ]
            return [
                candidate
                for candidate in candidates
                if len(cls._construct_visual_moments([*target_groups, candidate])) == 1
            ]

        return cls._groups_in_missing_cross_staff_slot(
            moment_id,
            moment_ids,
            moment_index_by_id,
            available_groups,
            groups_by_moment,
        )

    @classmethod
    def _groups_in_missing_cross_staff_slot(
        cls,
        moment_id: str,
        moment_ids: list[str],
        moment_index_by_id: dict[str, int],
        available_groups: list[VisualGroup],
        groups_by_moment: dict[str, list[VisualGroup]],
    ) -> list[VisualGroup]:
        target_index = moment_index_by_id[moment_id]
        previous_indices = [
            index
            for index in range(target_index - 1, -1, -1)
            if groups_by_moment.get(moment_ids[index])
        ]
        following_indices = [
            index
            for index in range(target_index + 1, len(moment_ids))
            if groups_by_moment.get(moment_ids[index])
        ]
        if not previous_indices or not following_indices:
            return []
        previous_index = previous_indices[0]
        following_index = following_indices[0]
        if previous_index != target_index - 1 or following_index != target_index + 1:
            return []
        previous_x = float(
            np.median(
                [
                    group.prediction_center[0]
                    for group in groups_by_moment[moment_ids[previous_index]]
                ]
            )
        )
        following_x = float(
            np.median(
                [
                    group.prediction_center[0]
                    for group in groups_by_moment[moment_ids[following_index]]
                ]
            )
        )
        if previous_x >= following_x:
            return []
        candidates = [
            group
            for group in available_groups
            if previous_x < group.prediction_center[0] < following_x
        ]
        if not candidates:
            return []
        visual_moments = cls._construct_visual_moments(candidates)
        if len(visual_moments) != 1:
            return []
        return [candidates[group_index] for group_index in visual_moments[0]]

    @staticmethod
    def _group_encodes_symbol_pitch(
        group: VisualGroup,
        symbol: EncodedSymbol,
        clefs: dict[int, tuple[str, int]],
    ) -> bool:
        clef = clefs.get(group.staff_index)
        pitch_index = diatonic_pitch_index(symbol.pitch)
        if clef is None or pitch_index is None:
            return False
        sign, line = clef
        reference_pitch_index = diatonic_pitch_index(CLEF_REFERENCE_PITCHES[sign])
        if reference_pitch_index is None:
            return False
        expected_position = 2 * line - 1 + pitch_index - reference_pitch_index
        return group.staff_position == expected_position

    def _structural_moment_assignments(
        self,
        symbols: list[EncodedSymbol],
        note_symbols: list[EncodedSymbol],
        visual_groups: list[VisualGroup],
    ) -> StructuralMatchPlan | None:
        """Align compatible recognition and visual moments in page order.

        Token chords preserve left-to-right musical moments, while visual x clusters
        preserve their page order. A weighted sequence alignment locks every complete
        per-staff moment it can prove, while skipping isolated missing or surplus
        moments. This is more reliable than individual transformer attention for
        repeated pitches and prevents one detection defect from shifting the rest of
        a system.

        TrOMR can also emit a real note rhythm whose pitch branch is ``_`` or ``.``.
        Such a symbol is deliberately absent from MusicXML, but it still occupies a
        visual moment. Keep it in the structural sequence and reserve its notehead so
        the following MusicXML note cannot claim that geometry.
        """
        symbol_index_by_match_id = {
            symbol.visual_match_id: index for index, symbol in enumerate(note_symbols)
        }
        symbol_moments = [
            [symbol for symbol in chord if self._symbol_occupies_visual_moment(symbol)]
            for chord in token_moments(symbols)
        ]
        symbol_moments = [moment for moment in symbol_moments if moment]
        if not visual_groups:
            return None
        visual_moments = self._construct_visual_moments(visual_groups)

        assignments: list[tuple[int, int]] = []
        reserved_group_indices: set[int] = set()
        fallback_assignments: set[tuple[int, int]] = set()
        for moment_index, symbol_moment in enumerate(symbol_moments):
            moment_id = f"moment-{visual_groups[0].staff_group_index + 1}-{moment_index + 1}"
            for symbol in symbol_moment:
                self._moment_id_by_symbol_id[symbol.visual_match_id] = moment_id
        for symbol_moment_index, visual_moment_index in self._align_structural_moments(
            symbol_moments, visual_moments, visual_groups
        ):
            symbol_moment = symbol_moments[symbol_moment_index]
            visual_moment = visual_moments[visual_moment_index]
            self._repair_aligned_visual_moment_staffs(symbol_moment, visual_moment, visual_groups)
            compatibility = self._compatible_visual_moment(
                symbol_moment, visual_moment, visual_groups
            )
            if compatibility is None:
                continue
            self._apply_visual_moment_staffs(compatibility.staff_by_group_index, visual_groups)
            moment_id = f"moment-{visual_groups[0].staff_group_index + 1}-{symbol_moment_index + 1}"
            symbols_by_staff: dict[int, list[EncodedSymbol]] = {}
            for symbol in symbol_moment:
                position = symbol.position
                staff_index = 1 if position == "lower" else 0
                symbols_by_staff.setdefault(staff_index, []).append(symbol)
            groups_by_staff: dict[int, list[int]] = {}
            for group_index, staff_index in compatibility.staff_by_group_index.items():
                groups_by_staff.setdefault(staff_index, []).append(group_index)
            for staff_index, staff_symbols in symbols_by_staff.items():
                staff_group_indices = groups_by_staff.get(staff_index, [])
                partial_symbols = {
                    group_index: compatibility.symbol_by_group_index[group_index]
                    for group_index in staff_group_indices
                    if group_index in compatibility.symbol_by_group_index
                }
                if partial_symbols:
                    for group_index, symbol in partial_symbols.items():
                        visual_groups[group_index].moment_id = moment_id
                        if symbol.visual_match_id not in symbol_index_by_match_id:
                            reserved_group_indices.add(group_index)
                            continue
                        assignment = (
                            symbol_index_by_match_id[symbol.visual_match_id],
                            group_index,
                        )
                        assignments.append(assignment)
                        fallback_assignments.add(assignment)
                    continue
                if len(staff_group_indices) != len(staff_symbols):
                    continue
                placeholder_symbols = [
                    symbol
                    for symbol in staff_symbols
                    if symbol.visual_match_id not in symbol_index_by_match_id
                ]
                group_order = sorted(
                    staff_group_indices,
                    key=lambda index: visual_groups[index].prediction_center[1],
                )
                # Transformer chord token order is the stable member order. Pitch
                # values are recognition output, not visual evidence, so they must
                # never be allowed to exchange two noteheads.
                for symbol, group_index in zip(staff_symbols, group_order, strict=True):
                    visual_groups[group_index].moment_id = moment_id
                    if symbol in placeholder_symbols:
                        reserved_group_indices.add(group_index)
                        continue
                    assignment = (
                        symbol_index_by_match_id[symbol.visual_match_id],
                        group_index,
                    )
                    assignments.append(assignment)
                    if compatibility.fallback_subset:
                        fallback_assignments.add(assignment)

        if not assignments and not reserved_group_indices:
            return None
        return StructuralMatchPlan(
            sorted(assignments),
            reserved_group_indices,
            fallback_assignments,
        )

    def _repair_aligned_visual_moment_staffs(
        self,
        symbol_moment: list[EncodedSymbol],
        visual_moment: list[int],
        visual_groups: list[VisualGroup],
    ) -> None:
        repaired_staffs = self._repairable_visual_moment_staffs(
            symbol_moment, visual_moment, visual_groups
        )
        if repaired_staffs is None:
            return
        self._apply_visual_moment_staffs(repaired_staffs, visual_groups)

    def _apply_visual_moment_staffs(
        self,
        repaired_staffs: dict[int, int],
        visual_groups: list[VisualGroup],
    ) -> None:
        for group_index, staff_index in repaired_staffs.items():
            group = visual_groups[group_index]
            if group.staff_index == staff_index:
                continue
            group.staff_index = staff_index
            staff_positions = self._duplicate_staff_positions_by_visual_id.get(group.visual_id, {})
            if staff_index in staff_positions:
                group.staff_position = staff_positions[staff_index]
            if "staff_membership_repaired" not in group.repair_actions:
                group.repair_actions.append("staff_membership_repaired")

    @classmethod
    def _construct_visual_moments(cls, visual_groups: list[VisualGroup]) -> list[list[int]]:
        """Build staff-normalized x moments while preserving physical chords."""
        if not visual_groups:
            return []
        widths = [max(group.prediction_notehead_size[0], 1.0) for group in visual_groups]
        typical_width = float(np.median(widths)) if widths else 10.0
        x_tolerance = max(
            2.0,
            typical_width * VISUAL_MOMENT_NOTEHEAD_WIDTH_RATIO,
        )

        # First combine noteheads that share a plausible physical stem. A
        # displaced second may sit outside the ordinary x tolerance.
        components: list[list[int]] = []
        remaining = set(range(len(visual_groups)))
        while remaining:
            seed = min(remaining)
            component = {seed}
            frontier = [seed]
            remaining.remove(seed)
            while frontier:
                current = frontier.pop()
                current_group = visual_groups[current]
                for candidate in list(remaining):
                    candidate_group = visual_groups[candidate]
                    if current_group.staff_index != candidate_group.staff_index:
                        continue
                    shared_components = set(current_group.owned_stem_component_ids).intersection(
                        candidate_group.owned_stem_component_ids
                    )
                    if shared_components and noteheads_can_share_chord_stem(
                        current_group, candidate_group
                    ):
                        remaining.remove(candidate)
                        component.add(candidate)
                        frontier.append(candidate)
            components.append(sorted(component))

        units = sorted(
            components,
            key=lambda component: float(
                np.median([visual_groups[index].prediction_center[0] for index in component])
            ),
        )
        moments: list[list[int]] = []
        for component in units:
            center = float(
                np.median([visual_groups[index].prediction_center[0] for index in component])
            )
            if moments:
                previous_center = float(
                    np.median([visual_groups[index].prediction_center[0] for index in moments[-1]])
                )
            else:
                previous_center = float("-inf")
            if moments and abs(center - previous_center) <= x_tolerance:
                moments[-1].extend(component)
            else:
                moments.append(list(component))

        # A simultaneous second is conventionally displaced left or right by
        # roughly one notehead width. This occurs both inside stemless chords and
        # between close opposing voices, and can put part of its stack just outside
        # the ordinary column tolerance. Rejoin adjacent stacks only when at least
        # one side contains a chord and the physical notehead outlines still touch;
        # two ordinary sequential singleton notes remain separate.
        merged_moments: list[list[int]] = []
        for moment in moments:
            if merged_moments and cls._moments_form_displaced_notehead_stack(
                merged_moments[-1], moment, visual_groups
            ):
                merged_moments[-1].extend(moment)
            else:
                merged_moments.append(list(moment))
        return merged_moments

    @classmethod
    def _moments_form_displaced_notehead_stack(
        cls,
        first_moment: list[int],
        second_moment: list[int],
        visual_groups: list[VisualGroup],
    ) -> bool:
        for staff_index in (0, 1):
            first_staff = [
                index for index in first_moment if visual_groups[index].staff_index == staff_index
            ]
            second_staff = [
                index for index in second_moment if visual_groups[index].staff_index == staff_index
            ]
            if (
                not first_staff
                or not second_staff
                or (len(first_staff) == 1 and len(second_staff) == 1)
            ):
                continue
            first_positions = {visual_groups[index].staff_position for index in first_staff}
            second_positions = {visual_groups[index].staff_position for index in second_staff}
            if first_positions & second_positions:
                # Parallel columns containing the same rounded staff positions are
                # commonly alternate segmentation evidence for hollow heads, not
                # two halves of a larger chord. Keep the columns separate so
                # attention can select the one that represents the token moment.
                continue

            other_staff_index = 1 - staff_index
            if any(
                visual_groups[index].staff_index == other_staff_index for index in first_moment
            ) and any(
                visual_groups[index].staff_index == other_staff_index for index in second_moment
            ):
                continue

            for first_index in first_staff:
                first_group = visual_groups[first_index]
                for second_index in second_staff:
                    second_group = visual_groups[second_index]
                    if not noteheads_can_share_chord_stem(first_group, second_group):
                        continue
                    maximum_vertical_distance = (
                        max(
                            first_group.prediction_notehead_size[1],
                            second_group.prediction_notehead_size[1],
                        )
                        * DISPLACED_CHORD_MAX_VERTICAL_NOTEHEAD_RATIO
                    )
                    if (
                        abs(first_group.prediction_center[1] - second_group.prediction_center[1])
                        <= maximum_vertical_distance
                    ):
                        return True
        return False

    @staticmethod
    def _symbol_occupies_visual_moment(symbol: EncodedSymbol) -> bool:
        if symbol.rhythm.startswith("note"):
            return True
        return symbol.rhythm.startswith("rest") and symbol.pitch not in ("_", ".")

    def _align_structural_moments(
        self,
        symbol_moments: list[list[EncodedSymbol]],
        visual_moments: list[list[int]],
        visual_groups: list[VisualGroup],
    ) -> list[tuple[int, int]]:
        """Return assignments common to every best order-preserving alignment.

        Insertions and deletions are ordinary gaps. Attention contributes a bounded
        local cost, so it can choose between otherwise equivalent gaps but cannot
        make two moments cross. If two repeated-note alignments remain equally good,
        neither disputed edge is returned.
        """

        def visual_shape(moment: list[int]) -> tuple[int, int]:
            return (
                sum(visual_groups[index].staff_index == 0 for index in moment),
                sum(visual_groups[index].staff_index == 1 for index in moment),
            )

        visual_shapes = [visual_shape(moment) for moment in visual_moments]
        symbol_count = len(symbol_moments)
        visual_count = len(visual_moments)
        if symbol_count == 0 or visual_count == 0:
            return []
        compatibilities = [
            [
                self._compatible_visual_moment(symbol_moment, visual_moment, visual_groups)
                for visual_moment in visual_moments
            ]
            for symbol_moment in symbol_moments
        ]
        visual_centers = [
            float(
                np.median(
                    [visual_groups[group_index].prediction_center[0] for group_index in moment]
                )
            )
            for moment in visual_moments
        ]
        widths = [
            max(
                (group.transformer_notehead_size or group.prediction_notehead_size)[0],
                1.0,
            )
            for group in visual_groups
        ]
        typical_width = float(np.median(widths)) if widths else 10.0
        # A staff line can split every head of a hollow chord into matching left
        # and right fragment columns. Neither column is a complete musical moment,
        # even though each has the expected per-staff count. Leave this distinctive
        # close duplicate pattern to attention matching and fragment rejoining.
        split_hollow_fragment_moments = {
            visual_index
            for visual_index, moment in enumerate(visual_moments)
            if sum(visual_shapes[visual_index]) > 1
            and all(
                visual_groups[group_index].is_hollow_notehead
                and not visual_groups[group_index].stem_contours
                for group_index in moment
            )
        }
        ambiguous_visual_moments = {
            visual_index
            for visual_index in split_hollow_fragment_moments
            if any(
                neighbor in split_hollow_fragment_moments
                and visual_shapes[neighbor] == visual_shapes[visual_index]
                and abs(visual_centers[neighbor] - visual_centers[visual_index])
                <= typical_width * 2
                for neighbor in (visual_index - 1, visual_index + 1)
                if 0 <= neighbor < visual_count
            )
        }

        def attention_cost(
            symbol_moment: list[EncodedSymbol],
            compatibility: StructuralMomentCompatibility,
        ) -> float:
            distances: list[float] = []
            for staff_index in (0, 1):
                staff_symbols = [
                    symbol
                    for symbol in symbol_moment
                    if (1 if symbol.position == "lower" else 0) == staff_index
                ]
                staff_groups = sorted(
                    [
                        index
                        for index, assigned_staff in compatibility.staff_by_group_index.items()
                        if assigned_staff == staff_index
                    ],
                    key=lambda index: visual_groups[index].prediction_center[1],
                )
                partial_pairs = [
                    (compatibility.symbol_by_group_index[group_index], group_index)
                    for group_index in staff_groups
                    if group_index in compatibility.symbol_by_group_index
                ]
                symbol_group_pairs = (
                    partial_pairs
                    if partial_pairs
                    else list(zip(staff_symbols, staff_groups, strict=True))
                )
                for symbol, group_index in symbol_group_pairs:
                    group = visual_groups[group_index]
                    if symbol.coordinates is None or group.transformer_center is None:
                        continue
                    distances.append(
                        float(
                            np.linalg.norm(
                                np.subtract(symbol.coordinates, group.transformer_center)
                            )
                        )
                    )
            if not distances:
                return 0.5
            return min(
                float(np.mean(distances)) / max(typical_width * 2, 1.0),
                0.4,
            )

        gap_cost = 1.0
        costs = [[float("inf") for _ in range(visual_count + 1)] for _ in range(symbol_count + 1)]
        common_pairs: list[list[set[tuple[int, int]]]] = [
            [set() for _ in range(visual_count + 1)] for _ in range(symbol_count + 1)
        ]
        costs[0][0] = 0.0
        for symbol_index in range(1, symbol_count + 1):
            costs[symbol_index][0] = symbol_index * gap_cost
        for visual_index in range(1, visual_count + 1):
            costs[0][visual_index] = visual_index * gap_cost

        for symbol_index in range(1, symbol_count + 1):
            for visual_index in range(1, visual_count + 1):
                options: list[tuple[float, set[tuple[int, int]]]] = [
                    (
                        costs[symbol_index - 1][visual_index] + gap_cost,
                        common_pairs[symbol_index - 1][visual_index],
                    ),
                    (
                        costs[symbol_index][visual_index - 1] + gap_cost,
                        common_pairs[symbol_index][visual_index - 1],
                    ),
                ]
                if (
                    compatibilities[symbol_index - 1][visual_index - 1] is not None
                    and visual_index - 1 not in ambiguous_visual_moments
                ):
                    compatibility = compatibilities[symbol_index - 1][visual_index - 1]
                    if compatibility is None:
                        raise RuntimeError("missing compatibility for an aligned moment")
                    pair = (symbol_index - 1, visual_index - 1)
                    symbol_position = (symbol_index - 0.5) / symbol_count
                    visual_position = (visual_index - 0.5) / visual_count
                    options.append(
                        (
                            costs[symbol_index - 1][visual_index - 1]
                            + 0.1
                            + abs(symbol_position - visual_position)
                            + attention_cost(
                                symbol_moments[symbol_index - 1],
                                compatibility,
                            ),
                            common_pairs[symbol_index - 1][visual_index - 1] | {pair},
                        )
                    )
                best_cost = min(option[0] for option in options)
                best_pair_sets = [
                    option[1] for option in options if abs(option[0] - best_cost) <= 1e-6
                ]
                costs[symbol_index][visual_index] = best_cost
                common = set(best_pair_sets[0])
                for pair_set in best_pair_sets[1:]:
                    common.intersection_update(pair_set)
                common_pairs[symbol_index][visual_index] = common

        return sorted(common_pairs[symbol_count][visual_count])

    def _compatible_visual_moment(
        self,
        symbol_moment: list[EncodedSymbol],
        visual_moment: list[int],
        visual_groups: list[VisualGroup],
    ) -> StructuralMomentCompatibility | None:
        """Select a structurally safe visual subset for one musical moment.

        An otherwise complete cross-staff moment can contain surplus candidates or
        be missing a chord head on one staff. Retain a subset only when attention,
        physical stems, or diatonic staff-position intervals uniquely prove it.
        The resulting links remain fallback evidence because the whole moment was
        not independently complete.
        """
        repaired_staffs = self._repairable_visual_moment_staffs(
            symbol_moment, visual_moment, visual_groups
        )
        if repaired_staffs is not None:
            return StructuralMomentCompatibility(repaired_staffs)

        staff_options: list[list[tuple[int, int]]] = []
        for group_index in visual_moment:
            group = visual_groups[group_index]
            positions = {
                group.staff_index: group.staff_position,
                **self._duplicate_staff_positions_by_visual_id.get(group.visual_id, {}),
            }
            staff_options.append(sorted(positions.items()))
        compatibilities: list[StructuralMomentCompatibility] = []
        for staff_choices in itertools.product(*staff_options):
            staff_by_group_index = {
                group_index: staff_index
                for group_index, (staff_index, _staff_position) in zip(
                    visual_moment, staff_choices, strict=True
                )
            }
            staff_position_by_group_index = {
                group_index: staff_position
                for group_index, (_staff_index, staff_position) in zip(
                    visual_moment, staff_choices, strict=True
                )
            }
            compatibility = self._compatible_visual_moment_for_staffs(
                symbol_moment,
                visual_moment,
                visual_groups,
                staff_by_group_index,
                staff_position_by_group_index,
            )
            if compatibility is not None:
                compatibilities.append(compatibility)
        if len(compatibilities) != 1:
            return None
        return compatibilities[0]

    @classmethod
    def _compatible_visual_moment_for_staffs(
        cls,
        symbol_moment: list[EncodedSymbol],
        visual_moment: list[int],
        visual_groups: list[VisualGroup],
        staff_by_group_index: dict[int, int],
        staff_position_by_group_index: dict[int, int],
    ) -> StructuralMomentCompatibility | None:
        selected_staffs: dict[int, int] = {}
        symbol_by_group_index: dict[int, EncodedSymbol] = {}
        pruned_any = False
        for staff_index in (0, 1):
            staff_symbols = [
                symbol
                for symbol in symbol_moment
                if (1 if symbol.position == "lower" else 0) == staff_index
            ]
            staff_group_indices = [
                group_index
                for group_index in visual_moment
                if staff_by_group_index[group_index] == staff_index
            ]
            if len(staff_group_indices) < len(staff_symbols):
                partial_mapping = cls._partial_staff_symbol_group_mapping(
                    staff_symbols,
                    staff_group_indices,
                    visual_groups,
                    staff_position_by_group_index,
                )
                if partial_mapping is None:
                    return None
                selected_staffs.update(dict.fromkeys(partial_mapping, staff_index))
                symbol_by_group_index.update(partial_mapping)
                pruned_any = True
                continue
            if len(staff_group_indices) == len(staff_symbols):
                selected_staffs.update(dict.fromkeys(staff_group_indices, staff_index))
                continue
            if not staff_symbols:
                return None
            selected_group_indices = cls._attention_selected_group_subset(
                staff_symbols, staff_group_indices, visual_groups
            )
            if selected_group_indices is None:
                selected_group_indices = cls._shared_stem_ordered_group_subset(
                    staff_symbols, staff_group_indices, visual_groups
                )
            if selected_group_indices is None:
                return None
            selected_staffs.update(dict.fromkeys(selected_group_indices, staff_index))
            pruned_any = True

        return StructuralMomentCompatibility(
            selected_staffs,
            pruned_any,
            symbol_by_group_index,
        )

    @classmethod
    def _partial_staff_symbol_group_mapping(
        cls,
        symbols: list[EncodedSymbol],
        group_indices: list[int],
        visual_groups: list[VisualGroup],
        staff_position_by_group_index: dict[int, int] | None = None,
    ) -> dict[int, EncodedSymbol] | None:
        """Map an incomplete chord only when diatonic intervals prove the subset."""
        if len(group_indices) < 2 or len(group_indices) >= len(symbols):
            return None
        pitched_symbols: list[tuple[int, EncodedSymbol]] = []
        for symbol in symbols:
            pitch_index = diatonic_pitch_index(symbol.pitch)
            if pitch_index is None:
                return None
            pitched_symbols.append((pitch_index, symbol))
        pitched_symbols.sort(key=lambda item: item[0], reverse=True)
        staff_positions = staff_position_by_group_index or {}
        ordered_groups = sorted(
            group_indices,
            key=lambda index: staff_positions.get(index, visual_groups[index].staff_position),
            reverse=True,
        )
        mappings: list[dict[int, EncodedSymbol]] = []
        for symbol_subset in itertools.combinations(pitched_symbols, len(ordered_groups)):
            offsets = [
                staff_positions.get(group_index, visual_groups[group_index].staff_position)
                - pitch_index
                for group_index, (pitch_index, _symbol) in zip(
                    ordered_groups, symbol_subset, strict=True
                )
            ]
            if max(offsets) - min(offsets) > 1:
                continue
            mappings.append(
                {
                    group_index: symbol
                    for group_index, (_pitch_index, symbol) in zip(
                        ordered_groups, symbol_subset, strict=True
                    )
                }
            )
        if len(mappings) != 1:
            return None
        return mappings[0]

    @staticmethod
    def _attention_selected_group_subset(
        symbols: list[EncodedSymbol],
        group_indices: list[int],
        visual_groups: list[VisualGroup],
    ) -> list[int] | None:
        """Return a mutually unique attention-backed subset of surplus groups."""
        widths = [
            max(
                (
                    visual_groups[group_index].transformer_notehead_size
                    or visual_groups[group_index].prediction_notehead_size
                )[0],
                1.0,
            )
            for group_index in group_indices
        ]
        typical_width = float(np.median(widths)) if widths else 10.0
        max_distance = max(6.0, typical_width * ATTENTION_MATCH_NOTEHEAD_WIDTH_RATIO)
        uniqueness_margin = max(1.0, typical_width * ATTENTION_UNIQUENESS_NOTEHEAD_WIDTH_RATIO)
        distances: dict[tuple[int, int], float] = {}
        for symbol_index, symbol in enumerate(symbols):
            if symbol.coordinates is None or not bool(np.all(np.isfinite(symbol.coordinates))):
                return None
            for group_index in group_indices:
                center = visual_groups[group_index].transformer_center
                if center is None or not bool(np.all(np.isfinite(center))):
                    continue
                distances[(symbol_index, group_index)] = float(
                    np.linalg.norm(np.subtract(symbol.coordinates, center))
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
        for symbol_index in range(len(symbols)):
            nearest = unique_nearest(
                [
                    (distance, group_index)
                    for (candidate_symbol, group_index), distance in distances.items()
                    if candidate_symbol == symbol_index
                ]
            )
            if nearest is None:
                return None
            nearest_group_by_symbol[symbol_index] = nearest[1]
        if len(set(nearest_group_by_symbol.values())) != len(symbols):
            return None

        for symbol_index, group_index in nearest_group_by_symbol.items():
            nearest = unique_nearest(
                [
                    (distance, candidate_symbol)
                    for (candidate_symbol, candidate_group), distance in distances.items()
                    if candidate_group == group_index
                ]
            )
            if nearest is None or nearest[1] != symbol_index:
                return None
        return list(nearest_group_by_symbol.values())

    @classmethod
    def _shared_stem_ordered_group_subset(
        cls,
        symbols: list[EncodedSymbol],
        group_indices: list[int],
        visual_groups: list[VisualGroup],
    ) -> list[int] | None:
        """Select an ordered subset only when its physical stem is unambiguous.

        Transformer attention is occasionally absent for a recognized member of a
        partially recognized chord. A common owned stem proves that the surplus
        heads belong to one physical chord rather than separate aligned voices. If
        one unrelated or fragmented candidate lacks that stem, accept the unique
        exact-count stem subset instead of rejecting the complete visual moment.
        Token member order can then select the corresponding visual heads without
        consulting predicted pitch. These assignments remain fallback evidence.
        """
        if not symbols or len(symbols) >= len(group_indices):
            return None
        ordered_groups = sorted(
            group_indices,
            key=lambda group_index: visual_groups[group_index].prediction_center[1],
        )

        def forms_physical_chord(candidate_indices: tuple[int, ...]) -> bool:
            common_components = set(visual_groups[candidate_indices[0]].owned_stem_component_ids)
            for group_index in candidate_indices[1:]:
                common_components.intersection_update(
                    visual_groups[group_index].owned_stem_component_ids
                )
            return bool(common_components) and all(
                noteheads_can_share_chord_stem(
                    visual_groups[first_index], visual_groups[second_index]
                )
                for first_index, second_index in itertools.combinations(candidate_indices, 2)
            )

        all_groups = tuple(ordered_groups)
        if forms_physical_chord(all_groups):
            return ordered_groups[: len(symbols)]

        exact_count_subsets = [
            candidate_indices
            for candidate_indices in itertools.combinations(ordered_groups, len(symbols))
            if forms_physical_chord(candidate_indices)
        ]
        if len(exact_count_subsets) != 1:
            return None
        return list(exact_count_subsets[0])

    @staticmethod
    def _repairable_visual_moment_staffs(
        symbol_moment: list[EncodedSymbol],
        visual_moment: list[int],
        visual_groups: list[VisualGroup],
    ) -> dict[int, int] | None:
        """Return a safe staff assignment for an exact-duplicate boundary head."""
        upper_count = sum(symbol.position != "lower" for symbol in symbol_moment)
        lower_count = len(symbol_moment) - upper_count
        if len(visual_moment) != upper_count + lower_count:
            return None
        current = {
            group_index: visual_groups[group_index].staff_index for group_index in visual_moment
        }
        if (
            sum(staff_index == 0 for staff_index in current.values()) == upper_count
            and sum(staff_index == 1 for staff_index in current.values()) == lower_count
        ):
            return current

        vertical_order = sorted(
            visual_moment,
            key=lambda group_index: visual_groups[group_index].prediction_center[1],
        )
        repaired = {
            group_index: 0 if order_index < upper_count else 1
            for order_index, group_index in enumerate(vertical_order)
        }
        changed_groups = [
            visual_groups[group_index]
            for group_index in visual_moment
            if current[group_index] != repaired[group_index]
        ]
        if not changed_groups or any(
            "cross_staff_duplicate_consolidated" not in group.repair_actions
            for group in changed_groups
        ):
            return None
        return repaired
