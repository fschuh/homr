from typing import Any

import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Note, Staff
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols
from homr.visual_sidecar.candidate_cleanup import CandidateCleaner
from homr.visual_sidecar.chords import ChordResolver
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform
from homr.visual_sidecar.models import (
    MusicXmlNoteRecord,
    SidecarState,
    VisualGroup,
    VisualMatch,
    sounding_pitch,
)
from homr.visual_sidecar.moments import MomentMatcher
from homr.visual_sidecar.noteheads import NoteheadGeometry
from homr.visual_sidecar.recovery import RecoveryManager
from homr.visual_sidecar.sequence import SequenceMatcher
from homr.visual_sidecar.serialization import (
    VisualSidecarSerializer,
)
from homr.visual_sidecar.serialization import write_visual_sidecar as write_document
from homr.visual_sidecar.stems import StemGeometry


class VisualSidecarBuilder:
    def __init__(
        self,
        coordinate_transform: PredictionCoordinateTransform,
        stem_fragments: list[RotatedBoundingBox] | None = None,
        notehead_mask: Any | None = None,
        notehead_candidates: list[Any] | None = None,
        source_image: Any | None = None,
    ) -> None:
        self.coordinate_transform = coordinate_transform
        self.stem_fragments = stem_fragments or []
        self.notehead_mask = notehead_mask
        self.notehead_candidates = notehead_candidates or []
        self.source_image = source_image
        self.state = SidecarState()

        # These collections are intentionally part of the builder's inspection API.
        self.visual_groups = self.state.visual_groups
        self.matches_by_symbol_id = self.state.matches_by_symbol_id
        self.musicxml_notes = self.state.musicxml_notes
        self.unmatched_visual_notes = self.state.unmatched_visual_notes

        self.stems = StemGeometry(self.stem_fragments)
        self.noteheads = NoteheadGeometry(coordinate_transform, notehead_mask, source_image)
        self.recovery = RecoveryManager(
            self.state,
            self.notehead_candidates,
            notehead_mask,
            self.stems,
        )
        self.candidate_cleaner = CandidateCleaner(self.state, coordinate_transform)
        self.moments = MomentMatcher(self.state)
        self.sequence = SequenceMatcher()
        self.chords = ChordResolver(
            self.state,
            coordinate_transform,
            source_image,
            self.noteheads,
            self.stems,
        )
        self.serializer = VisualSidecarSerializer(
            self.state,
            coordinate_transform,
            self.stem_fragments,
            self.chords,
        )

    def prepare_recovery_notes(self, staffs: list[Staff]) -> None:
        self.recovery.prepare(staffs)

    def recovery_notes_for_staff(self, staff: Staff) -> list[Note]:
        return self.recovery.for_staff(staff)

    def add_staff_visual_notes(
        self, staff_index: int, original_notes: list[Note], transformed_notes: list[Note]
    ) -> None:
        stem_ownership = self.state.stem_ownership_cache or self.stems.build_ownership_cache(
            original_notes
        )
        for original, transformed in zip(original_notes, transformed_notes, strict=False):
            if original.visual_id is None:
                continue
            notehead_contour = self.coordinate_transform.prediction_contour_to_source(
                original.box.polygon
            )
            detected_notehead_contour = self.noteheads.detected_notehead_contour(original)
            refined_notehead_contour = self.noteheads.refined_notehead_contour(
                original, original_notes
            )
            recovered_stretched_notehead = (
                self.noteheads.is_stretched_notehead(original)
                and refined_notehead_contour is not None
            )
            is_hollow_notehead = self.noteheads.is_hollow_notehead(original)
            if recovered_stretched_notehead:
                notehead_contour = refined_notehead_contour
                notehead_ellipse = self.noteheads.ellipse_from_source_contour(
                    refined_notehead_contour
                )
            else:
                notehead_ellipse = self.noteheads.notehead_ellipse(original)
            notehead_ellipse["_is_hollow"] = is_hollow_notehead
            detected_stem_contours = []
            if original.stem is not None:
                detected_stem_contours.append(
                    self.coordinate_transform.prediction_contour_to_source(original.stem.contours)
                )
            stem_contours = []
            stem = self.stems.stem_for_note(original, stem_ownership)
            if stem is not None:
                stem_contours.append(
                    self.coordinate_transform.prediction_contour_to_source(stem.polygon)
                )
            repair_actions: list[str] = []
            if recovered_stretched_notehead:
                repair_actions.append("refined_stretched_notehead")
            if stem is not None and (
                original.stem is None or not self.stems.same_fragment(stem, original.stem)
            ):
                repair_actions.append("stem_geometry_repaired")
            provenance = (
                "recovered_candidate"
                if original.visual_id.startswith("vnote-recovered-")
                else "segmentation"
            )
            owned_stem_component_ids = sorted(
                f"staff-{staff_index}-stem-{component}"
                for component, owner_note_ids in stem_ownership.owner_note_ids_by_component.items()
                if id(original) in owner_note_ids and len(owner_note_ids) > 1
            )
            self.visual_groups[original.visual_id] = VisualGroup(
                visual_id=original.visual_id,
                staff_index=staff_index,
                stave_index=self.state.stave_index_by_visual_id.get(original.visual_id, 0),
                staff_position=original.position,
                prediction_center=original.center,
                prediction_notehead_size=(
                    float(original.box.size[0]),
                    float(original.box.size[1]),
                ),
                transformer_center=transformed.center,
                transformer_notehead_size=(
                    float(transformed.box.size[0]),
                    float(transformed.box.size[1]),
                ),
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
                visual_status="fallback",
                provenance=provenance,
                repair_actions=repair_actions,
            )
            self.unmatched_visual_notes.add(original.visual_id)

    def add_staff_matches(
        self,
        symbols: list[EncodedSymbol],
        staff_index: int,
        source_staff: Staff | None = None,
    ) -> None:
        """Repair, organize, and align one staff without changing recognition.

        This is deliberately a post-inference pipeline. It works from the cleaned
        transformer symbols and pixel-backed candidates, but never changes either
        the symbol stream used for MusicXML or the detected staff itself.
        """
        # Match the same cleaned symbol identities that MusicXML generation retains.
        symbols = remove_duplicated_symbols(symbols, cleanup_tuplets=False)
        self.candidate_cleaner.repair(symbols, staff_index)
        # Stage 3 constructs physical chord units and normalized visual moments;
        # stage 4 performs the order-preserving global sequence alignment.
        visual_groups = [
            group
            for group in self.visual_groups.values()
            if group.staff_index == staff_index and group.visual_status != "diagnostic"
        ]
        note_symbols = [
            symbol
            for symbol in symbols
            if symbol.rhythm.startswith(("note", "rest")) and symbol.pitch not in ("_", ".")
        ]
        moment_plan = self.moments.structural_assignments(symbols, note_symbols, visual_groups)
        locked_assignments = moment_plan.assignments if moment_plan is not None else []
        reserved_group_indices = (
            moment_plan.reserved_group_indices if moment_plan is not None else set()
        )
        fallback_assignments = (
            moment_plan.fallback_assignments if moment_plan is not None else set()
        )
        assignments = self.sequence.assign(
            note_symbols,
            visual_groups,
            locked_assignments,
            reserved_group_indices,
        )
        assignments = self.sequence.release_split_moment_outliers(
            symbols, note_symbols, visual_groups, assignments
        )
        assigned_symbols = {symbol_index for symbol_index, _ in assignments}
        locked_pairs = set(locked_assignments)

        assigned_group_by_symbol_id: dict[int, VisualGroup] = {}
        for symbol_index, group_index in assignments:
            symbol = note_symbols[symbol_index]
            visual_group = visual_groups[group_index]
            assigned_group_by_symbol_id[symbol.visual_match_id] = visual_group
            confidence = self._score_match(symbol, visual_group)
            visual_group.duration = symbol.rhythm
            alignment_method = (
                "sequence_repair"
                if (symbol_index, group_index) in fallback_assignments
                else (
                    "structural"
                    if (symbol_index, group_index) in locked_pairs
                    else ("sequence_repair" if symbol.coordinates is None else "attention")
                )
            )
            if alignment_method == "structural":
                visual_group.visual_status = (
                    "fallback" if visual_group.provenance == "recovered_candidate" else "canonical"
                )
            else:
                visual_group.visual_status = "fallback"
                repair_action = f"{alignment_method}_aligned"
                if repair_action not in visual_group.repair_actions:
                    visual_group.repair_actions.append(repair_action)
                visual_group.moment_id = self.state.moment_id_by_symbol_id.get(
                    symbol.visual_match_id
                )
            self.matches_by_symbol_id[symbol.visual_match_id] = VisualMatch(
                symbol=symbol,
                visual_id=visual_group.visual_id,
                confidence=confidence,
                alignment_method=alignment_method,
            )
            self.unmatched_visual_notes.discard(visual_group.visual_id)

        token_moment_by_symbol_id = {
            symbol.visual_match_id: moment
            for moment in self.moments.token_moments(symbols)
            for symbol in moment
        }
        assigned_visual_ids = {group.visual_id for group in assigned_group_by_symbol_id.values()}
        assigned_visual_ids.update(
            visual_groups[index].visual_id for index in reserved_group_indices
        )
        pending_symbols = [
            symbol
            for symbol_index, symbol in enumerate(note_symbols)
            if symbol_index not in assigned_symbols
        ]
        while pending_symbols:
            still_pending: list[EncodedSymbol] = []
            recovered_any = False
            for symbol in pending_symbols:
                chord_mates = [
                    (mate, assigned_group_by_symbol_id[mate.visual_match_id])
                    for mate in token_moment_by_symbol_id.get(symbol.visual_match_id, [])
                    if (
                        mate.visual_match_id != symbol.visual_match_id
                        and mate.visual_match_id in assigned_group_by_symbol_id
                    )
                ]
                recovered_group = self.chords.recover_transformer_notehead(
                    symbol,
                    staff_index,
                    source_staff=source_staff,
                    neighboring_groups=visual_groups,
                    chord_mates=chord_mates,
                    available_groups=[
                        group
                        for group in visual_groups
                        if group.visual_id not in assigned_visual_ids
                    ],
                )
                if recovered_group is None:
                    still_pending.append(symbol)
                    continue
                if recovered_group.visual_id not in self.visual_groups:
                    self.visual_groups[recovered_group.visual_id] = recovered_group
                    visual_groups.append(recovered_group)
                else:
                    recovered_group.visual_status = "fallback"
                    recovered_group.duration = symbol.rhythm
                    if (
                        "transformer_chord_candidate_recovered"
                        not in recovered_group.repair_actions
                    ):
                        recovered_group.repair_actions.append(
                            "transformer_chord_candidate_recovered"
                        )
                    self.unmatched_visual_notes.discard(recovered_group.visual_id)
                self.matches_by_symbol_id[symbol.visual_match_id] = VisualMatch(
                    symbol=symbol,
                    visual_id=recovered_group.visual_id,
                    confidence=self._score_match(symbol, recovered_group),
                    alignment_method="sequence_repair",
                )
                recovered_group.moment_id = self.state.moment_id_by_symbol_id.get(
                    symbol.visual_match_id
                )
                assigned_group_by_symbol_id[symbol.visual_match_id] = recovered_group
                assigned_visual_ids.add(recovered_group.visual_id)
                recovered_any = True
            pending_symbols = still_pending
            if not recovered_any:
                break

        for symbol in pending_symbols:
            self.matches_by_symbol_id[symbol.visual_match_id] = VisualMatch(
                symbol=symbol,
                visual_id=None,
                confidence=0.0,
                alignment_method="none",
            )

        self.candidate_cleaner.merge_split_whole_note_fragments(staff_index)
        self.chords.assign_physical_chord_ids(symbols, staff_index)
        self._finalize_unmatched_groups(staff_index)

    def _finalize_unmatched_groups(self, staff_index: int) -> None:
        for visual_id in sorted(self.unmatched_visual_notes):
            group = self.visual_groups.get(visual_id)
            if group is None or group.staff_index != staff_index:
                continue
            group.visual_status = "diagnostic"
            if not any(
                action in ("clef_artifact", "suspected_duplicate")
                or action.startswith("merged_into:")
                for action in group.repair_actions
            ):
                group.repair_actions.append("unmatched_candidate")

    def create_musicxml_id(self) -> str:
        musicxml_id = f"homr-note-{self.state.next_musicxml_note_id}"
        self.state.next_musicxml_note_id += 1
        return musicxml_id

    def record_musicxml_note(
        self,
        musicxml_id: str,
        symbol: EncodedSymbol,
        *,
        part: int,
        measure: int,
        staff: int,
        voice: int,
    ) -> None:
        match = self.matches_by_symbol_id.get(symbol.visual_match_id)
        visual_id = match.visual_id if match is not None else None
        confidence = match.confidence if match is not None else 0.0
        alignment_method = match.alignment_method if match is not None else "none"
        pitch = sounding_pitch(symbol)
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
                alignment_method=alignment_method,
            )
        )

    def unmatched_musicxml_notes(self) -> list[str]:
        return [note.musicxml_id for note in self.musicxml_notes if note.visual_group_id is None]

    def to_json_dict(self) -> dict[str, Any]:
        return self.serializer.to_dict()

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


def write_visual_sidecar(path: str, builder: VisualSidecarBuilder) -> None:
    write_document(path, builder.to_json_dict())
