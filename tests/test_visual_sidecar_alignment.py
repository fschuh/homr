import unittest

import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import Note, Staff, StaffPoint
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from tests.visual_sidecar_helpers import (
    diagnostic_visual_group_ids,
    ellipse_contour,
    linked_visual_id,
)


class TestVisualSidecarBuilderAlignment(unittest.TestCase):
    def test_visual_notes_are_matched_by_attention_position_not_flat_cursor(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = ellipse_contour((x, y), (5, 4), 0, 10)
            return Note(
                BoundingEllipse(((x, y), (10, 8), 0), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        notes = [
            make_note(20, 20, "upper-left"),
            make_note(20, 80, "lower-left"),
            make_note(60, 20, "upper-right"),
            make_note(60, 80, "lower-right"),
        ]
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        symbols = [
            EncodedSymbol("note_4", "C5", coordinates=(20, 20)),
            EncodedSymbol("note_8", "D5", coordinates=(60, 20)),
            EncodedSymbol("note_16", "C3", coordinates=(20, 80)),
            EncodedSymbol("note_2", "D3", coordinates=(60, 80)),
        ]

        builder.add_staff_matches(symbols, 0)

        self.assertEqual(builder.matches_by_symbol_id[id(symbols[0])].visual_id, "upper-left")
        self.assertEqual(builder.matches_by_symbol_id[id(symbols[1])].visual_id, "upper-right")
        self.assertEqual(builder.matches_by_symbol_id[id(symbols[2])].visual_id, "lower-left")
        self.assertEqual(builder.matches_by_symbol_id[id(symbols[3])].visual_id, "lower-right")

    def test_token_chord_order_maps_heads_without_using_predicted_pitch(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        stem_contour = np.array(
            [[[49, 15]], [[51, 15]], [[51, 75]], [[49, 75]]],
            dtype=np.float32,
        )
        shared_stem = RotatedBoundingBox(cv2.minAreaRect(stem_contour), stem_contour)

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = ellipse_contour((x, y), (10, 7), -20, 5)
            return Note(
                BoundingEllipse(((x, y), (20, 14), -20), contour),
                position=4,
                stem=shared_stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        top = make_note(60, 30, "top")
        displaced_bottom = make_note(40, 60, "displaced-bottom")
        builder = VisualSidecarBuilder(coordinate_transform, stem_fragments=[shared_stem])
        builder.add_staff_visual_notes(
            0,
            [top, displaced_bottom],
            [top.copy(), displaced_bottom.copy()],
        )
        token_first_with_wrong_low_pitch = EncodedSymbol(
            "note_8", "C3", position="upper", coordinates=(40, 60)
        )
        token_second_with_wrong_high_pitch = EncodedSymbol(
            "note_8", "C6", position="upper", coordinates=(60, 30)
        )

        builder.add_staff_matches(
            [
                token_first_with_wrong_low_pitch,
                EncodedSymbol("chord"),
                token_second_with_wrong_high_pitch,
            ],
            0,
        )

        self.assertEqual(
            builder.matches_by_symbol_id[
                token_first_with_wrong_low_pitch.visual_match_id
            ].visual_id,
            "top",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[
                token_second_with_wrong_high_pitch.visual_match_id
            ].visual_id,
            "displaced-bottom",
        )
        self.assertEqual(
            builder.visual_groups["top"].moment_id,
            builder.visual_groups["displaced-bottom"].moment_id,
        )
        self.assertEqual(
            builder.visual_groups["top"].chord_id,
            builder.visual_groups["displaced-bottom"].chord_id,
        )
        self.assertIsNotNone(builder.visual_groups["top"].chord_id)

    def test_ambiguous_repeated_notes_remain_unmatched(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        contour = ellipse_contour((50, 40), (7, 5), -20, 5)
        note = Note(
            BoundingEllipse(((50, 40), (14, 10), -20), contour),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="repeated-candidate",
        )
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(0, [note], [note.copy()])
        first = EncodedSymbol("note_16", "C5", coordinates=(50, 40))
        second = EncodedSymbol("note_16", "C5", coordinates=(50, 40))

        builder.add_staff_matches([first, second], 0)

        self.assertIsNone(builder.matches_by_symbol_id[first.visual_match_id].visual_id)
        self.assertIsNone(builder.matches_by_symbol_id[second.visual_match_id].visual_id)
        group = builder.to_json_dict()["visual_groups"][0]
        self.assertEqual(group["visual_status"], "diagnostic")
        self.assertEqual(
            diagnostic_visual_group_ids(builder.to_json_dict()),
            ["repeated-candidate"],
        )

    def test_shared_stem_repairs_chord_member_swapped_with_neighbor(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        shared_stem_contour = np.array(
            [[[64, 25]], [[66, 25]], [[66, 70]], [[64, 70]]], dtype=np.float32
        )
        shared_stem = RotatedBoundingBox(cv2.minAreaRect(shared_stem_contour), shared_stem_contour)

        def make_note(
            x: int,
            y: int,
            visual_id: str,
            stem: RotatedBoundingBox | None = None,
        ) -> Note:
            contour = ellipse_contour((x, y), (7, 5), -20, 5)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        single = make_note(40, 80, "single")
        chord_top = make_note(60, 40, "chord-top", shared_stem)
        chord_bottom = make_note(60, 60, "chord-bottom", shared_stem)
        extra = make_note(110, 100, "extra")
        builder = VisualSidecarBuilder(coordinate_transform, stem_fragments=[shared_stem])
        builder.add_staff_visual_notes(
            0,
            [single, chord_top, chord_bottom, extra],
            [single.copy(), chord_top.copy(), chord_bottom.copy(), extra.copy()],
        )
        single_symbol = EncodedSymbol("note_16", "G4", coordinates=(60, 60))
        chord_top_symbol = EncodedSymbol("note_16", "D5", coordinates=(60, 40))
        chord_bottom_symbol = EncodedSymbol("note_16", "Bb4", coordinates=(40, 80))

        builder.add_staff_matches(
            [
                single_symbol,
                chord_top_symbol,
                EncodedSymbol("chord"),
                chord_bottom_symbol,
            ],
            0,
        )

        self.assertEqual(
            builder.matches_by_symbol_id[single_symbol.visual_match_id].visual_id,
            "single",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[chord_top_symbol.visual_match_id].visual_id,
            "chord-top",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[chord_bottom_symbol.visual_match_id].visual_id,
            "chord-bottom",
        )

    def test_adjacent_opposing_stems_do_not_swap_neighbor_with_chord_member(
        self,
    ) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )

        def stem(x: int, top: int, bottom: int) -> RotatedBoundingBox:
            contour = np.array(
                [[[x - 1, top]], [[x + 1, top]], [[x + 1, bottom]], [[x - 1, bottom]]],
                dtype=np.float32,
            )
            return RotatedBoundingBox(cv2.minAreaRect(contour), contour)

        preceding_up_stem = stem(58, 20, 50)
        chord_top_up_stem = stem(86, 5, 35)
        chord_bottom_down_stem = stem(66, 60, 95)

        def note(
            x: int,
            y: int,
            visual_id: str,
            note_stem: RotatedBoundingBox,
        ) -> Note:
            contour = ellipse_contour((x, y), (10, 7), -20, 5)
            return Note(
                BoundingEllipse(((x, y), (20, 14), -20), contour),
                position=4,
                stem=note_stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        preceding = note(48, 50, "preceding", preceding_up_stem)
        chord_top = note(76, 35, "chord-top", chord_top_up_stem)
        chord_bottom = note(76, 60, "chord-bottom", chord_bottom_down_stem)
        notes = [preceding, chord_top, chord_bottom]
        builder = VisualSidecarBuilder(
            coordinate_transform,
            stem_fragments=[
                preceding_up_stem,
                chord_top_up_stem,
                chord_bottom_down_stem,
            ],
        )
        builder.add_staff_visual_notes(
            0,
            notes,
            [candidate.copy() for candidate in notes],
        )
        preceding_symbol = EncodedSymbol("note_32", "G4", coordinates=(48, 50))
        chord_top_symbol = EncodedSymbol("note_8", "B4", coordinates=(76, 35))
        chord_bottom_symbol = EncodedSymbol("note_4", "F#4", coordinates=(76, 60))

        builder.add_staff_matches(
            [
                preceding_symbol,
                chord_top_symbol,
                EncodedSymbol("chord"),
                chord_bottom_symbol,
            ],
            0,
        )

        self.assertEqual(
            builder.matches_by_symbol_id[preceding_symbol.visual_match_id].visual_id,
            "preceding",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[chord_top_symbol.visual_match_id].visual_id,
            "chord-top",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[chord_bottom_symbol.visual_match_id].visual_id,
            "chord-bottom",
        )
        self.assertTrue(
            set(builder.visual_groups["preceding"].owned_stem_component_ids).isdisjoint(
                builder.visual_groups["chord-bottom"].owned_stem_component_ids
            )
        )
        for visual_id in ("preceding", "chord-bottom"):
            points = builder.visual_groups[visual_id].stem_contours[0]
            self.assertLess(
                max(point[1] for point in points) - min(point[1] for point in points),
                50,
            )

    def test_complete_moments_override_repeated_note_attention_across_staffs(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = ellipse_contour((x, y), (7, 5), -20, 5)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        upper_first = make_note(20, 35, "upper-first")
        lower_first = make_note(20, 95, "lower-first")
        upper_second = make_note(60, 35, "upper-second")
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(
            0,
            [upper_first, lower_first, upper_second],
            [upper_first.copy(), lower_first.copy(), upper_second.copy()],
        )
        builder.visual_groups["lower-first"].staff_index = 1
        first_upper_symbol = EncodedSymbol("note_16", "Gb4", position="upper", coordinates=(60, 35))
        first_lower_symbol = EncodedSymbol("note_2", "Bb3", position="lower", coordinates=(20, 95))
        second_upper_symbol = EncodedSymbol(
            "note_16", "Gb4", position="upper", coordinates=(20, 35)
        )

        builder.add_staff_matches(
            [
                first_upper_symbol,
                EncodedSymbol("chord"),
                first_lower_symbol,
                second_upper_symbol,
            ],
            0,
        )

        self.assertEqual(
            builder.matches_by_symbol_id[first_upper_symbol.visual_match_id].visual_id,
            "upper-first",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[first_lower_symbol.visual_match_id].visual_id,
            "lower-first",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[second_upper_symbol.visual_match_id].visual_id,
            "upper-second",
        )
        first_upper_group = builder.visual_groups["upper-first"]
        first_lower_group = builder.visual_groups["lower-first"]
        self.assertEqual(first_upper_group.moment_id, first_lower_group.moment_id)
        self.assertIsNone(first_upper_group.chord_id)
        self.assertIsNone(first_lower_group.chord_id)

    def test_cross_staff_duplicate_at_ledger_boundary_is_consolidated(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(180, 180),
            autocrop_box=(0, 0, 180, 180),
            cropped_size=(180, 180),
            resized_size=(180, 180),
            resize_scale=(1.0, 1.0),
            prediction_size=(180, 180),
        )
        upper_point = StaffPoint(80, [20, 30, 40, 50, 60], 0)
        lower_point = StaffPoint(80, [110, 120, 130, 140, 150], 0)
        upper_staff = Staff([upper_point])
        lower_staff = Staff([lower_point])

        def make_note(
            y: int, visual_id: str, point: StaffPoint, box: BoundingEllipse | None = None
        ) -> Note:
            notehead = box or BoundingEllipse(
                ((80, y), (14, 10), -20),
                ellipse_contour((80, y), (7, 5), -20, 5),
            )
            return Note(
                notehead,
                position=point.find_position_in_unit_sizes(notehead),
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        upper_top = make_note(30, "upper-top", upper_point)
        upper_bottom = make_note(50, "upper-bottom", upper_point)
        shared_boundary_box = BoundingEllipse(
            ((80, 85), (14, 10), -20),
            ellipse_contour((80, 85), (7, 5), -20, 5),
        )
        boundary_upper = make_note(85, "boundary-a-upper", upper_point, shared_boundary_box)
        boundary_lower = make_note(85, "boundary-b-lower", lower_point, shared_boundary_box)
        lower_middle = make_note(120, "lower-middle", lower_point)
        lower_bottom = make_note(140, "lower-bottom", lower_point)
        for note in (upper_top, upper_bottom, boundary_upper):
            upper_staff.add_symbol(note)
        for note in (boundary_lower, lower_middle, lower_bottom):
            lower_staff.add_symbol(note)
        grand_staff = upper_staff.merge(lower_staff)
        notes = grand_staff.get_notes()
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.prepare_recovery_notes([grand_staff])
        builder.add_staff_visual_notes(0, notes, [candidate.copy() for candidate in notes])

        upper_top_symbol = EncodedSymbol("note_16", "C6", position="upper", coordinates=(20, 140))
        upper_bottom_symbol = EncodedSymbol(
            "note_16", "C5", position="upper", coordinates=(20, 120)
        )
        lower_top_symbol = EncodedSymbol("note_8", "C4", position="lower", coordinates=(20, 50))
        lower_middle_symbol = EncodedSymbol("note_8", "C3", position="lower", coordinates=(20, 30))
        lower_bottom_symbol = EncodedSymbol("note_8", "C2", position="lower", coordinates=(20, 20))
        builder.add_staff_matches(
            [
                upper_top_symbol,
                EncodedSymbol("chord"),
                upper_bottom_symbol,
                EncodedSymbol("chord"),
                lower_top_symbol,
                EncodedSymbol("chord"),
                lower_middle_symbol,
                EncodedSymbol("chord"),
                lower_bottom_symbol,
            ],
            0,
        )

        self.assertEqual(
            builder.matches_by_symbol_id[lower_top_symbol.visual_match_id].visual_id,
            "boundary-a-upper",
        )
        repaired = builder.visual_groups["boundary-a-upper"]
        rejected = builder.visual_groups["boundary-b-lower"]
        self.assertEqual(repaired.staff_index, 1)
        self.assertEqual(repaired.staff_position, boundary_lower.position)
        self.assertEqual(repaired.visual_status, "canonical")
        self.assertIn("duplicate_candidates_consolidated", repaired.repair_actions)
        self.assertIn("staff_membership_repaired", repaired.repair_actions)
        self.assertEqual(rejected.visual_status, "diagnostic")
        self.assertIn("suspected_duplicate", rejected.repair_actions)
        self.assertEqual(builder.unmatched_visual_group_ids, {"boundary-b-lower"})
        self.assertEqual(
            {
                builder.visual_groups[
                    linked_visual_id(builder.matches_by_symbol_id[symbol.visual_match_id])
                ].moment_id
                for symbol in (
                    upper_top_symbol,
                    upper_bottom_symbol,
                    lower_top_symbol,
                    lower_middle_symbol,
                    lower_bottom_symbol,
                )
            },
            {"moment-1-1"},
        )

    def test_surplus_notehead_in_one_moment_does_not_disable_other_structural_matches(
        self,
    ) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(140, 120),
            autocrop_box=(0, 0, 140, 120),
            cropped_size=(140, 120),
            resized_size=(140, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 120),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = ellipse_contour((x, y), (7, 5), -20, 5)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        first = make_note(20, 35, "first")
        second = make_note(60, 35, "second")
        third = make_note(100, 35, "third")
        surplus = make_note(100, 55, "surplus")
        notes = [first, second, third, surplus]
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        first_symbol = EncodedSymbol("note_16", "C5", position="upper", coordinates=(60, 35))
        second_symbol = EncodedSymbol("note_16", "D5", position="upper", coordinates=(20, 35))
        third_symbol = EncodedSymbol("note_16", "E5", position="upper", coordinates=(100, 35))

        builder.add_staff_matches(
            [first_symbol, second_symbol, third_symbol],
            0,
        )

        self.assertEqual(
            builder.matches_by_symbol_id[first_symbol.visual_match_id].visual_id,
            "first",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[second_symbol.visual_match_id].visual_id,
            "second",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[third_symbol.visual_match_id].visual_id,
            "third",
        )
        self.assertEqual(builder.unmatched_visual_group_ids, {"surplus"})

    def test_attention_anchor_repairs_exact_other_staff_beside_surplus_head(
        self,
    ) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(140, 160),
            autocrop_box=(0, 0, 140, 160),
            cropped_size=(140, 160),
            resized_size=(140, 160),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 160),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = ellipse_contour((x, y), (7, 5), -20, 5)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        upper_anchor = make_note(60, 30, "upper-anchor")
        surplus_upper = make_note(60, 50, "surplus-upper")
        lower_top = make_note(60, 100, "lower-top")
        lower_bottom = make_note(60, 120, "lower-bottom")
        original_notes = [
            upper_anchor,
            surplus_upper,
            lower_top,
            lower_bottom,
        ]
        transformed_notes = [note.copy() for note in original_notes]
        transformed_notes[0].center = (20, 30)
        transformed_notes[1].center = (90, 50)
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(
            0,
            original_notes,
            transformed_notes,
        )
        builder.visual_groups["lower-top"].staff_index = 1
        builder.visual_groups["lower-bottom"].staff_index = 1

        upper_symbol = EncodedSymbol("note_16", "Eb6", position="upper", coordinates=(20, 30))
        lower_top_symbol = EncodedSymbol("note_8", "Bb4", position="lower")
        lower_bottom_symbol = EncodedSymbol("note_8", "Db4", position="lower")
        builder.add_staff_matches(
            [
                upper_symbol,
                EncodedSymbol("chord"),
                lower_top_symbol,
                EncodedSymbol("chord"),
                lower_bottom_symbol,
            ],
            0,
        )

        expected_matches = {
            upper_symbol.visual_match_id: "upper-anchor",
            lower_top_symbol.visual_match_id: "lower-top",
            lower_bottom_symbol.visual_match_id: "lower-bottom",
        }
        for match_id, visual_id in expected_matches.items():
            match = builder.matches_by_symbol_id[match_id]
            self.assertEqual(match.visual_id, visual_id)
            self.assertEqual(match.alignment_method, "sequence_repair")
            self.assertEqual(
                builder.visual_groups[visual_id].visual_status,
                "fallback",
            )
        self.assertEqual(
            {builder.visual_groups[visual_id].moment_id for visual_id in expected_matches.values()},
            {"moment-1-1"},
        )
        self.assertEqual(builder.unmatched_visual_group_ids, {"surplus-upper"})
        self.assertEqual(
            builder.visual_groups["surplus-upper"].visual_status,
            "diagnostic",
        )

    def test_unique_missing_moment_repairs_transformer_cross_staff_link(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(140, 180),
            autocrop_box=(0, 0, 140, 180),
            cropped_size=(140, 180),
            resized_size=(140, 180),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 180),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = ellipse_contour((x, y), (7, 5), -20, 5)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        previous = make_note(20, 140, "previous-lower")
        cross_staff_candidate = make_note(60, 70, "cross-staff-candidate")
        following = make_note(100, 140, "following-lower")
        notes = [previous, cross_staff_candidate, following]
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        for visual_id in ("previous-lower", "following-lower"):
            builder.visual_groups[visual_id].staff_index = 1
        builder.visual_groups["cross-staff-candidate"].staff_position = -5

        previous_symbol = EncodedSymbol("note_8", "D2", position="lower", coordinates=(20, 140))
        misplaced_symbol = EncodedSymbol(
            "note_8",
            "F3",
            lift="#",
            position="lower",
            coordinates=(60, 140),
        )
        following_symbol = EncodedSymbol("note_8", "D2", position="lower", coordinates=(100, 140))
        builder.add_staff_matches(
            [
                EncodedSymbol("clef_G2", position="upper"),
                EncodedSymbol("clef_F4", position="lower"),
                previous_symbol,
                misplaced_symbol,
                following_symbol,
            ],
            0,
        )

        match = builder.matches_by_symbol_id[misplaced_symbol.visual_match_id]
        repaired_group = builder.visual_groups["cross-staff-candidate"]
        self.assertEqual(match.visual_id, "cross-staff-candidate")
        self.assertEqual(match.alignment_method, "cross_staff_repair")
        self.assertEqual(repaired_group.staff_index, 0)
        self.assertEqual(repaired_group.staff_position, -5)
        self.assertEqual(repaired_group.moment_id, "moment-1-2")
        self.assertEqual(repaired_group.visual_status, "fallback")
        self.assertIn("cross_staff_link_repaired", repaired_group.repair_actions)
        self.assertNotIn("cross-staff-candidate", builder.unmatched_visual_group_ids)

        builder.record_musicxml_note(
            "homr-note-1",
            misplaced_symbol,
            part=1,
            measure=1,
            musicxml_staff_number=2,
            voice=1,
        )
        self.assertEqual(repaired_group.musicxml_id, "homr-note-1")

    def test_ambiguous_cross_staff_candidates_remain_diagnostic(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(140, 180),
            autocrop_box=(0, 0, 140, 180),
            cropped_size=(140, 180),
            resized_size=(140, 180),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 180),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = ellipse_contour((x, y), (7, 5), -20, 5)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        notes = [
            make_note(20, 140, "previous-lower"),
            make_note(58, 70, "candidate-a"),
            make_note(62, 70, "candidate-b"),
            make_note(100, 140, "following-lower"),
        ]
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        for visual_id in ("previous-lower", "following-lower"):
            builder.visual_groups[visual_id].staff_index = 1
        for visual_id in ("candidate-a", "candidate-b"):
            builder.visual_groups[visual_id].staff_position = -5

        misplaced_symbol = EncodedSymbol(
            "note_8", "F3", lift="#", position="lower", coordinates=(60, 140)
        )
        builder.add_staff_matches(
            [
                EncodedSymbol("clef_G2", position="upper"),
                EncodedSymbol("clef_F4", position="lower"),
                EncodedSymbol("note_8", "D2", position="lower", coordinates=(20, 140)),
                misplaced_symbol,
                EncodedSymbol("note_8", "D2", position="lower", coordinates=(100, 140)),
            ],
            0,
        )

        self.assertIsNone(builder.matches_by_symbol_id[misplaced_symbol.visual_match_id].visual_id)
        self.assertEqual(
            builder.unmatched_visual_group_ids.intersection({"candidate-a", "candidate-b"}),
            {"candidate-a", "candidate-b"},
        )
