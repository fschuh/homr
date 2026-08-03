import unittest

import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import Note, Staff, StaffPoint
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder


class TestVisualSidecarBuilderSequenceRepair(unittest.TestCase):
    def test_token_order_repairs_shared_stem_subset_without_attention(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(140, 180),
            autocrop_box=(0, 0, 140, 180),
            cropped_size=(140, 180),
            resized_size=(140, 180),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 180),
        )
        shared_stem_contour = np.array(
            [[[53, 25]], [[55, 25]], [[55, 75]], [[53, 75]]],
            dtype=np.float32,
        )
        shared_stem = RotatedBoundingBox(cv2.minAreaRect(shared_stem_contour), shared_stem_contour)

        def make_note(
            x: int,
            y: int,
            visual_id: str,
            note_stem: RotatedBoundingBox | None = None,
        ) -> Note:
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=note_stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        upper_top = make_note(60, 30, "upper-top", shared_stem)
        surplus_upper_bottom = make_note(60, 55, "surplus-upper-bottom", shared_stem)
        lower_top = make_note(60, 115, "lower-top")
        lower_bottom = make_note(60, 140, "lower-bottom")
        notes = [upper_top, surplus_upper_bottom, lower_top, lower_bottom]
        builder = VisualSidecarBuilder(coordinate_transform, stem_fragments=[shared_stem])
        builder.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        builder.visual_groups["lower-top"].stave_index = 1
        builder.visual_groups["lower-bottom"].stave_index = 1

        upper_symbol = EncodedSymbol("note_16", "Eb6", position="upper")
        lower_top_symbol = EncodedSymbol("note_8", "Db4", position="lower")
        lower_bottom_symbol = EncodedSymbol("note_8", "Db3", position="lower")
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
            upper_symbol.visual_match_id: "upper-top",
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
        self.assertEqual(
            builder.unmatched_visual_notes,
            {"surplus-upper-bottom"},
        )
        self.assertEqual(
            builder.visual_groups["surplus-upper-bottom"].visual_status,
            "diagnostic",
        )

    def test_unique_shared_stem_subset_excludes_unrelated_chord_candidate(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(140, 180),
            autocrop_box=(0, 0, 140, 180),
            cropped_size=(140, 180),
            resized_size=(140, 180),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 180),
        )

        def make_note(
            x: int,
            y: int,
            visual_id: str,
        ) -> Note:
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        upper_anchor = make_note(60, 30, "upper-anchor")
        surplus_lower = make_note(54, 150, "surplus-lower")
        lower_top = make_note(66, 105, "lower-top")
        lower_middle = make_note(60, 120, "lower-middle")
        lower_bottom = make_note(60, 135, "lower-bottom")
        notes = [
            upper_anchor,
            surplus_lower,
            lower_top,
            lower_middle,
            lower_bottom,
        ]
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        for visual_id in (
            "surplus-lower",
            "lower-top",
            "lower-middle",
            "lower-bottom",
        ):
            builder.visual_groups[visual_id].stave_index = 1
        for visual_id in ("lower-top", "lower-middle", "lower-bottom"):
            builder.visual_groups[visual_id].owned_stem_component_ids = ["shared-lower-stem"]

        upper_symbol = EncodedSymbol("note_32", "D5", position="upper")
        lower_top_symbol = EncodedSymbol("note_16", "G4", position="lower")
        lower_middle_symbol = EncodedSymbol("note_16", "F4", position="lower")
        lower_bottom_symbol = EncodedSymbol("note_16", "D4", position="lower")
        builder.add_staff_matches(
            [
                upper_symbol,
                EncodedSymbol("chord"),
                lower_top_symbol,
                EncodedSymbol("chord"),
                lower_middle_symbol,
                EncodedSymbol("chord"),
                lower_bottom_symbol,
            ],
            0,
        )

        expected_matches = {
            upper_symbol.visual_match_id: "upper-anchor",
            lower_top_symbol.visual_match_id: "lower-top",
            lower_middle_symbol.visual_match_id: "lower-middle",
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
        self.assertEqual(builder.unmatched_visual_notes, {"surplus-lower"})
        self.assertEqual(
            builder.visual_groups["surplus-lower"].visual_status,
            "diagnostic",
        )

    def test_extra_visual_moment_does_not_shift_later_structural_matches(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(160, 120),
            autocrop_box=(0, 0, 160, 120),
            cropped_size=(160, 120),
            resized_size=(160, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(160, 120),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        first = make_note(20, 35, "first")
        surplus = make_note(50, 90, "surplus")
        second = make_note(80, 35, "second")
        third = make_note(120, 35, "third")
        notes = [first, surplus, second, third]
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        builder.visual_groups["surplus"].stave_index = 1
        first_symbol = EncodedSymbol("note_16", "C5", position="upper", coordinates=(80, 35))
        second_symbol = EncodedSymbol("note_16", "D5", position="upper", coordinates=(20, 35))
        third_symbol = EncodedSymbol("note_16", "E5", position="upper", coordinates=(50, 90))

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
        self.assertEqual(builder.unmatched_visual_notes, {"surplus"})

    def test_unpitched_note_reserves_its_visual_moment_before_final_chord(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(180, 140),
            autocrop_box=(0, 0, 180, 140),
            cropped_size=(180, 140),
            resized_size=(180, 140),
            resize_scale=(1.0, 1.0),
            prediction_size=(180, 140),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        first = make_note(20, 35, "first")
        placeholder = make_note(70, 30, "placeholder")
        chord_top = make_note(130, 30, "chord-top")
        chord_middle = make_note(130, 45, "chord-middle")
        chord_bottom = make_note(130, 65, "chord-bottom")
        notes = [first, placeholder, chord_top, chord_middle, chord_bottom]
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        first_symbol = EncodedSymbol("note_32", "C5", position="upper", coordinates=(130, 45))
        unknown_symbol = EncodedSymbol("note_32", ".", position="upper", coordinates=(70, 30))
        chord_top_symbol = EncodedSymbol("note_2", "A5", position="upper", coordinates=(20, 35))
        chord_middle_symbol = EncodedSymbol("note_2", "F5", position="upper", coordinates=(130, 65))
        chord_bottom_symbol = EncodedSymbol("note_2", "A4", position="upper", coordinates=(130, 30))

        builder.add_staff_matches(
            [
                first_symbol,
                unknown_symbol,
                chord_top_symbol,
                EncodedSymbol("chord"),
                chord_middle_symbol,
                EncodedSymbol("chord"),
                chord_bottom_symbol,
            ],
            0,
        )

        self.assertEqual(
            builder.matches_by_symbol_id[first_symbol.visual_match_id].visual_id,
            "first",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[chord_top_symbol.visual_match_id].visual_id,
            "chord-top",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[chord_middle_symbol.visual_match_id].visual_id,
            "chord-middle",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[chord_bottom_symbol.visual_match_id].visual_id,
            "chord-bottom",
        )
        self.assertNotIn(unknown_symbol.visual_match_id, builder.matches_by_symbol_id)
        self.assertEqual(builder.unmatched_visual_notes, {"placeholder"})

    def test_duplicate_predicted_pitch_does_not_steal_the_next_visual_moment(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 100),
            autocrop_box=(0, 0, 120, 100),
            cropped_size=(120, 100),
            resized_size=(120, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 100),
        )

        def make_note(x: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((x, 40), (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, 40), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        first = make_note(20, "first")
        second = make_note(80, "second")
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(
            0,
            [first, second],
            [first.copy(), second.copy()],
        )
        retained_symbol = EncodedSymbol("note_32", "C5", position="upper", coordinates=(80, 40))
        duplicate_symbol = EncodedSymbol("note_32", "C5", position="upper", coordinates=(20, 40))
        following_symbol = EncodedSymbol("note_32", "D5", position="upper", coordinates=(20, 40))

        builder.add_staff_matches(
            [
                retained_symbol,
                EncodedSymbol("chord"),
                duplicate_symbol,
                following_symbol,
            ],
            0,
        )

        self.assertEqual(
            builder.matches_by_symbol_id[retained_symbol.visual_match_id].visual_id,
            "first",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[following_symbol.visual_match_id].visual_id,
            "second",
        )
        self.assertNotIn(duplicate_symbol.visual_match_id, builder.matches_by_symbol_id)

    def test_split_chord_outlier_is_released_for_pixel_backed_recovery(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(180, 140),
            autocrop_box=(0, 0, 180, 140),
            cropped_size=(180, 140),
            resized_size=(180, 140),
            resize_scale=(1.0, 1.0),
            prediction_size=(180, 140),
        )
        image = np.full((140, 180), 255, dtype=np.uint8)
        cv2.ellipse(image, (100, 25), (7, 5), -20, 0, 360, 0, -1)
        cv2.ellipse(image, (100, 50), (7, 5), -20, 0, 360, 0, -1)
        staff = Staff(
            [
                StaffPoint(0, [20, 30, 40, 50, 60], 0),
                StaffPoint(180, [20, 30, 40, 50, 60], 0),
            ]
        )

        def make_note(x: int, y: int, visual_id: str, position: int) -> Note:
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        stray = make_note(20, 50, "stray", 3)
        previous = make_note(60, 40, "previous", 5)
        chord_top = make_note(100, 25, "chord-top", 8)
        following = make_note(140, 40, "following", 5)
        notes = [stray, previous, chord_top, following]
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        builder.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        builder.visual_groups["previous"].stave_index = 1
        previous_symbol = EncodedSymbol("note_16", "C3", position="lower", coordinates=(60, 40))
        chord_top_symbol = EncodedSymbol("note_2", "E5", position="upper", coordinates=(100, 25))
        chord_bottom_symbol = EncodedSymbol("note_2", "G4", position="upper", coordinates=(20, 50))
        following_symbol = EncodedSymbol("note_16", "D5", position="upper", coordinates=(140, 40))

        builder.add_staff_matches(
            [
                previous_symbol,
                chord_top_symbol,
                EncodedSymbol("chord"),
                chord_bottom_symbol,
                following_symbol,
            ],
            0,
            source_staff=staff,
        )

        recovered_id = builder.matches_by_symbol_id[chord_bottom_symbol.visual_match_id].visual_id
        self.assertIsNotNone(recovered_id)
        self.assertTrue(str(recovered_id).startswith("vnote-transformer-recovered-"))
        recovered = builder.visual_groups[str(recovered_id)]
        self.assertAlmostEqual(recovered.prediction_center[0], 100, delta=2)
        self.assertAlmostEqual(recovered.prediction_center[1], 50, delta=2)
        self.assertEqual(
            recovered.chord_id,
            builder.visual_groups["chord-top"].chord_id,
        )
        self.assertIsNotNone(recovered.chord_id)
        self.assertEqual(recovered.visual_status, "fallback")
        self.assertIn("transformer_chord_recovered", recovered.repair_actions)
        self.assertEqual(builder.unmatched_visual_notes, {"stray"})
