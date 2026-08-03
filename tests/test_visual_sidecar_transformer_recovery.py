import unittest

import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse
from homr.model import Note, Staff, StaffPoint
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder


class TestVisualSidecarBuilderTransformerRecovery(unittest.TestCase):
    def test_recovers_unique_displaced_chord_candidate_after_attention_misses_it(
        self,
    ) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(140, 110),
            autocrop_box=(0, 0, 140, 110),
            cropped_size=(140, 110),
            resized_size=(140, 110),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 110),
        )
        image = np.full((110, 140), 255, dtype=np.uint8)
        for center in ((60, 50), (78, 45), (115, 80)):
            cv2.ellipse(image, center, (7, 5), -20, 0, 360, 0, -1)
        staff = Staff(
            [
                StaffPoint(0, [30, 40, 50, 60, 70], 0),
                StaffPoint(140, [30, 40, 50, 60, 70], 0),
            ]
        )

        def make_note(center: tuple[int, int], position: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly(center, (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse((center, (14, 10), -20), contour),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        anchor = make_note((60, 50), 5, "anchor")
        displaced = make_note((78, 45), 6, "displaced")
        stray = make_note((115, 80), 1, "stray")
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        builder.add_staff_visual_notes(
            0,
            [anchor, displaced, stray],
            [anchor.copy(), displaced.copy(), stray.copy()],
        )
        anchor_symbol = EncodedSymbol("note_4", "G4", position="upper", coordinates=(60, 50))
        displaced_symbol = EncodedSymbol("note_4", "A4", position="upper", coordinates=(5, 5))

        builder.add_staff_matches(
            [anchor_symbol, EncodedSymbol("chord"), displaced_symbol],
            0,
            source_staff=staff,
        )

        match = builder.matches_by_symbol_id[displaced_symbol.visual_match_id]
        self.assertEqual(match.visual_id, "displaced")
        self.assertEqual(match.alignment_method, "sequence_repair")
        self.assertEqual(builder.visual_groups["displaced"].visual_status, "fallback")
        self.assertIn(
            "transformer_chord_candidate_recovered",
            builder.visual_groups["displaced"].repair_actions,
        )
        self.assertEqual(builder.unmatched_visual_notes, {"stray"})

    def test_incomplete_lower_chord_uses_interval_anchors_to_recover_middle_head(
        self,
    ) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(160, 170),
            autocrop_box=(0, 0, 160, 170),
            cropped_size=(160, 170),
            resized_size=(160, 170),
            resize_scale=(1.0, 1.0),
            prediction_size=(160, 170),
        )
        image = np.full((170, 160), 255, dtype=np.uint8)
        for center in ((80, 30), (80, 100), (80, 110), (80, 130)):
            cv2.ellipse(image, center, (8, 5), -20, 0, 360, 0, -1)
        source_staff = Staff(
            [
                StaffPoint(0, [80, 90, 100, 110, 120], 0),
                StaffPoint(160, [80, 90, 100, 110, 120], 0),
            ]
        )

        def note(
            visual_id: str,
            center: tuple[int, int],
            position: int,
        ) -> Note:
            contour = cv2.ellipse2Poly(center, (8, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse((center, (16, 10), -20), contour),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        upper = note("upper-g5", (80, 30), 10)
        upper_duplicate = note("f4-a-upper", (80, 100), -6)
        lower_top = note("f4-z-lower", (80, 100), 14)
        lower_bottom = note("lower-g3", (80, 130), 8)
        notes = [upper, upper_duplicate, lower_top, lower_bottom]
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        builder.add_staff_visual_notes(0, notes, [candidate.copy() for candidate in notes])
        builder.visual_groups["f4-z-lower"].stave_index = 1
        builder.visual_groups["lower-g3"].stave_index = 1

        upper_symbol = EncodedSymbol("note_8", "G5", position="upper", coordinates=(80, 30))
        top_symbol = EncodedSymbol("note_4", "F4", position="lower", coordinates=(80, 100))
        middle_symbol = EncodedSymbol("note_4", "D4", position="lower", coordinates=(80, 110))
        bottom_symbol = EncodedSymbol("note_4", "G3", position="lower", coordinates=(80, 130))
        symbols = [
            upper_symbol,
            EncodedSymbol("chord"),
            top_symbol,
            EncodedSymbol("chord"),
            middle_symbol,
            EncodedSymbol("chord"),
            bottom_symbol,
        ]

        builder.add_staff_matches(symbols, 0, source_staff=source_staff)

        upper_match = builder.matches_by_symbol_id[upper_symbol.visual_match_id]
        top_match = builder.matches_by_symbol_id[top_symbol.visual_match_id]
        middle_match = builder.matches_by_symbol_id[middle_symbol.visual_match_id]
        bottom_match = builder.matches_by_symbol_id[bottom_symbol.visual_match_id]
        self.assertEqual(upper_match.visual_id, "upper-g5")
        self.assertEqual(top_match.visual_id, "f4-a-upper")
        self.assertEqual(bottom_match.visual_id, "lower-g3")
        self.assertTrue(str(middle_match.visual_id).startswith("vnote-transformer-recovered-"))
        groups = [
            builder.visual_groups[str(match.visual_id)]
            for match in (upper_match, top_match, middle_match, bottom_match)
        ]
        self.assertEqual(len({group.moment_id for group in groups}), 1)
        lower_groups = groups[1:]
        self.assertEqual(lower_groups[0].stave_index, 1)
        self.assertEqual(lower_groups[0].staff_position, 14)
        self.assertIn("stave_membership_repaired", lower_groups[0].repair_actions)
        self.assertEqual(len({group.chord_id for group in lower_groups}), 1)
        self.assertIsNotNone(lower_groups[0].chord_id)
        self.assertEqual(builder.unmatched_visual_notes, {"f4-z-lower"})

    def test_recovers_missing_head_at_the_edge_of_a_dense_hollow_chord(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        image = np.full((120, 120), 255, dtype=np.uint8)
        centers = ((60, 30), (60, 46), (60, 62))
        for center in centers:
            cv2.ellipse(image, center, (10, 7), -20, 0, 360, 2)
        staff = Staff(
            [
                StaffPoint(0, [30, 46, 62, 78, 94], 0),
                StaffPoint(120, [30, 46, 62, 78, 94], 0),
            ]
        )

        def make_note(center: tuple[int, int], position: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly(center, (10, 7), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse((center, (20, 14), -20), contour),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        top = make_note(centers[0], 9, "top")
        middle = make_note(centers[1], 7, "middle")
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        builder.add_staff_visual_notes(0, [top, middle], [top.copy(), middle.copy()])
        bottom_symbol = EncodedSymbol("note_1", "F4", position="upper", coordinates=centers[2])
        top_symbol = EncodedSymbol("note_1", "C5", position="upper", coordinates=centers[0])
        middle_symbol = EncodedSymbol("note_1", "A4", position="upper", coordinates=centers[1])

        builder.add_staff_matches(
            [
                bottom_symbol,
                EncodedSymbol("chord"),
                top_symbol,
                EncodedSymbol("chord"),
                middle_symbol,
            ],
            0,
            source_staff=staff,
        )

        bottom_match = builder.matches_by_symbol_id[bottom_symbol.visual_match_id]
        self.assertIsNotNone(bottom_match.visual_id)
        self.assertTrue(str(bottom_match.visual_id).startswith("vnote-transformer-recovered-"))
        recovered = builder.visual_groups[str(bottom_match.visual_id)]
        self.assertAlmostEqual(recovered.prediction_center[0], 60, delta=2)
        self.assertAlmostEqual(recovered.prediction_center[1], 62, delta=2)
        self.assertEqual(recovered.visual_status, "fallback")

    def test_recovers_hollow_notehead_positioned_by_transformer(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 150),
            autocrop_box=(0, 0, 120, 150),
            cropped_size=(120, 150),
            resized_size=(120, 150),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 150),
        )
        image = np.full((150, 120), 255, dtype=np.uint8)
        cv2.ellipse(image, (60, 25), (7, 5), -20, 0, 360, 0, 2)
        cv2.ellipse(image, (60, 50), (7, 5), -20, 0, 360, 0, -1)
        staff = Staff(
            [
                StaffPoint(0, [20, 30, 40, 50, 60, 80, 90, 100, 110, 120], 0),
                StaffPoint(120, [20, 30, 40, 50, 60, 80, 90, 100, 110, 120], 0),
            ]
        )
        lower_contour = cv2.ellipse2Poly((60, 50), (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
        lower = Note(
            BoundingEllipse(((60, 50), (14, 10), -20), lower_contour),
            position=3,
            stem=None,
            stem_direction=None,
            visual_id="vnote-lower",
        )
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        builder.add_staff_visual_notes(0, [lower], [lower.copy()])
        lower_symbol = EncodedSymbol("note_2.", "G4", position="upper", coordinates=(60, 50))
        # Attention is deliberately offset. Chord pitch and the matched lower head
        # provide the exact same-stave position in the grand-staff source grid.
        upper_symbol = EncodedSymbol("note_2.", "E5", position="upper", coordinates=(68, 10))

        builder.add_staff_matches(
            [upper_symbol, EncodedSymbol("chord"), lower_symbol],
            0,
            source_staff=staff,
        )

        upper_match = builder.matches_by_symbol_id[id(upper_symbol)]
        self.assertIsNotNone(upper_match.visual_id)
        self.assertTrue(str(upper_match.visual_id).startswith("vnote-transformer-recovered-"))
        recovered = builder.visual_groups[str(upper_match.visual_id)]
        self.assertAlmostEqual(recovered.notehead_ellipses[0]["center"][0], 60, delta=2)
        self.assertAlmostEqual(recovered.notehead_ellipses[0]["center"][1], 25, delta=2)
        self.assertEqual(recovered.staff_position, 8)
        self.assertEqual(recovered.visual_status, "fallback")
        self.assertEqual(recovered.provenance, "transformer_recovered")
        self.assertEqual(recovered.chord_id, builder.visual_groups["vnote-lower"].chord_id)
        self.assertIsNotNone(recovered.chord_id)
        self.assertIn("transformer_chord_recovered", recovered.repair_actions)

    def test_transformer_recovered_ledger_head_inherits_its_chord_mates_stave(
        self,
    ) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 160),
            autocrop_box=(0, 0, 120, 160),
            cropped_size=(120, 160),
            resized_size=(120, 160),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 160),
        )
        image = np.full((160, 120), 255, dtype=np.uint8)
        for center in ((60, 30), (60, 70), (60, 110)):
            cv2.ellipse(image, center, (7, 5), -20, 0, 360, 0, -1)
        staff = Staff(
            [
                StaffPoint(0, [20, 30, 40, 50, 60, 100, 110, 120, 130, 140], 0),
                StaffPoint(120, [20, 30, 40, 50, 60, 100, 110, 120, 130, 140], 0),
            ]
        )

        def make_note(y: int, visual_id: str, position: int) -> Note:
            contour = cv2.ellipse2Poly((60, y), (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((60, y), (14, 10), -20), contour),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        treble = make_note(30, "treble", 3)
        bass_bottom = make_note(110, "bass-bottom", 3)
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        builder.add_staff_visual_notes(
            0,
            [treble, bass_bottom],
            [treble.copy(), bass_bottom.copy()],
        )
        builder.visual_groups["treble"].stave_index = 0
        builder.visual_groups["bass-bottom"].stave_index = 1
        treble_symbol = EncodedSymbol("note_2", "C6", position="upper", coordinates=(60, 30))
        bass_top_symbol = EncodedSymbol("note_2", "D5", position="lower", coordinates=(68, 70))
        bass_bottom_symbol = EncodedSymbol("note_2", "C4", position="lower", coordinates=(60, 110))

        builder.add_staff_matches(
            [
                treble_symbol,
                EncodedSymbol("chord"),
                bass_top_symbol,
                EncodedSymbol("chord"),
                bass_bottom_symbol,
            ],
            0,
            source_staff=staff,
        )

        recovered_id = builder.matches_by_symbol_id[bass_top_symbol.visual_match_id].visual_id
        self.assertIsNotNone(recovered_id)
        recovered = builder.visual_groups[str(recovered_id)]
        self.assertEqual(recovered.stave_index, 1)
        self.assertEqual(
            recovered.chord_id,
            builder.visual_groups["bass-bottom"].chord_id,
        )
        self.assertIsNotNone(recovered.chord_id)
        self.assertNotEqual(
            recovered.chord_id,
            builder.visual_groups["treble"].chord_id,
        )
        self.assertEqual(recovered.visual_status, "fallback")

    def test_does_not_recover_transformer_note_without_notehead_ink(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        image = np.full((120, 120), 255, dtype=np.uint8)
        staff = Staff(
            [
                StaffPoint(0, [40, 50, 60, 70, 80], 0),
                StaffPoint(120, [40, 50, 60, 70, 80], 0),
            ]
        )
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        symbol = EncodedSymbol("note_2.", "E5", coordinates=(60, 35))

        builder.add_staff_matches([symbol], 0, source_staff=staff)

        self.assertIsNone(builder.matches_by_symbol_id[id(symbol)].visual_id)
        self.assertEqual(builder.visual_groups, {})
