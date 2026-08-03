import unittest

import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import Note, Staff, StaffPoint
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.note_detection import NoteheadWithStem
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder, sounding_pitch
from tests.visual_sidecar_helpers import musicxml_note_ids


class TestVisualSidecarBuilderCore(unittest.TestCase):
    def test_sidecar_pitch_includes_resolved_accidentals(self) -> None:
        self.assertEqual(
            sounding_pitch(EncodedSymbol("note_4", "A3", "b", "_", "_", "upper")),
            "Ab3",
        )
        self.assertEqual(
            sounding_pitch(EncodedSymbol("note_4", "G3", "#", "_", "_", "upper")),
            "G#3",
        )
        self.assertEqual(
            sounding_pitch(EncodedSymbol("note_4", "C4", "N", "_", "_", "upper")),
            "C4",
        )

    def test_exports_distinct_stave_indices_for_a_grand_staff(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 160),
            autocrop_box=(0, 0, 100, 160),
            cropped_size=(100, 160),
            resized_size=(100, 160),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 160),
        )
        builder = VisualSidecarBuilder(coordinate_transform)

        def make_note(y: int, visual_id: str) -> Note:
            return Note(
                BoundingEllipse(
                    ((20, y), (8, 6), 0),
                    np.array([[16, y - 3], [24, y + 3]]),
                    1,
                ),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        upper_note = make_note(30, "vnote-upper")
        lower_note = make_note(120, "vnote-lower")
        upper_staff = Staff([StaffPoint(20, [10, 20, 30, 40, 50], 0)])
        lower_staff = Staff([StaffPoint(20, [100, 110, 120, 130, 140], 0)])
        upper_staff.add_symbol(upper_note)
        lower_staff.add_symbol(lower_note)
        grand_staff = upper_staff.merge(lower_staff)

        builder.prepare_recovery_notes([grand_staff])
        builder.add_staff_visual_notes(
            0,
            [upper_note, lower_note],
            [upper_note.copy(), lower_note.copy()],
        )
        groups = {
            group["visual_group_id"]: group for group in builder.to_json_dict()["visual_groups"]
        }

        self.assertEqual(groups["vnote-upper"]["stave_index"], 0)
        self.assertEqual(groups["vnote-lower"]["stave_index"], 1)

    def test_recovers_real_fifth_ledger_line_candidate_for_sidecar_only(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(200, 200),
            autocrop_box=(0, 0, 200, 200),
            cropped_size=(200, 200),
            resized_size=(200, 200),
            resize_scale=(1.0, 1.0),
            prediction_size=(200, 200),
        )
        staff = Staff(
            [
                StaffPoint(0, [100, 110, 120, 130, 140], 0),
                StaffPoint(100, [100, 110, 120, 130, 140], 0),
            ]
        )
        contour = cv2.ellipse2Poly((50, 50), (4, 3), 0, 0, 360, 10).reshape(-1, 1, 2)
        notehead = BoundingEllipse(((50, 50), (8, 6), 0), contour, 1)
        candidate = NoteheadWithStem(notehead, None)
        existing_notehead = BoundingEllipse(
            ((50, 100), (8, 6), 0), np.array([[46, 97], [54, 103]]), 2
        )
        existing_note = Note(existing_notehead, 9, None, None, "vnote-existing")
        staff.add_symbol(existing_note)
        builder = VisualSidecarBuilder(coordinate_transform, notehead_candidates=[candidate])

        builder.prepare_recovery_notes([staff])
        recovered = builder.recovery_notes_for_staff(staff)

        self.assertEqual(len(recovered), 1)
        self.assertIs(recovered[0].box, notehead)
        self.assertEqual(recovered[0].center, (50, 50))
        self.assertEqual(staff.symbols, [existing_note])

    def test_discards_visual_group_at_recognized_clef_position(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(200, 100),
            autocrop_box=(0, 0, 200, 100),
            cropped_size=(200, 100),
            resized_size=(200, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(200, 100),
        )

        def note(x: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((x, 50), (5, 4), 0, 0, 360, 10).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, 50), (10, 8), 0), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        clef_fragment = note(100, "clef-fragment")
        real_note = note(150, "real-note")
        transformed_clef_fragment = clef_fragment.copy()
        transformed_clef_fragment.center = (102, 51)
        transformed_real_note = real_note.copy()
        transformed_real_note.center = (150, 50)
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(
            0,
            [clef_fragment, real_note],
            [transformed_clef_fragment, transformed_real_note],
        )
        musicxml_note = EncodedSymbol("note_16", "E3", coordinates=(150, 50))

        builder.add_staff_matches(
            [EncodedSymbol("clef_F4", coordinates=(100, 50)), musicxml_note],
            0,
        )

        sidecar = builder.to_json_dict()
        groups = {group["visual_group_id"]: group for group in sidecar["visual_groups"]}
        self.assertEqual(set(groups), {"clef-fragment", "real-note"})
        self.assertEqual(groups["clef-fragment"]["visual_status"], "diagnostic")
        self.assertIn("clef_artifact", groups["clef-fragment"]["repair_actions"])
        self.assertEqual(groups["real-note"]["visual_status"], "canonical")
        self.assertEqual(sidecar["unmatched_visual_notes"], ["clef-fragment"])
        self.assertEqual(
            builder.matches_by_symbol_id[musicxml_note.visual_match_id].visual_id,
            "real-note",
        )

    def test_prediction_to_source_mapping_accounts_for_crop_and_resize(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(1000, 800),
            autocrop_box=(100, 50, 400, 300),
            cropped_size=(400, 300),
            resized_size=(800, 600),
            resize_scale=(2.0, 2.0),
            prediction_size=(400, 300),
        )

        self.assertEqual(coordinate_transform.prediction_point_to_source((200, 150)), (300, 200))

    def test_musicxml_ids_are_recorded_in_visual_sidecar(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        builder = VisualSidecarBuilder(coordinate_transform)
        original = Note(
            BoundingEllipse(((10, 20), (8, 6), 0), np.array([[6, 17], [14, 23]]), 1),
            position=4,
            stem=RotatedBoundingBox(((14, 15), (2, 20), 0), np.array([[14, 5], [14, 25]])),
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()
        transformed.center = (30, 40)
        builder.add_staff_visual_notes(0, [original], [transformed])

        symbol = EncodedSymbol("note_4", "C4", "_", "_", "_", "upper")
        builder.add_staff_matches([symbol], 0)
        xml = generate_xml(XmlGeneratorArguments(), [[symbol]], "", visual_sidecar=builder)

        xml_ids = musicxml_note_ids(xml)
        visual_sidecar = builder.to_json_dict()
        visual_sidecar_ids = [note["musicxml_id"] for note in visual_sidecar["notes"]]
        linked_ids = visual_sidecar["visual_groups"][0]["musicxml_ids"]

        self.assertEqual(visual_sidecar["version"], 2)
        self.assertEqual(visual_sidecar["notes"][0]["alignment_method"], "structural")
        self.assertEqual(visual_sidecar["visual_groups"][0]["visual_status"], "canonical")
        self.assertEqual(xml_ids, visual_sidecar_ids)
        self.assertEqual(xml_ids, linked_ids)
        self.assertEqual(visual_sidecar["unmatched_musicxml_notes"], [])
        self.assertEqual(visual_sidecar["unmatched_visual_notes"], [])

    def test_musicxml_ids_survive_tuplet_cleanup_copy(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        builder = VisualSidecarBuilder(coordinate_transform)
        original = Note(
            BoundingEllipse(((10, 20), (8, 6), 0), np.array([[6, 17], [14, 23]]), 1),
            position=4,
            stem=RotatedBoundingBox(((14, 15), (2, 20), 0), np.array([[14, 5], [14, 25]])),
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()
        builder.add_staff_visual_notes(0, [original], [transformed])

        matched_symbol = EncodedSymbol("note_6", "C4", "_", "_", "_", "upper")
        builder.add_staff_matches([matched_symbol], 0)
        cleaned_symbol = matched_symbol.remove_tuplet()
        self.assertIsNot(cleaned_symbol, matched_symbol)
        self.assertEqual(cleaned_symbol.rhythm, "note_4")

        xml = generate_xml(XmlGeneratorArguments(), [[cleaned_symbol]], "", visual_sidecar=builder)
        xml_ids = musicxml_note_ids(xml)
        visual_sidecar = builder.to_json_dict()

        self.assertEqual(xml_ids, visual_sidecar["visual_groups"][0]["musicxml_ids"])
        self.assertEqual(visual_sidecar["unmatched_musicxml_notes"], [])
        self.assertEqual(visual_sidecar["unmatched_visual_notes"], [])

    def test_pitched_rest_rhythm_is_linked_as_the_musicxml_note_it_generates(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        builder = VisualSidecarBuilder(coordinate_transform)
        original = Note(
            BoundingEllipse(((10, 20), (8, 6), 0), np.array([[6, 17], [14, 23]]), 1),
            position=4,
            stem=RotatedBoundingBox(((14, 15), (2, 20), 0), np.array([[14, 5], [14, 25]])),
            stem_direction=None,
            visual_id="vnote-1",
        )
        builder.add_staff_visual_notes(0, [original], [original.copy()])

        pitched_rest = EncodedSymbol("rest_8", "C4", "_", "_", "_", "lower")
        builder.add_staff_matches([pitched_rest], 0)
        xml = generate_xml(XmlGeneratorArguments(), [[pitched_rest]], "", visual_sidecar=builder)
        xml_ids = musicxml_note_ids(xml)
        visual_sidecar = builder.to_json_dict()

        self.assertEqual(xml_ids, visual_sidecar["visual_groups"][0]["musicxml_ids"])
        self.assertEqual(visual_sidecar["notes"][0]["pitch"], "C4")
        self.assertEqual(visual_sidecar["unmatched_musicxml_notes"], [])
        self.assertEqual(visual_sidecar["unmatched_visual_notes"], [])
