import unittest

import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import Note, Staff, StaffPoint
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.note_detection import NoteheadWithStem
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar import PreprocessingMetadata, VisualSidecar, sounding_pitch


class TestVisualSidecar(unittest.TestCase):
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
        metadata = PreprocessingMetadata(
            source_image_size=(100, 160),
            autocrop_box=(0, 0, 100, 160),
            cropped_size=(100, 160),
            resized_size=(100, 160),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 160),
        )
        collector = VisualSidecar(metadata)

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

        collector.prepare_recovery_notes([grand_staff])
        collector.add_staff_visual_notes(
            0,
            [upper_note, lower_note],
            [upper_note.copy(), lower_note.copy()],
        )
        groups = {
            group["visual_group_id"]: group
            for group in collector.to_json_dict()["visual_groups"]
        }

        self.assertEqual(groups["vnote-upper"]["stave_index"], 0)
        self.assertEqual(groups["vnote-lower"]["stave_index"], 1)

    def test_recovers_real_fifth_ledger_line_candidate_for_sidecar_only(self) -> None:
        metadata = PreprocessingMetadata(
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
        collector = VisualSidecar(metadata, notehead_candidates=[candidate])

        collector.prepare_recovery_notes([staff])
        recovered = collector.recovery_notes_for_staff(staff)

        self.assertEqual(len(recovered), 1)
        self.assertIs(recovered[0].box, notehead)
        self.assertEqual(recovered[0].center, (50, 50))
        self.assertEqual(staff.symbols, [existing_note])

    def test_discards_visual_group_at_recognized_clef_position(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(200, 100),
            autocrop_box=(0, 0, 200, 100),
            cropped_size=(200, 100),
            resized_size=(200, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(200, 100),
        )

        def note(x: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly(
                (x, 50), (5, 4), 0, 0, 360, 10
            ).reshape(-1, 1, 2)
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
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
            0,
            [clef_fragment, real_note],
            [transformed_clef_fragment, transformed_real_note],
        )
        musicxml_note = EncodedSymbol(
            "note_16", "E3", coordinates=(150, 50)
        )

        collector.add_staff_matches(
            [EncodedSymbol("clef_F4", coordinates=(100, 50)), musicxml_note],
            0,
        )

        sidecar = collector.to_json_dict()
        groups = {
            group["visual_group_id"]: group for group in sidecar["visual_groups"]
        }
        self.assertEqual(set(groups), {"clef-fragment", "real-note"})
        self.assertEqual(groups["clef-fragment"]["visual_status"], "diagnostic")
        self.assertIn("clef_artifact", groups["clef-fragment"]["repair_actions"])
        self.assertEqual(groups["real-note"]["visual_status"], "canonical")
        self.assertEqual(sidecar["unmatched_visual_notes"], ["clef-fragment"])
        self.assertEqual(
            collector.matches_by_symbol_id[musicxml_note.visual_match_id].visual_id,
            "real-note",
        )

    def test_prediction_to_source_mapping_accounts_for_crop_and_resize(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(1000, 800),
            autocrop_box=(100, 50, 400, 300),
            cropped_size=(400, 300),
            resized_size=(800, 600),
            resize_scale=(2.0, 2.0),
            prediction_size=(400, 300),
        )

        self.assertEqual(metadata.prediction_point_to_source((200, 150)), (300, 200))

    def test_musicxml_ids_are_recorded_in_visual_sidecar(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        collector = VisualSidecar(metadata)
        original = Note(
            BoundingEllipse(((10, 20), (8, 6), 0), np.array([[6, 17], [14, 23]]), 1),
            position=4,
            stem=RotatedBoundingBox(((14, 15), (2, 20), 0), np.array([[14, 5], [14, 25]])),
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()
        transformed.center = (30, 40)
        collector.add_staff_visual_notes(0, [original], [transformed])

        symbol = EncodedSymbol("note_4", "C4", "_", "_", "_", "upper")
        collector.add_staff_matches([symbol], 0)
        xml = generate_xml(XmlGeneratorArguments(), [[symbol]], "", visual_sidecar=collector)

        xml_ids = self._musicxml_note_ids(xml)
        visual_sidecar = collector.to_json_dict()
        visual_sidecar_ids = [note["musicxml_id"] for note in visual_sidecar["notes"]]
        linked_ids = visual_sidecar["visual_groups"][0]["musicxml_ids"]

        self.assertEqual(visual_sidecar["version"], 2)
        self.assertEqual(
            visual_sidecar["notes"][0]["alignment_method"], "structural"
        )
        self.assertEqual(
            visual_sidecar["visual_groups"][0]["visual_status"], "canonical"
        )
        self.assertEqual(xml_ids, visual_sidecar_ids)
        self.assertEqual(xml_ids, linked_ids)
        self.assertEqual(visual_sidecar["unmatched_musicxml_notes"], [])
        self.assertEqual(visual_sidecar["unmatched_visual_notes"], [])

    def test_musicxml_ids_survive_tuplet_cleanup_copy(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        collector = VisualSidecar(metadata)
        original = Note(
            BoundingEllipse(((10, 20), (8, 6), 0), np.array([[6, 17], [14, 23]]), 1),
            position=4,
            stem=RotatedBoundingBox(((14, 15), (2, 20), 0), np.array([[14, 5], [14, 25]])),
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()
        collector.add_staff_visual_notes(0, [original], [transformed])

        matched_symbol = EncodedSymbol("note_6", "C4", "_", "_", "_", "upper")
        collector.add_staff_matches([matched_symbol], 0)
        cleaned_symbol = matched_symbol.remove_tuplet()
        self.assertIsNot(cleaned_symbol, matched_symbol)
        self.assertEqual(cleaned_symbol.rhythm, "note_4")

        xml = generate_xml(
            XmlGeneratorArguments(), [[cleaned_symbol]], "", visual_sidecar=collector
        )
        xml_ids = self._musicxml_note_ids(xml)
        visual_sidecar = collector.to_json_dict()

        self.assertEqual(xml_ids, visual_sidecar["visual_groups"][0]["musicxml_ids"])
        self.assertEqual(visual_sidecar["unmatched_musicxml_notes"], [])
        self.assertEqual(visual_sidecar["unmatched_visual_notes"], [])

    def test_pitched_rest_rhythm_is_linked_as_the_musicxml_note_it_generates(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        collector = VisualSidecar(metadata)
        original = Note(
            BoundingEllipse(((10, 20), (8, 6), 0), np.array([[6, 17], [14, 23]]), 1),
            position=4,
            stem=RotatedBoundingBox(((14, 15), (2, 20), 0), np.array([[14, 5], [14, 25]])),
            stem_direction=None,
            visual_id="vnote-1",
        )
        collector.add_staff_visual_notes(0, [original], [original.copy()])

        pitched_rest = EncodedSymbol("rest_8", "C4", "_", "_", "_", "lower")
        collector.add_staff_matches([pitched_rest], 0)
        xml = generate_xml(
            XmlGeneratorArguments(), [[pitched_rest]], "", visual_sidecar=collector
        )
        xml_ids = self._musicxml_note_ids(xml)
        visual_sidecar = collector.to_json_dict()

        self.assertEqual(xml_ids, visual_sidecar["visual_groups"][0]["musicxml_ids"])
        self.assertEqual(visual_sidecar["notes"][0]["pitch"], "C4")
        self.assertEqual(visual_sidecar["unmatched_musicxml_notes"], [])
        self.assertEqual(visual_sidecar["unmatched_visual_notes"], [])

    def test_shared_stem_components_are_exported_as_chord_identity(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        shared_stem_contour = np.array(
            [[[19, 5]], [[21, 5]], [[21, 45]], [[19, 45]]], dtype=np.float32
        )
        separate_stem_contour = np.array(
            [[[69, 5]], [[71, 5]], [[71, 25]], [[69, 25]]], dtype=np.float32
        )
        shared_stem = RotatedBoundingBox(
            cv2.minAreaRect(shared_stem_contour), shared_stem_contour
        )
        separate_stem = RotatedBoundingBox(
            cv2.minAreaRect(separate_stem_contour), separate_stem_contour
        )

        def make_note(x: int, y: int, visual_id: str, stem: RotatedBoundingBox) -> Note:
            contour = cv2.ellipse2Poly((x, y), (5, 4), 0, 0, 360, 10).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, y), (10, 8), 0), contour),
                position=4,
                stem=stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        notes = [
            make_note(15, 15, "vnote-1", shared_stem),
            make_note(15, 30, "vnote-2", shared_stem),
            make_note(15, 42, "vnote-4", shared_stem),
            make_note(65, 15, "vnote-3", separate_stem),
        ]
        collector = VisualSidecar(metadata, stem_fragments=[shared_stem, separate_stem])
        collector.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        symbols = [
            EncodedSymbol("note_4", "C4", coordinates=(15, 15)),
            # A stray augmentation-dot prediction must not split noteheads that
            # share both a physical stem and the same base duration.
            EncodedSymbol("note_4.", "E4", coordinates=(15, 30)),
            EncodedSymbol("note_8", "F4", coordinates=(15, 42)),
            EncodedSymbol("note_4", "G4", coordinates=(65, 15)),
        ]
        collector.add_staff_matches(symbols, 0)

        groups = {
            group["visual_group_id"]: group for group in collector.to_json_dict()["visual_groups"]
        }
        self.assertTrue(groups["vnote-1"]["stem_component_ids"])
        self.assertEqual(
            groups["vnote-1"]["stem_component_ids"],
            groups["vnote-2"]["stem_component_ids"],
        )
        self.assertEqual(groups["vnote-4"]["stem_component_ids"], [])
        self.assertEqual(groups["vnote-3"]["stem_component_ids"], [])

    def test_structural_chord_without_stem_components_gets_physical_identity(
        self,
    ) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )

        def make_note(y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((50, y), (7, 5), -20, 0, 360, 5).reshape(
                -1, 1, 2
            )
            return Note(
                BoundingEllipse(((50, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        top = make_note(25, "top")
        bottom = make_note(55, "bottom")
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
            0,
            [top, bottom],
            [top.copy(), bottom.copy()],
        )
        top_symbol = EncodedSymbol(
            "note_16", "Eb6", position="upper", coordinates=(50, 25)
        )
        bottom_symbol = EncodedSymbol(
            "note_16", "Eb5", position="upper", coordinates=(50, 55)
        )

        collector.add_staff_matches(
            [top_symbol, EncodedSymbol("chord"), bottom_symbol],
            0,
        )

        top_group = collector.visual_groups["top"]
        bottom_group = collector.visual_groups["bottom"]
        self.assertEqual(
            collector.matches_by_symbol_id[top_symbol.visual_match_id].alignment_method,
            "structural",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[bottom_symbol.visual_match_id].alignment_method,
            "structural",
        )
        self.assertEqual(top_group.chord_id, bottom_group.chord_id)
        self.assertIsNotNone(top_group.chord_id)
        self.assertIn("structural_chord_proven", top_group.repair_actions)
        self.assertIn("structural_chord_proven", bottom_group.repair_actions)
        self.assertEqual(top_group.visual_status, "canonical")
        self.assertEqual(bottom_group.visual_status, "canonical")

    def test_close_opposed_stem_voices_never_share_physical_chord(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 140),
            autocrop_box=(0, 0, 120, 140),
            cropped_size=(120, 140),
            resized_size=(120, 140),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 140),
        )
        upward_contour = np.array(
            [[[57, 10]], [[59, 10]], [[59, 34]], [[57, 34]]],
            dtype=np.float32,
        )
        downward_contour = np.array(
            [[[41, 60]], [[43, 60]], [[43, 120]], [[41, 120]]],
            dtype=np.float32,
        )
        upward_stem = RotatedBoundingBox(
            cv2.minAreaRect(upward_contour), upward_contour
        )
        downward_stem = RotatedBoundingBox(
            cv2.minAreaRect(downward_contour), downward_contour
        )

        def make_note(y: int, position: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly(
                (50, y), (10, 7), -20, 0, 360, 5
            ).reshape(-1, 1, 2)
            # Reproduce segmentation assigning the visible downward stem to both
            # touching heads even though a separate upward component also exists.
            return Note(
                BoundingEllipse(((50, y), (20, 14), -20), contour),
                position=position,
                stem=downward_stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        for lower_rhythm, expected_action in (
            ("note_16", "mixed_duration_stems_separated"),
            ("note_8", "opposed_stems_separated"),
        ):
            with self.subTest(lower_rhythm=lower_rhythm):
                upper = make_note(40, 12, "upper")
                lower = make_note(54, 10, "lower")
                collector = VisualSidecar(
                    metadata, stem_fragments=[upward_stem, downward_stem]
                )
                collector.add_staff_visual_notes(
                    0,
                    [upper, lower],
                    [upper.copy(), lower.copy()],
                )
                collector.add_staff_matches(
                    [
                        EncodedSymbol("note_8", "Bb5", coordinates=(50, 40)),
                        EncodedSymbol("chord"),
                        EncodedSymbol(
                            lower_rhythm, "Gb5", coordinates=(50, 54)
                        ),
                    ],
                    0,
                )

                upper_group = collector.visual_groups["upper"]
                lower_group = collector.visual_groups["lower"]
                self.assertEqual(upper_group.moment_id, lower_group.moment_id)
                self.assertIsNotNone(upper_group.moment_id)
                self.assertIsNone(upper_group.chord_id)
                self.assertIsNone(lower_group.chord_id)
                self.assertIn(expected_action, upper_group.repair_actions)
                self.assertIn(expected_action, lower_group.repair_actions)

    def test_separate_notes_do_not_share_chord_identity_from_misassigned_stem(
        self,
    ) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        stem_contour = np.array(
            [[[29, 15]], [[31, 15]], [[31, 70]], [[29, 70]]], dtype=np.float32
        )
        shared_stem = RotatedBoundingBox(
            cv2.minAreaRect(stem_contour), stem_contour
        )

        def make_note(
            x: int,
            y: int,
            visual_id: str,
            width: int = 20,
            note_stem: RotatedBoundingBox | None = shared_stem,
        ) -> Note:
            contour = cv2.ellipse2Poly((x, y), (width // 2, 7), 0, 0, 360, 10).reshape(
                -1, 1, 2
            )
            return Note(
                BoundingEllipse(((x, y), (width, 14), 0), contour),
                position=4,
                stem=note_stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        notes = [
            make_note(20, 60, "first"),
            make_note(50, 40, "second"),
            # A wider neighboring candidate reproduces the global ownership
            # search radius present in the full score.
            make_note(90, 80, "padding", width=40, note_stem=None),
        ]
        collector = VisualSidecar(metadata, stem_fragments=[shared_stem])
        collector.add_staff_visual_notes(
            0, notes, [candidate.copy() for candidate in notes]
        )
        collector.add_staff_matches(
            [
                EncodedSymbol("note_16", "D4", coordinates=(20, 60)),
                EncodedSymbol("note_16", "G4", coordinates=(50, 40)),
                EncodedSymbol("note_16", "C4", coordinates=(90, 80)),
            ],
            0,
        )

        self.assertTrue(
            collector.visual_groups["first"].owned_stem_component_ids
        )
        self.assertEqual(
            collector.visual_groups["first"].owned_stem_component_ids,
            collector.visual_groups["second"].owned_stem_component_ids,
        )
        groups = {
            group["visual_group_id"]: group
            for group in collector.to_json_dict()["visual_groups"]
        }
        self.assertEqual(groups["first"]["stem_component_ids"], [])
        self.assertEqual(groups["second"]["stem_component_ids"], [])

    def test_rejoins_horizontally_split_whole_note_chord_heads(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        image = np.full((100, 100), 255, dtype=np.uint8)
        cv2.ellipse(image, ((50, 38), (26, 18), 0), 0, 2)
        cv2.ellipse(image, ((50, 62), (26, 18), 0), 0, 2)
        cv2.line(image, (15, 38), (85, 38), 0, 1)
        cv2.line(image, (15, 62), (85, 62), 0, 1)
        cv2.ellipse(image, ((75, 30), (12, 10), -20), 0, -1)
        cv2.ellipse(image, ((90, 42), (12, 10), -20), 0, -1)
        collector = VisualSidecar(metadata, source_image=image)

        def fragment(
            visual_id: str, center_x: int, center_y: int, position: int
        ) -> Note:
            contour = np.array(
                [
                    [[center_x - 6, center_y - 9]],
                    [[center_x + 6, center_y - 9]],
                    [[center_x + 6, center_y + 9]],
                    [[center_x - 6, center_y + 9]],
                ],
                dtype=np.int32,
            )
            return Note(
                BoundingEllipse(
                    ((center_x, center_y), (12, 18), 0),
                    contour,
                    position,
                ),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        notes = [
            fragment("top-left", 44, 38, 9),
            fragment("bottom-left", 44, 62, 7),
            fragment("top-right", 56, 38, 9),
            fragment("bottom-right", 56, 62, 7),
            fragment("sequence-a", 75, 30, 11),
            fragment("sequence-g", 90, 42, 10),
        ]
        collector.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        lower_whole = EncodedSymbol("note_1", "F3", coordinates=(44, 62))
        upper_whole = EncodedSymbol("note_1", "A3", coordinates=(56, 38))
        sequence_a = EncodedSymbol("note_16", "A5", coordinates=(90, 42))
        sequence_g = EncodedSymbol("note_16", "G5", coordinates=(75, 30))
        collector.add_staff_matches(
            [
                lower_whole,
                EncodedSymbol("chord"),
                upper_whole,
                sequence_a,
                sequence_g,
            ],
            0,
        )

        sidecar = collector.to_json_dict()
        groups = {
            group["visual_group_id"]: group for group in sidecar["visual_groups"]
        }

        self.assertEqual(
            set(groups),
            {
                "bottom-left",
                "top-left",
                "top-right",
                "bottom-right",
                "sequence-a",
                "sequence-g",
            },
        )
        self.assertEqual(
            sidecar["unmatched_visual_notes"], ["bottom-right", "top-right"]
        )
        self.assertEqual(groups["top-right"]["visual_status"], "diagnostic")
        self.assertEqual(groups["bottom-right"]["visual_status"], "diagnostic")
        self.assertEqual(groups["bottom-left"]["provenance"], "merged_fragments")
        self.assertEqual(groups["top-left"]["provenance"], "merged_fragments")
        self.assertAlmostEqual(groups["bottom-left"]["center"][0], 50, delta=0.5)
        self.assertAlmostEqual(groups["top-left"]["center"][0], 50, delta=0.5)
        self.assertGreater(groups["bottom-left"]["notehead_ellipses"][0]["rx"], 10)
        self.assertGreater(groups["top-left"]["notehead_ellipses"][0]["rx"], 10)
        self.assertEqual(
            collector.matches_by_symbol_id[sequence_a.visual_match_id].visual_id,
            "sequence-a",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[sequence_g.visual_match_id].visual_id,
            "sequence-g",
        )

    def test_split_hollow_heads_are_consolidated_before_cross_stave_matching(
        self,
    ) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(160, 190),
            autocrop_box=(0, 0, 160, 190),
            cropped_size=(160, 190),
            resized_size=(160, 190),
            resize_scale=(1.0, 1.0),
            prediction_size=(160, 190),
        )
        image = np.full((190, 160), 255, dtype=np.uint8)
        for center in ((80, 35), (80, 65), (80, 90), (80, 135), (80, 175)):
            cv2.ellipse(image, center, (13, 9), 0, 0, 360, 2)

        def fragment(
            visual_id: str,
            center_x: int,
            center_y: int,
            position: int,
            stem: RotatedBoundingBox | None = None,
        ) -> Note:
            contour = np.array(
                [
                    [[center_x - 6, center_y - 9]],
                    [[center_x + 6, center_y - 9]],
                    [[center_x + 6, center_y + 9]],
                    [[center_x - 6, center_y + 9]],
                ],
                dtype=np.int32,
            )
            return Note(
                BoundingEllipse(((center_x, center_y), (12, 18), 0), contour),
                position=position,
                stem=stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        def whole(visual_id: str, center_y: int, position: int) -> Note:
            contour = cv2.ellipse2Poly(
                (80, center_y), (13, 9), 0, 0, 360, 5
            ).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((80, center_y), (26, 18), 0), contour),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        tiny_stem = RotatedBoundingBox(
            ((94, 135), (2, 6), 0),
            np.array([[94, 132], [94, 138]]),
        )
        notes = [
            fragment("upper-left", 74, 35, 6),
            fragment("upper-right", 86, 35, 6),
            whole("upper-middle", 65, 3),
            whole("upper-bottom", 90, 1),
            fragment("lower-left", 74, 135, 4, tiny_stem),
            fragment("lower-right", 86, 135, 4, tiny_stem),
            whole("lower-bass", 175, -3),
        ]
        collector = VisualSidecar(metadata, source_image=image)
        collector.add_staff_visual_notes(
            0, notes, [note.copy() for note in notes]
        )
        for visual_id in ("lower-left", "lower-right", "lower-bass"):
            collector.visual_groups[visual_id].stave_index = 1

        symbols = [
            EncodedSymbol("note_1", "C5", position="upper", coordinates=(80, 35)),
            EncodedSymbol("chord"),
            EncodedSymbol("note_1", "G4", position="upper", coordinates=(80, 65)),
            EncodedSymbol("chord"),
            EncodedSymbol("note_1", "E4", position="upper", coordinates=(80, 90)),
            EncodedSymbol("chord"),
            EncodedSymbol("note_1", "C3", position="lower", coordinates=(80, 135)),
            EncodedSymbol("chord"),
            EncodedSymbol("note_1", "C2", position="lower", coordinates=(80, 175)),
        ]

        collector.add_staff_matches(symbols, 0)

        note_symbols = [
            symbol for symbol in symbols if symbol.rhythm.startswith("note")
        ]
        matches = [
            collector.matches_by_symbol_id[symbol.visual_match_id]
            for symbol in note_symbols
        ]
        self.assertTrue(all(match.visual_id is not None for match in matches))
        self.assertEqual(
            {match.alignment_method for match in matches}, {"structural"}
        )
        matched_groups = [
            collector.visual_groups[str(match.visual_id)] for match in matches
        ]
        self.assertEqual(len({group.moment_id for group in matched_groups}), 1)
        self.assertEqual(len({group.chord_id for group in matched_groups}), 2)
        self.assertEqual(
            collector.visual_groups["upper-left"].provenance,
            "merged_fragments",
        )
        self.assertEqual(
            collector.visual_groups["lower-left"].provenance,
            "merged_fragments",
        )

    def test_displaced_second_stays_in_stemless_whole_note_chord_moment(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(160, 120),
            autocrop_box=(0, 0, 160, 120),
            cropped_size=(160, 120),
            resized_size=(160, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(160, 120),
        )
        image = np.full((120, 160), 255, dtype=np.uint8)

        def hollow_note(
            visual_id: str, center_x: int, center_y: int, position: int
        ) -> Note:
            contour = cv2.ellipse2Poly(
                (center_x, center_y), (12, 8), 0, 0, 360, 5
            ).reshape(-1, 1, 2)
            cv2.ellipse(image, (center_x, center_y), (12, 8), 0, 0, 360, 2)
            return Note(
                BoundingEllipse(
                    ((center_x, center_y), (24, 16), 0), contour, position
                ),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        top = hollow_note("top", 60, 35, 7)
        middle = hollow_note("middle", 60, 55, 4)
        displaced_bottom = hollow_note("displaced-bottom", 38, 56, 3)
        collector = VisualSidecar(metadata, source_image=image)
        collector.add_staff_visual_notes(
            0,
            [top, middle, displaced_bottom],
            [top.copy(), middle.copy(), displaced_bottom.copy()],
        )
        top_symbol = EncodedSymbol("note_1", "D5", position="upper")
        middle_symbol = EncodedSymbol("note_1", "A4", position="upper")
        bottom_symbol = EncodedSymbol("note_1", "G4", position="upper")

        collector.add_staff_matches(
            [
                top_symbol,
                EncodedSymbol("chord"),
                middle_symbol,
                EncodedSymbol("chord"),
                bottom_symbol,
            ],
            0,
        )

        matched_groups = [
            collector.visual_groups[
                collector.matches_by_symbol_id[symbol.visual_match_id].visual_id
            ]
            for symbol in (top_symbol, middle_symbol, bottom_symbol)
        ]
        self.assertEqual(
            {group.visual_id for group in matched_groups},
            {"top", "middle", "displaced-bottom"},
        )
        self.assertEqual(len({group.moment_id for group in matched_groups}), 1)
        self.assertIsNotNone(matched_groups[0].moment_id)
        self.assertEqual(len({group.chord_id for group in matched_groups}), 1)
        self.assertIsNotNone(matched_groups[0].chord_id)

    def test_displaced_opposing_voice_stays_in_the_shared_visual_moment(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(140, 120),
            autocrop_box=(0, 0, 140, 120),
            cropped_size=(140, 120),
            resized_size=(140, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 120),
        )

        def make_note(x: int, y: int, position: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly(
                (x, y), (10, 7), -20, 0, 360, 5
            ).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, y), (20, 14), -20), contour),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        notes = [
            make_note(60, 30, 8, "chord-top"),
            make_note(60, 50, 5, "chord-middle"),
            make_note(60, 64, 3, "chord-bottom"),
            make_note(40, 72, 2, "displaced-voice"),
            make_note(40, 105, 4, "bass-anchor"),
        ]
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
            0, notes, [note.copy() for note in notes]
        )
        collector.visual_groups["bass-anchor"].stave_index = 1
        symbols = [
            EncodedSymbol("note_2", "D5", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_2", "A4", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_2", "F4", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_8", "E4", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_8", "C3", position="lower"),
        ]

        collector.add_staff_matches(symbols, 0)

        matched_groups = [
            collector.visual_groups[
                collector.matches_by_symbol_id[symbol.visual_match_id].visual_id
            ]
            for symbol in symbols
            if symbol.rhythm.startswith("note")
        ]
        self.assertEqual(
            {group.visual_id for group in matched_groups},
            {
                "chord-top",
                "chord-middle",
                "chord-bottom",
                "displaced-voice",
                "bass-anchor",
            },
        )
        self.assertEqual(len({group.moment_id for group in matched_groups}), 1)
        self.assertIsNotNone(matched_groups[0].moment_id)
        self.assertEqual(collector.unmatched_visual_notes, set())

    def test_dense_filled_chords_are_not_treated_as_split_whole_note_fragments(
        self,
    ) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly(
                (x, y), (7, 5), -20, 0, 360, 5
            ).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        notes = [
            make_note(30, 35, "first-top"),
            make_note(30, 55, "first-bottom"),
            make_note(50, 35, "second-top"),
            make_note(50, 55, "second-bottom"),
        ]
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
            0, notes, [note.copy() for note in notes]
        )
        symbols = [
            EncodedSymbol("note_16", "F5", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_16", "C5", position="upper"),
            EncodedSymbol("note_16", "F5", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_16", "C5", position="upper"),
        ]

        collector.add_staff_matches(symbols, 0)

        note_symbols = [symbol for symbol in symbols if symbol.rhythm.startswith("note")]
        self.assertEqual(
            [
                collector.matches_by_symbol_id[symbol.visual_match_id].visual_id
                for symbol in note_symbols
            ],
            ["first-top", "first-bottom", "second-top", "second-bottom"],
        )
        self.assertEqual(collector.unmatched_visual_notes, set())

    def test_discards_small_notehead_fragment_that_duplicates_a_detected_stem(
        self,
    ) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(160, 120),
            autocrop_box=(0, 0, 160, 120),
            cropped_size=(160, 120),
            resized_size=(160, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(160, 120),
        )
        image = np.full((120, 160), 255, dtype=np.uint8)
        cv2.ellipse(image, (40, 60), (10, 7), 0, 0, 360, 0, 2)
        cv2.ellipse(image, (110, 60), (10, 7), 0, 0, 360, 0, -1)
        shared_stem_contour = np.array(
            [[[29, 60]], [[31, 60]], [[31, 100]], [[29, 100]]], dtype=np.float32
        )
        separate_stem_contour = np.array(
            [[[99, 60]], [[101, 60]], [[101, 100]], [[99, 100]]], dtype=np.float32
        )
        shared_stem = RotatedBoundingBox(
            cv2.minAreaRect(shared_stem_contour), shared_stem_contour
        )
        separate_stem = RotatedBoundingBox(
            cv2.minAreaRect(separate_stem_contour), separate_stem_contour
        )

        def note(
            visual_id: str,
            center_x: int,
            stem: RotatedBoundingBox,
            contour: np.ndarray,
        ) -> Note:
            return Note(
                BoundingEllipse(((center_x, 60), (20, 14), 0), contour),
                position=10,
                stem=stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        full_first_contour = cv2.ellipse2Poly(
            (40, 60), (10, 7), 0, 0, 360, 10
        ).reshape(-1, 1, 2)
        fragment_contour = np.array(
            [[[58, 58]], [[63, 58]], [[63, 62]], [[58, 62]]], dtype=np.float32
        )
        full_second_contour = cv2.ellipse2Poly(
            (110, 60), (10, 7), 0, 0, 360, 10
        ).reshape(-1, 1, 2)
        notes = [
            note("full-first", 40, shared_stem, full_first_contour),
            note("fragment", 61, shared_stem, fragment_contour),
            note("full-second", 110, separate_stem, full_second_contour),
        ]
        collector = VisualSidecar(metadata, source_image=image)
        collector.add_staff_visual_notes(
            0, notes, [candidate.copy() for candidate in notes]
        )
        first_symbol = EncodedSymbol("note_16", "B3", coordinates=(40, 60))
        second_symbol = EncodedSymbol("note_16", "D4", coordinates=(110, 60))

        collector.add_staff_matches([first_symbol, second_symbol], 0)

        sidecar = collector.to_json_dict()
        groups = {
            group["visual_group_id"]: group for group in sidecar["visual_groups"]
        }
        self.assertEqual(set(groups), {"full-first", "fragment", "full-second"})
        self.assertEqual(groups["fragment"]["visual_status"], "diagnostic")
        self.assertIn("suspected_duplicate", groups["fragment"]["repair_actions"])
        self.assertEqual(sidecar["unmatched_visual_notes"], ["fragment"])
        self.assertEqual(
            collector.matches_by_symbol_id[first_symbol.visual_match_id].visual_id,
            "full-first",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[second_symbol.visual_match_id].visual_id,
            "full-second",
        )

    def test_split_stem_across_displaced_noteheads_is_exported_as_one_chord(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        upper_stem_contour = np.array(
            [[[50, 20]], [[52, 20]], [[52, 50]], [[50, 50]]], dtype=np.float32
        )
        lower_stem_contour = np.array(
            [[[50, 66]], [[52, 66]], [[52, 96]], [[50, 96]]], dtype=np.float32
        )
        upper_stem = RotatedBoundingBox(
            cv2.minAreaRect(upper_stem_contour), upper_stem_contour
        )
        lower_stem = RotatedBoundingBox(
            cv2.minAreaRect(lower_stem_contour), lower_stem_contour
        )

        def make_note(
            x: int,
            y: int,
            visual_id: str,
            stem: RotatedBoundingBox,
        ) -> Note:
            contour = cv2.ellipse2Poly((x, y), (10, 7), 0, 0, 360, 10).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, y), (20, 14), 0), contour),
                position=4,
                stem=stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        notes = [
            make_note(60, 30, "vnote-top", upper_stem),
            make_note(60, 50, "vnote-middle", upper_stem),
            make_note(40, 60, "vnote-displaced", lower_stem),
        ]
        collector = VisualSidecar(metadata, stem_fragments=[upper_stem, lower_stem])
        collector.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        collector.add_staff_matches(
            [
                EncodedSymbol("note_8", "C6", coordinates=(60, 30)),
                EncodedSymbol("note_8", "G5", coordinates=(60, 50)),
                EncodedSymbol("note_8", "F5", coordinates=(40, 60)),
            ],
            0,
        )

        groups = {
            group["visual_group_id"]: group
            for group in collector.to_json_dict()["visual_groups"]
        }
        chord_component_ids = groups["vnote-top"]["stem_component_ids"]

        self.assertTrue(chord_component_ids)
        self.assertEqual(
            groups["vnote-middle"]["stem_component_ids"], chord_component_ids
        )
        self.assertEqual(
            groups["vnote-displaced"]["stem_component_ids"], chord_component_ids
        )

    def test_visual_notes_are_matched_by_attention_position_not_flat_cursor(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((x, y), (5, 4), 0, 0, 360, 10).reshape(-1, 1, 2)
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
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        symbols = [
            EncodedSymbol("note_4", "C5", coordinates=(20, 20)),
            EncodedSymbol("note_8", "D5", coordinates=(60, 20)),
            EncodedSymbol("note_16", "C3", coordinates=(20, 80)),
            EncodedSymbol("note_2", "D3", coordinates=(60, 80)),
        ]

        collector.add_staff_matches(symbols, 0)

        self.assertEqual(collector.matches_by_symbol_id[id(symbols[0])].visual_id, "upper-left")
        self.assertEqual(collector.matches_by_symbol_id[id(symbols[1])].visual_id, "upper-right")
        self.assertEqual(collector.matches_by_symbol_id[id(symbols[2])].visual_id, "lower-left")
        self.assertEqual(collector.matches_by_symbol_id[id(symbols[3])].visual_id, "lower-right")

    def test_token_chord_order_maps_heads_without_using_predicted_pitch(self) -> None:
        metadata = PreprocessingMetadata(
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
        shared_stem = RotatedBoundingBox(
            cv2.minAreaRect(stem_contour), stem_contour
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly(
                (x, y), (10, 7), -20, 0, 360, 5
            ).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((x, y), (20, 14), -20), contour),
                position=4,
                stem=shared_stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        top = make_note(60, 30, "top")
        displaced_bottom = make_note(40, 60, "displaced-bottom")
        collector = VisualSidecar(metadata, stem_fragments=[shared_stem])
        collector.add_staff_visual_notes(
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

        collector.add_staff_matches(
            [
                token_first_with_wrong_low_pitch,
                EncodedSymbol("chord"),
                token_second_with_wrong_high_pitch,
            ],
            0,
        )

        self.assertEqual(
            collector.matches_by_symbol_id[
                token_first_with_wrong_low_pitch.visual_match_id
            ].visual_id,
            "top",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[
                token_second_with_wrong_high_pitch.visual_match_id
            ].visual_id,
            "displaced-bottom",
        )
        self.assertEqual(
            collector.visual_groups["top"].moment_id,
            collector.visual_groups["displaced-bottom"].moment_id,
        )
        self.assertEqual(
            collector.visual_groups["top"].chord_id,
            collector.visual_groups["displaced-bottom"].chord_id,
        )
        self.assertIsNotNone(collector.visual_groups["top"].chord_id)

    def test_ambiguous_repeated_notes_remain_unmatched(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        contour = cv2.ellipse2Poly(
            (50, 40), (7, 5), -20, 0, 360, 5
        ).reshape(-1, 1, 2)
        note = Note(
            BoundingEllipse(((50, 40), (14, 10), -20), contour),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="repeated-candidate",
        )
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(0, [note], [note.copy()])
        first = EncodedSymbol("note_16", "C5", coordinates=(50, 40))
        second = EncodedSymbol("note_16", "C5", coordinates=(50, 40))

        collector.add_staff_matches([first, second], 0)

        self.assertIsNone(
            collector.matches_by_symbol_id[first.visual_match_id].visual_id
        )
        self.assertIsNone(
            collector.matches_by_symbol_id[second.visual_match_id].visual_id
        )
        group = collector.to_json_dict()["visual_groups"][0]
        self.assertEqual(group["visual_status"], "diagnostic")
        self.assertEqual(
            collector.to_json_dict()["unmatched_visual_notes"],
            ["repeated-candidate"],
        )

    def test_shared_stem_repairs_chord_member_swapped_with_neighbor(self) -> None:
        metadata = PreprocessingMetadata(
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
        shared_stem = RotatedBoundingBox(
            cv2.minAreaRect(shared_stem_contour), shared_stem_contour
        )

        def make_note(
            x: int,
            y: int,
            visual_id: str,
            stem: RotatedBoundingBox | None = None,
        ) -> Note:
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(
                -1, 1, 2
            )
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
        collector = VisualSidecar(metadata, stem_fragments=[shared_stem])
        collector.add_staff_visual_notes(
            0,
            [single, chord_top, chord_bottom, extra],
            [single.copy(), chord_top.copy(), chord_bottom.copy(), extra.copy()],
        )
        single_symbol = EncodedSymbol("note_16", "G4", coordinates=(60, 60))
        chord_top_symbol = EncodedSymbol("note_16", "D5", coordinates=(60, 40))
        chord_bottom_symbol = EncodedSymbol("note_16", "Bb4", coordinates=(40, 80))

        collector.add_staff_matches(
            [
                single_symbol,
                chord_top_symbol,
                EncodedSymbol("chord"),
                chord_bottom_symbol,
            ],
            0,
        )

        self.assertEqual(
            collector.matches_by_symbol_id[single_symbol.visual_match_id].visual_id,
            "single",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[chord_top_symbol.visual_match_id].visual_id,
            "chord-top",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[chord_bottom_symbol.visual_match_id].visual_id,
            "chord-bottom",
        )

    def test_adjacent_opposing_stems_do_not_swap_neighbor_with_chord_member(
        self,
    ) -> None:
        metadata = PreprocessingMetadata(
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
            contour = cv2.ellipse2Poly((x, y), (10, 7), -20, 0, 360, 5).reshape(
                -1, 1, 2
            )
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
        collector = VisualSidecar(
            metadata,
            stem_fragments=[
                preceding_up_stem,
                chord_top_up_stem,
                chord_bottom_down_stem,
            ],
        )
        collector.add_staff_visual_notes(
            0,
            notes,
            [candidate.copy() for candidate in notes],
        )
        preceding_symbol = EncodedSymbol("note_32", "G4", coordinates=(48, 50))
        chord_top_symbol = EncodedSymbol("note_8", "B4", coordinates=(76, 35))
        chord_bottom_symbol = EncodedSymbol("note_4", "F#4", coordinates=(76, 60))

        collector.add_staff_matches(
            [
                preceding_symbol,
                chord_top_symbol,
                EncodedSymbol("chord"),
                chord_bottom_symbol,
            ],
            0,
        )

        self.assertEqual(
            collector.matches_by_symbol_id[preceding_symbol.visual_match_id].visual_id,
            "preceding",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[chord_top_symbol.visual_match_id].visual_id,
            "chord-top",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[chord_bottom_symbol.visual_match_id].visual_id,
            "chord-bottom",
        )
        self.assertTrue(
            set(collector.visual_groups["preceding"].owned_stem_component_ids).isdisjoint(
                collector.visual_groups["chord-bottom"].owned_stem_component_ids
            )
        )
        for visual_id in ("preceding", "chord-bottom"):
            points = collector.visual_groups[visual_id].stem_contours[0]
            self.assertLess(
                max(point[1] for point in points) - min(point[1] for point in points),
                50,
            )

    def test_complete_moments_override_repeated_note_attention_across_staves(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(
                -1, 1, 2
            )
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
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
            0,
            [upper_first, lower_first, upper_second],
            [upper_first.copy(), lower_first.copy(), upper_second.copy()],
        )
        collector.visual_groups["lower-first"].stave_index = 1
        first_upper_symbol = EncodedSymbol(
            "note_16", "Gb4", position="upper", coordinates=(60, 35)
        )
        first_lower_symbol = EncodedSymbol(
            "note_2", "Bb3", position="lower", coordinates=(20, 95)
        )
        second_upper_symbol = EncodedSymbol(
            "note_16", "Gb4", position="upper", coordinates=(20, 35)
        )

        collector.add_staff_matches(
            [
                first_upper_symbol,
                EncodedSymbol("chord"),
                first_lower_symbol,
                second_upper_symbol,
            ],
            0,
        )

        self.assertEqual(
            collector.matches_by_symbol_id[first_upper_symbol.visual_match_id].visual_id,
            "upper-first",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[first_lower_symbol.visual_match_id].visual_id,
            "lower-first",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[second_upper_symbol.visual_match_id].visual_id,
            "upper-second",
        )
        first_upper_group = collector.visual_groups["upper-first"]
        first_lower_group = collector.visual_groups["lower-first"]
        self.assertEqual(first_upper_group.moment_id, first_lower_group.moment_id)
        self.assertIsNone(first_upper_group.chord_id)
        self.assertIsNone(first_lower_group.chord_id)

    def test_cross_stave_duplicate_at_ledger_boundary_is_consolidated(self) -> None:
        metadata = PreprocessingMetadata(
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
                cv2.ellipse2Poly((80, y), (7, 5), -20, 0, 360, 5).reshape(
                    -1, 1, 2
                ),
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
            cv2.ellipse2Poly((80, 85), (7, 5), -20, 0, 360, 5).reshape(
                -1, 1, 2
            ),
        )
        boundary_upper = make_note(
            85, "boundary-a-upper", upper_point, shared_boundary_box
        )
        boundary_lower = make_note(
            85, "boundary-b-lower", lower_point, shared_boundary_box
        )
        lower_middle = make_note(120, "lower-middle", lower_point)
        lower_bottom = make_note(140, "lower-bottom", lower_point)
        for note in (upper_top, upper_bottom, boundary_upper):
            upper_staff.add_symbol(note)
        for note in (boundary_lower, lower_middle, lower_bottom):
            lower_staff.add_symbol(note)
        grand_staff = upper_staff.merge(lower_staff)
        notes = grand_staff.get_notes()
        collector = VisualSidecar(metadata)
        collector.prepare_recovery_notes([grand_staff])
        collector.add_staff_visual_notes(
            0, notes, [candidate.copy() for candidate in notes]
        )

        upper_top_symbol = EncodedSymbol(
            "note_16", "C6", position="upper", coordinates=(20, 140)
        )
        upper_bottom_symbol = EncodedSymbol(
            "note_16", "C5", position="upper", coordinates=(20, 120)
        )
        lower_top_symbol = EncodedSymbol(
            "note_8", "C4", position="lower", coordinates=(20, 50)
        )
        lower_middle_symbol = EncodedSymbol(
            "note_8", "C3", position="lower", coordinates=(20, 30)
        )
        lower_bottom_symbol = EncodedSymbol(
            "note_8", "C2", position="lower", coordinates=(20, 20)
        )
        collector.add_staff_matches(
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
            collector.matches_by_symbol_id[lower_top_symbol.visual_match_id].visual_id,
            "boundary-a-upper",
        )
        repaired = collector.visual_groups["boundary-a-upper"]
        rejected = collector.visual_groups["boundary-b-lower"]
        self.assertEqual(repaired.stave_index, 1)
        self.assertEqual(repaired.staff_position, boundary_lower.position)
        self.assertEqual(repaired.visual_status, "canonical")
        self.assertIn("duplicate_candidates_consolidated", repaired.repair_actions)
        self.assertIn("stave_membership_repaired", repaired.repair_actions)
        self.assertEqual(rejected.visual_status, "diagnostic")
        self.assertIn("suspected_duplicate", rejected.repair_actions)
        self.assertEqual(collector.unmatched_visual_notes, {"boundary-b-lower"})
        self.assertEqual(
            {
                collector.visual_groups[
                    collector.matches_by_symbol_id[symbol.visual_match_id].visual_id
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
        metadata = PreprocessingMetadata(
            source_image_size=(140, 120),
            autocrop_box=(0, 0, 140, 120),
            cropped_size=(140, 120),
            resized_size=(140, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 120),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(
                -1, 1, 2
            )
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
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        first_symbol = EncodedSymbol(
            "note_16", "C5", position="upper", coordinates=(60, 35)
        )
        second_symbol = EncodedSymbol(
            "note_16", "D5", position="upper", coordinates=(20, 35)
        )
        third_symbol = EncodedSymbol(
            "note_16", "E5", position="upper", coordinates=(100, 35)
        )

        collector.add_staff_matches(
            [first_symbol, second_symbol, third_symbol],
            0,
        )

        self.assertEqual(
            collector.matches_by_symbol_id[first_symbol.visual_match_id].visual_id,
            "first",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[second_symbol.visual_match_id].visual_id,
            "second",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[third_symbol.visual_match_id].visual_id,
            "third",
        )
        self.assertEqual(collector.unmatched_visual_notes, {"surplus"})

    def test_attention_anchor_repairs_exact_other_stave_beside_surplus_head(
        self,
    ) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(140, 160),
            autocrop_box=(0, 0, 140, 160),
            cropped_size=(140, 160),
            resized_size=(140, 160),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 160),
        )

        def make_note(x: int, y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(
                -1, 1, 2
            )
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
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
            0,
            original_notes,
            transformed_notes,
        )
        collector.visual_groups["lower-top"].stave_index = 1
        collector.visual_groups["lower-bottom"].stave_index = 1

        upper_symbol = EncodedSymbol(
            "note_16", "Eb6", position="upper", coordinates=(20, 30)
        )
        lower_top_symbol = EncodedSymbol(
            "note_8", "Bb4", position="lower"
        )
        lower_bottom_symbol = EncodedSymbol(
            "note_8", "Db4", position="lower"
        )
        collector.add_staff_matches(
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
            match = collector.matches_by_symbol_id[match_id]
            self.assertEqual(match.visual_id, visual_id)
            self.assertEqual(match.alignment_method, "sequence_repair")
            self.assertEqual(
                collector.visual_groups[visual_id].visual_status,
                "fallback",
            )
        self.assertEqual(
            {
                collector.visual_groups[visual_id].moment_id
                for visual_id in expected_matches.values()
            },
            {"moment-1-1"},
        )
        self.assertEqual(collector.unmatched_visual_notes, {"surplus-upper"})
        self.assertEqual(
            collector.visual_groups["surplus-upper"].visual_status,
            "diagnostic",
        )

    def test_token_order_repairs_shared_stem_subset_without_attention(self) -> None:
        metadata = PreprocessingMetadata(
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
        shared_stem = RotatedBoundingBox(
            cv2.minAreaRect(shared_stem_contour), shared_stem_contour
        )

        def make_note(
            x: int,
            y: int,
            visual_id: str,
            note_stem: RotatedBoundingBox | None = None,
        ) -> Note:
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(
                -1, 1, 2
            )
            return Note(
                BoundingEllipse(((x, y), (14, 10), -20), contour),
                position=4,
                stem=note_stem,
                stem_direction=None,
                visual_id=visual_id,
            )

        upper_top = make_note(60, 30, "upper-top", shared_stem)
        surplus_upper_bottom = make_note(
            60, 55, "surplus-upper-bottom", shared_stem
        )
        lower_top = make_note(60, 115, "lower-top")
        lower_bottom = make_note(60, 140, "lower-bottom")
        notes = [upper_top, surplus_upper_bottom, lower_top, lower_bottom]
        collector = VisualSidecar(metadata, stem_fragments=[shared_stem])
        collector.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        collector.visual_groups["lower-top"].stave_index = 1
        collector.visual_groups["lower-bottom"].stave_index = 1

        upper_symbol = EncodedSymbol("note_16", "Eb6", position="upper")
        lower_top_symbol = EncodedSymbol("note_8", "Db4", position="lower")
        lower_bottom_symbol = EncodedSymbol("note_8", "Db3", position="lower")
        collector.add_staff_matches(
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
            match = collector.matches_by_symbol_id[match_id]
            self.assertEqual(match.visual_id, visual_id)
            self.assertEqual(match.alignment_method, "sequence_repair")
            self.assertEqual(
                collector.visual_groups[visual_id].visual_status,
                "fallback",
            )
        self.assertEqual(
            {
                collector.visual_groups[visual_id].moment_id
                for visual_id in expected_matches.values()
            },
            {"moment-1-1"},
        )
        self.assertEqual(
            collector.unmatched_visual_notes,
            {"surplus-upper-bottom"},
        )
        self.assertEqual(
            collector.visual_groups["surplus-upper-bottom"].visual_status,
            "diagnostic",
        )

    def test_unique_shared_stem_subset_excludes_unrelated_chord_candidate(self) -> None:
        metadata = PreprocessingMetadata(
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
            contour = cv2.ellipse2Poly((x, y), (7, 5), -20, 0, 360, 5).reshape(
                -1, 1, 2
            )
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
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
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
            collector.visual_groups[visual_id].stave_index = 1
        for visual_id in ("lower-top", "lower-middle", "lower-bottom"):
            collector.visual_groups[visual_id].owned_stem_component_ids = [
                "shared-lower-stem"
            ]

        upper_symbol = EncodedSymbol("note_32", "D5", position="upper")
        lower_top_symbol = EncodedSymbol("note_16", "G4", position="lower")
        lower_middle_symbol = EncodedSymbol("note_16", "F4", position="lower")
        lower_bottom_symbol = EncodedSymbol("note_16", "D4", position="lower")
        collector.add_staff_matches(
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
            match = collector.matches_by_symbol_id[match_id]
            self.assertEqual(match.visual_id, visual_id)
            self.assertEqual(match.alignment_method, "sequence_repair")
            self.assertEqual(
                collector.visual_groups[visual_id].visual_status,
                "fallback",
            )
        self.assertEqual(
            {
                collector.visual_groups[visual_id].moment_id
                for visual_id in expected_matches.values()
            },
            {"moment-1-1"},
        )
        self.assertEqual(collector.unmatched_visual_notes, {"surplus-lower"})
        self.assertEqual(
            collector.visual_groups["surplus-lower"].visual_status,
            "diagnostic",
        )

    def test_extra_visual_moment_does_not_shift_later_structural_matches(self) -> None:
        metadata = PreprocessingMetadata(
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
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        collector.visual_groups["surplus"].stave_index = 1
        first_symbol = EncodedSymbol("note_16", "C5", position="upper", coordinates=(80, 35))
        second_symbol = EncodedSymbol("note_16", "D5", position="upper", coordinates=(20, 35))
        third_symbol = EncodedSymbol("note_16", "E5", position="upper", coordinates=(50, 90))

        collector.add_staff_matches(
            [first_symbol, second_symbol, third_symbol],
            0,
        )

        self.assertEqual(
            collector.matches_by_symbol_id[first_symbol.visual_match_id].visual_id,
            "first",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[second_symbol.visual_match_id].visual_id,
            "second",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[third_symbol.visual_match_id].visual_id,
            "third",
        )
        self.assertEqual(collector.unmatched_visual_notes, {"surplus"})

    def test_unpitched_note_reserves_its_visual_moment_before_final_chord(self) -> None:
        metadata = PreprocessingMetadata(
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
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        first_symbol = EncodedSymbol("note_32", "C5", position="upper", coordinates=(130, 45))
        unknown_symbol = EncodedSymbol("note_32", ".", position="upper", coordinates=(70, 30))
        chord_top_symbol = EncodedSymbol("note_2", "A5", position="upper", coordinates=(20, 35))
        chord_middle_symbol = EncodedSymbol("note_2", "F5", position="upper", coordinates=(130, 65))
        chord_bottom_symbol = EncodedSymbol("note_2", "A4", position="upper", coordinates=(130, 30))

        collector.add_staff_matches(
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
            collector.matches_by_symbol_id[first_symbol.visual_match_id].visual_id,
            "first",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[chord_top_symbol.visual_match_id].visual_id,
            "chord-top",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[chord_middle_symbol.visual_match_id].visual_id,
            "chord-middle",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[chord_bottom_symbol.visual_match_id].visual_id,
            "chord-bottom",
        )
        self.assertNotIn(unknown_symbol.visual_match_id, collector.matches_by_symbol_id)
        self.assertEqual(collector.unmatched_visual_notes, {"placeholder"})

    def test_duplicate_predicted_pitch_does_not_steal_the_next_visual_moment(self) -> None:
        metadata = PreprocessingMetadata(
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
        collector = VisualSidecar(metadata)
        collector.add_staff_visual_notes(
            0,
            [first, second],
            [first.copy(), second.copy()],
        )
        retained_symbol = EncodedSymbol("note_32", "C5", position="upper", coordinates=(80, 40))
        duplicate_symbol = EncodedSymbol("note_32", "C5", position="upper", coordinates=(20, 40))
        following_symbol = EncodedSymbol("note_32", "D5", position="upper", coordinates=(20, 40))

        collector.add_staff_matches(
            [
                retained_symbol,
                EncodedSymbol("chord"),
                duplicate_symbol,
                following_symbol,
            ],
            0,
        )

        self.assertEqual(
            collector.matches_by_symbol_id[retained_symbol.visual_match_id].visual_id,
            "first",
        )
        self.assertEqual(
            collector.matches_by_symbol_id[following_symbol.visual_match_id].visual_id,
            "second",
        )
        self.assertNotIn(duplicate_symbol.visual_match_id, collector.matches_by_symbol_id)

    def test_split_chord_outlier_is_released_for_pixel_backed_recovery(self) -> None:
        metadata = PreprocessingMetadata(
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
        collector = VisualSidecar(metadata, source_image=image)
        collector.add_staff_visual_notes(
            0,
            notes,
            [note.copy() for note in notes],
        )
        collector.visual_groups["previous"].stave_index = 1
        previous_symbol = EncodedSymbol("note_16", "C3", position="lower", coordinates=(60, 40))
        chord_top_symbol = EncodedSymbol("note_2", "E5", position="upper", coordinates=(100, 25))
        chord_bottom_symbol = EncodedSymbol("note_2", "G4", position="upper", coordinates=(20, 50))
        following_symbol = EncodedSymbol("note_16", "D5", position="upper", coordinates=(140, 40))

        collector.add_staff_matches(
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

        recovered_id = collector.matches_by_symbol_id[chord_bottom_symbol.visual_match_id].visual_id
        self.assertIsNotNone(recovered_id)
        self.assertTrue(str(recovered_id).startswith("vnote-transformer-recovered-"))
        recovered = collector.visual_groups[str(recovered_id)]
        self.assertAlmostEqual(recovered.prediction_center[0], 100, delta=2)
        self.assertAlmostEqual(recovered.prediction_center[1], 50, delta=2)
        self.assertEqual(
            recovered.chord_id,
            collector.visual_groups["chord-top"].chord_id,
        )
        self.assertIsNotNone(recovered.chord_id)
        self.assertEqual(recovered.visual_status, "fallback")
        self.assertIn("transformer_chord_recovered", recovered.repair_actions)
        self.assertEqual(collector.unmatched_visual_notes, {"stray"})

    def test_recovers_unique_displaced_chord_candidate_after_attention_misses_it(
        self,
    ) -> None:
        metadata = PreprocessingMetadata(
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

        def make_note(
            center: tuple[int, int], position: int, visual_id: str
        ) -> Note:
            contour = cv2.ellipse2Poly(
                center, (7, 5), -20, 0, 360, 5
            ).reshape(-1, 1, 2)
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
        collector = VisualSidecar(metadata, source_image=image)
        collector.add_staff_visual_notes(
            0,
            [anchor, displaced, stray],
            [anchor.copy(), displaced.copy(), stray.copy()],
        )
        anchor_symbol = EncodedSymbol(
            "note_4", "G4", position="upper", coordinates=(60, 50)
        )
        displaced_symbol = EncodedSymbol(
            "note_4", "A4", position="upper", coordinates=(5, 5)
        )

        collector.add_staff_matches(
            [anchor_symbol, EncodedSymbol("chord"), displaced_symbol],
            0,
            source_staff=staff,
        )

        match = collector.matches_by_symbol_id[displaced_symbol.visual_match_id]
        self.assertEqual(match.visual_id, "displaced")
        self.assertEqual(match.alignment_method, "sequence_repair")
        self.assertEqual(
            collector.visual_groups["displaced"].visual_status, "fallback"
        )
        self.assertIn(
            "transformer_chord_candidate_recovered",
            collector.visual_groups["displaced"].repair_actions,
        )
        self.assertEqual(collector.unmatched_visual_notes, {"stray"})

    def test_recovers_missing_head_at_the_edge_of_a_dense_hollow_chord(self) -> None:
        metadata = PreprocessingMetadata(
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

        def make_note(
            center: tuple[int, int], position: int, visual_id: str
        ) -> Note:
            contour = cv2.ellipse2Poly(
                center, (10, 7), -20, 0, 360, 5
            ).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse((center, (20, 14), -20), contour),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        top = make_note(centers[0], 9, "top")
        middle = make_note(centers[1], 7, "middle")
        collector = VisualSidecar(metadata, source_image=image)
        collector.add_staff_visual_notes(
            0, [top, middle], [top.copy(), middle.copy()]
        )
        bottom_symbol = EncodedSymbol(
            "note_1", "F4", position="upper", coordinates=centers[2]
        )
        top_symbol = EncodedSymbol(
            "note_1", "C5", position="upper", coordinates=centers[0]
        )
        middle_symbol = EncodedSymbol(
            "note_1", "A4", position="upper", coordinates=centers[1]
        )

        collector.add_staff_matches(
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

        bottom_match = collector.matches_by_symbol_id[
            bottom_symbol.visual_match_id
        ]
        self.assertIsNotNone(bottom_match.visual_id)
        self.assertTrue(
            str(bottom_match.visual_id).startswith("vnote-transformer-recovered-")
        )
        recovered = collector.visual_groups[str(bottom_match.visual_id)]
        self.assertAlmostEqual(recovered.prediction_center[0], 60, delta=2)
        self.assertAlmostEqual(recovered.prediction_center[1], 62, delta=2)
        self.assertEqual(recovered.visual_status, "fallback")

    def test_recovers_hollow_notehead_positioned_by_transformer(self) -> None:
        metadata = PreprocessingMetadata(
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
        lower_contour = cv2.ellipse2Poly(
            (60, 50), (7, 5), -20, 0, 360, 5
        ).reshape(-1, 1, 2)
        lower = Note(
            BoundingEllipse(((60, 50), (14, 10), -20), lower_contour),
            position=3,
            stem=None,
            stem_direction=None,
            visual_id="vnote-lower",
        )
        collector = VisualSidecar(metadata, source_image=image)
        collector.add_staff_visual_notes(0, [lower], [lower.copy()])
        lower_symbol = EncodedSymbol(
            "note_2.", "G4", position="upper", coordinates=(60, 50)
        )
        # Attention is deliberately offset. Chord pitch and the matched lower head
        # provide the exact same-stave position in the grand-staff source grid.
        upper_symbol = EncodedSymbol(
            "note_2.", "E5", position="upper", coordinates=(68, 10)
        )

        collector.add_staff_matches(
            [upper_symbol, EncodedSymbol("chord"), lower_symbol],
            0,
            source_staff=staff,
        )

        upper_match = collector.matches_by_symbol_id[id(upper_symbol)]
        self.assertIsNotNone(upper_match.visual_id)
        self.assertTrue(str(upper_match.visual_id).startswith("vnote-transformer-recovered-"))
        recovered = collector.visual_groups[str(upper_match.visual_id)]
        self.assertAlmostEqual(recovered.notehead_ellipses[0]["center"][0], 60, delta=2)
        self.assertAlmostEqual(recovered.notehead_ellipses[0]["center"][1], 25, delta=2)
        self.assertEqual(recovered.staff_position, 8)
        self.assertEqual(recovered.visual_status, "fallback")
        self.assertEqual(recovered.provenance, "transformer_recovered")
        self.assertEqual(recovered.chord_id, collector.visual_groups["vnote-lower"].chord_id)
        self.assertIsNotNone(recovered.chord_id)
        self.assertIn("transformer_chord_recovered", recovered.repair_actions)

    def test_transformer_recovered_ledger_head_inherits_its_chord_mates_stave(
        self,
    ) -> None:
        metadata = PreprocessingMetadata(
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
            contour = cv2.ellipse2Poly((60, y), (7, 5), -20, 0, 360, 5).reshape(
                -1, 1, 2
            )
            return Note(
                BoundingEllipse(((60, y), (14, 10), -20), contour),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        treble = make_note(30, "treble", 3)
        bass_bottom = make_note(110, "bass-bottom", 3)
        collector = VisualSidecar(metadata, source_image=image)
        collector.add_staff_visual_notes(
            0,
            [treble, bass_bottom],
            [treble.copy(), bass_bottom.copy()],
        )
        collector.visual_groups["treble"].stave_index = 0
        collector.visual_groups["bass-bottom"].stave_index = 1
        treble_symbol = EncodedSymbol(
            "note_2", "C6", position="upper", coordinates=(60, 30)
        )
        bass_top_symbol = EncodedSymbol(
            "note_2", "D5", position="lower", coordinates=(68, 70)
        )
        bass_bottom_symbol = EncodedSymbol(
            "note_2", "C4", position="lower", coordinates=(60, 110)
        )

        collector.add_staff_matches(
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

        recovered_id = collector.matches_by_symbol_id[
            bass_top_symbol.visual_match_id
        ].visual_id
        self.assertIsNotNone(recovered_id)
        recovered = collector.visual_groups[str(recovered_id)]
        self.assertEqual(recovered.stave_index, 1)
        self.assertEqual(
            recovered.chord_id,
            collector.visual_groups["bass-bottom"].chord_id,
        )
        self.assertIsNotNone(recovered.chord_id)
        self.assertNotEqual(
            recovered.chord_id,
            collector.visual_groups["treble"].chord_id,
        )
        self.assertEqual(recovered.visual_status, "fallback")

    def test_does_not_recover_transformer_note_without_notehead_ink(self) -> None:
        metadata = PreprocessingMetadata(
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
        collector = VisualSidecar(metadata, source_image=image)
        symbol = EncodedSymbol("note_2.", "E5", coordinates=(60, 35))

        collector.add_staff_matches([symbol], 0, source_staff=staff)

        self.assertIsNone(collector.matches_by_symbol_id[id(symbol)].visual_id)
        self.assertEqual(collector.visual_groups, {})

    def test_notehead_fitted_ellipse_is_recorded_in_source_coordinates(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(1000, 800),
            autocrop_box=(100, 50, 400, 300),
            cropped_size=(400, 300),
            resized_size=(800, 600),
            resize_scale=(2.0, 2.0),
            prediction_size=(400, 300),
        )
        collector = VisualSidecar(metadata)
        original = Note(
            BoundingEllipse(((200, 150), (40, 20), 15), np.array([[180, 140], [220, 160]]), 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()
        collector.add_staff_visual_notes(0, [original], [transformed])

        ellipse = collector.to_json_dict()["visual_groups"][0]["notehead_ellipses"][0]

        self.assertEqual(ellipse["center"], [300, 200])
        self.assertEqual(ellipse["rx"], 20)
        self.assertEqual(ellipse["ry"], 10)
        self.assertEqual(ellipse["angle"], 15)

    def test_notehead_contour_fit_uses_svg_compatible_major_axis_angle(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(300, 300),
            autocrop_box=(0, 0, 300, 300),
            cropped_size=(300, 300),
            resized_size=(300, 300),
            resize_scale=(1.0, 1.0),
            prediction_size=(300, 300),
        )
        contour = cv2.ellipse2Poly((100, 100), (40, 20), -30, 0, 360, 2).reshape(-1, 1, 2)
        collector = VisualSidecar(metadata)
        original = Note(
            BoundingEllipse(((100, 100), (80, 40), 0), contour, 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()
        collector.add_staff_visual_notes(0, [original], [transformed])

        ellipse = collector.to_json_dict()["visual_groups"][0]["notehead_ellipses"][0]

        self.assertAlmostEqual(ellipse["rx"], 40, delta=0.5)
        self.assertAlmostEqual(ellipse["ry"], 20, delta=0.5)
        self.assertAlmostEqual(ellipse["angle"], -30, delta=0.5)

    def test_detected_notehead_contour_is_exported_alongside_legacy_polygon(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        detected_contour = np.array(
            [[[46, 49]], [[48, 46]], [[53, 47]], [[55, 50]], [[52, 53]], [[47, 52]]],
            dtype=np.int32,
        )
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.fillPoly(mask, [detected_contour], 1)
        collector = VisualSidecar(metadata, notehead_mask=mask)
        original = Note(
            BoundingEllipse(((50, 50), (12, 8), 0), detected_contour, 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        group = collector.to_json_dict()["visual_groups"][0]

        self.assertIn("notehead_contours", group)
        self.assertIn("detected_notehead_contours", group)
        self.assertNotEqual(group["notehead_contours"], group["detected_notehead_contours"])
        self.assertGreaterEqual(len(group["detected_notehead_contours"][0]), 5)

    def test_refined_notehead_contour_robustly_fits_hollow_head_across_staff_line(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        image = np.full((100, 100), 255, dtype=np.uint8)
        cv2.ellipse(image, ((50, 50), (20, 12), -25), 0, 2)
        cv2.line(image, (25, 52), (75, 52), 0, 1)
        contour = cv2.ellipse2Poly((50, 50), (10, 6), -25, 0, 360, 10).reshape(-1, 1, 2)
        collector = VisualSidecar(metadata, source_image=image)
        original = Note(
            BoundingEllipse(((50, 50), (20, 12), -25), contour, 3),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        group = collector.to_json_dict()["visual_groups"][0]
        refined = group["refined_notehead_contours"][0]
        xs = [point[0] for point in refined]
        ys = [point[1] for point in refined]

        self.assertAlmostEqual((min(xs) + max(xs)) / 2, 50, delta=2)
        self.assertAlmostEqual((min(ys) + max(ys)) / 2, 50, delta=2)
        self.assertGreater(max(xs) - min(xs), max(ys) - min(ys))

    def test_refined_notehead_contours_do_not_borrow_adjacent_chord_ink(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        image = np.full((100, 100), 255, dtype=np.uint8)
        cv2.ellipse(image, ((50, 43), (18, 12), -25), 0, -1)
        cv2.ellipse(image, ((50, 55), (18, 12), -25), 0, -1)
        cv2.line(image, (20, 49), (80, 49), 0, 1)
        collector = VisualSidecar(metadata, source_image=image)

        def make_note(center_y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly(
                (50, center_y), (9, 6), -25, 0, 360, 10
            ).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((50, center_y), (18, 12), -25), contour, center_y),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        top_note = make_note(43, "vnote-top")
        bottom_note = make_note(55, "vnote-bottom")
        notes = [top_note, bottom_note]
        collector.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        groups = {
            group["visual_group_id"]: group for group in collector.to_json_dict()["visual_groups"]
        }
        top_ys = [point[1] for point in groups["vnote-top"]["refined_notehead_contours"][0]]
        bottom_ys = [
            point[1] for point in groups["vnote-bottom"]["refined_notehead_contours"][0]
        ]

        self.assertLessEqual(max(top_ys), 50)
        self.assertGreaterEqual(min(bottom_ys), 48)
        self.assertLess((min(top_ys) + max(top_ys)) / 2, 49)
        self.assertGreater((min(bottom_ys) + max(bottom_ys)) / 2, 49)

    def test_refined_notehead_recovers_center_before_fitting_corrupted_anchor(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 100), autocrop_box=(0, 0, 120, 100),
            cropped_size=(120, 100), resized_size=(120, 100),
            resize_scale=(1.0, 1.0), prediction_size=(120, 100),
        )
        image = np.full((100, 120), 255, dtype=np.uint8)
        cv2.ellipse(image, ((60, 50), (18, 12), -25), 0, -1)
        cv2.line(image, (25, 50), (85, 50), 0, 1)
        corrupted = cv2.ellipse2Poly((52, 54), (16, 6), 0, 0, 360, 10).reshape(-1, 1, 2)
        collector = VisualSidecar(metadata, source_image=image)
        original = Note(
            BoundingEllipse(((52, 54), (32, 12), 0), corrupted, 9),
            position=4, stem=None, stem_direction=None, visual_id="vnote-shifted",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        group = collector.to_json_dict()["visual_groups"][0]
        refined = group["refined_notehead_contours"][0]
        xs = [point[0] for point in refined]
        ys = [point[1] for point in refined]

        self.assertAlmostEqual((min(xs) + max(xs)) / 2, 60, delta=1.5)
        self.assertAlmostEqual((min(ys) + max(ys)) / 2, 50, delta=1.5)
        self.assertLessEqual(max(xs) - min(xs), 20)
        self.assertEqual(group["notehead_contours"][0], refined)
        ellipse = group["notehead_ellipses"][0]
        self.assertAlmostEqual(ellipse["center"][0], 60, delta=1.5)
        self.assertAlmostEqual(ellipse["center"][1], 50, delta=1.5)
        self.assertLessEqual(ellipse["rx"] * 2, 20)

    def test_refined_notehead_does_not_collapse_onto_staff_line(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100), autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100), resized_size=(100, 100),
            resize_scale=(1.0, 1.0), prediction_size=(100, 100),
        )
        image = np.full((100, 100), 255, dtype=np.uint8)
        cv2.ellipse(image, ((50, 50), (18, 12), -25), 0, -1)
        cv2.line(image, (15, 50), (85, 50), 0, 2)
        contour = cv2.ellipse2Poly((50, 50), (9, 6), -25, 0, 360, 10).reshape(-1, 1, 2)
        collector = VisualSidecar(metadata, source_image=image)
        original = Note(
            BoundingEllipse(((50, 50), (18, 12), -25), contour, 10),
            position=4, stem=None, stem_direction=None, visual_id="vnote-on-line",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        refined = collector.to_json_dict()["visual_groups"][0]["refined_notehead_contours"][0]
        xs = [point[0] for point in refined]
        ys = [point[1] for point in refined]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        self.assertGreaterEqual(height, 10)
        self.assertLessEqual(width / height, 1.9)

    def test_refined_filled_notehead_does_not_expand_into_white_staff_gap(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100), autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100), resized_size=(100, 100),
            resize_scale=(1.0, 1.0), prediction_size=(100, 100),
        )
        image = np.full((100, 100), 255, dtype=np.uint8)
        cv2.line(image, (10, 50), (90, 50), 0, 2)
        cv2.line(image, (10, 64), (90, 64), 0, 2)
        cv2.ellipse(image, ((50, 48), (18, 12), -25), 0, -1)
        cv2.line(image, (59, 48), (59, 78), 0, 2)
        contour = cv2.ellipse2Poly((50, 48), (9, 6), -25, 0, 360, 10).reshape(-1, 1, 2)
        collector = VisualSidecar(metadata, source_image=image)
        original = Note(
            BoundingEllipse(((50, 48), (18, 12), -25), contour, 11),
            position=4, stem=None, stem_direction=None, visual_id="vnote-filled-gap",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        refined = collector.to_json_dict()["visual_groups"][0]["refined_notehead_contours"][0]
        xs = [point[0] for point in refined]
        ys = [point[1] for point in refined]

        self.assertLessEqual(max(xs) - min(xs), 21)
        self.assertLessEqual(max(ys) - min(ys), 16)
        self.assertAlmostEqual((min(ys) + max(ys)) / 2, 48, delta=2)

    def test_split_chord_notehead_keeps_split_geometry_when_mask_is_ambiguous(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(300, 300),
            autocrop_box=(0, 0, 300, 300),
            cropped_size=(300, 300),
            resized_size=(300, 300),
            resize_scale=(1.0, 1.0),
            prediction_size=(300, 300),
        )
        mask = np.zeros((300, 300), dtype=np.uint8)
        contour = cv2.ellipse2Poly((100, 100), (40, 20), -30, 0, 360, 2).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [contour], 255)
        collector = VisualSidecar(metadata, notehead_mask=mask)
        original = Note(
            BoundingEllipse(((100, 100), (80, 40), 0), np.array([[60, 80], [140, 120]]), 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()
        collector.add_staff_visual_notes(0, [original], [transformed])

        ellipse = collector.to_json_dict()["visual_groups"][0]["notehead_ellipses"][0]

        self.assertEqual(ellipse["center"], [100, 100])
        self.assertEqual(ellipse["rx"], 40)
        self.assertEqual(ellipse["ry"], 20)
        self.assertEqual(ellipse["angle"], 0)
        self.assertEqual(original.box.angle, 0)

    def test_low_confidence_chord_ellipse_uses_staff_typical_angle(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(300, 300),
            autocrop_box=(0, 0, 300, 300),
            cropped_size=(300, 300),
            resized_size=(300, 300),
            resize_scale=(1.0, 1.0),
            prediction_size=(300, 300),
        )
        collector = VisualSidecar(metadata)
        reliable_contour = cv2.ellipse2Poly((100, 100), (40, 20), -30, 0, 360, 2).reshape(
            -1, 1, 2
        )
        reliable = Note(
            BoundingEllipse(((100, 100), (80, 40), 0), reliable_contour, 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )
        fallback = Note(
            BoundingEllipse(((160, 100), (80, 40), 0), np.array([[120, 80], [200, 120]]), 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-2",
        )

        collector.add_staff_visual_notes(0, [reliable, fallback], [reliable.copy(), fallback.copy()])
        groups = collector.to_json_dict()["visual_groups"]
        fallback_group = next(group for group in groups if group["visual_group_id"] == "vnote-2")

        self.assertAlmostEqual(fallback_group["notehead_ellipses"][0]["angle"], -30, delta=0.5)
        self.assertEqual(fallback.box.angle, 0)

    def test_elongated_horizontal_mask_ellipse_keeps_its_hollow_notehead_angle(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(300, 300), autocrop_box=(0, 0, 300, 300),
            cropped_size=(300, 300), resized_size=(300, 300),
            resize_scale=(1.0, 1.0), prediction_size=(300, 300),
        )
        mask = np.zeros((300, 300), dtype=np.uint8)
        angled_contour = cv2.ellipse2Poly((100, 100), (20, 10), -30, 0, 360, 2).reshape(
            -1, 1, 2
        )
        horizontal_contour = cv2.ellipse2Poly((180, 100), (24, 10), 0, 0, 360, 2).reshape(
            -1, 1, 2
        )
        cv2.fillPoly(mask, [angled_contour, horizontal_contour], 1)
        collector = VisualSidecar(metadata, notehead_mask=mask)
        angled = Note(
            BoundingEllipse(((100, 100), (40, 20), -30), angled_contour, 1),
            position=4, stem=None, stem_direction=None, visual_id="vnote-angled",
        )
        horizontal = Note(
            BoundingEllipse(((180, 100), (48, 20), 0), horizontal_contour, 2),
            position=4, stem=None, stem_direction=None, visual_id="vnote-horizontal",
        )

        notes = [angled, horizontal]
        collector.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        groups = {
            group["visual_group_id"]: group for group in collector.to_json_dict()["visual_groups"]
        }

        self.assertAlmostEqual(
            groups["vnote-horizontal"]["notehead_ellipses"][0]["angle"], 0, delta=1
        )

    def test_compact_hollow_image_notehead_keeps_its_horizontal_mask_angle(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(300, 300), autocrop_box=(0, 0, 300, 300),
            cropped_size=(300, 300), resized_size=(300, 300),
            resize_scale=(1.0, 1.0), prediction_size=(300, 300),
        )
        image = np.full((300, 300), 255, dtype=np.uint8)
        mask = np.zeros((300, 300), dtype=np.uint8)
        angled_contour = cv2.ellipse2Poly((100, 100), (20, 10), -30, 0, 360, 2).reshape(
            -1, 1, 2
        )
        horizontal_contour = cv2.ellipse2Poly((180, 100), (18, 12), 0, 0, 360, 2).reshape(
            -1, 1, 2
        )
        cv2.fillPoly(mask, [angled_contour, horizontal_contour], 1)
        cv2.ellipse(image, ((100, 100), (40, 20), -30), 0, -1)
        cv2.ellipse(image, ((180, 100), (36, 24), 0), 0, 2)
        collector = VisualSidecar(metadata, notehead_mask=mask, source_image=image)
        angled = Note(
            BoundingEllipse(((100, 100), (40, 20), -30), angled_contour, 1),
            position=4, stem=None, stem_direction=None, visual_id="vnote-angled",
        )
        horizontal = Note(
            BoundingEllipse(((180, 100), (36, 24), 0), horizontal_contour, 2),
            position=4, stem=None, stem_direction=None, visual_id="vnote-horizontal",
        )

        notes = [angled, horizontal]
        collector.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        groups = {
            group["visual_group_id"]: group for group in collector.to_json_dict()["visual_groups"]
        }

        self.assertAlmostEqual(
            groups["vnote-horizontal"]["notehead_ellipses"][0]["angle"], 0, delta=1
        )

    def test_musicxml_without_visual_sidecar_has_no_generated_ids(self) -> None:
        symbol = EncodedSymbol("note_4", "C4", "_", "_", "_", "upper")
        xml = generate_xml(XmlGeneratorArguments(), [[symbol]], "")

        self.assertEqual(self._musicxml_note_ids(xml), [])

    def test_split_stem_fragments_are_combined_for_visual_sidecar_geometry_only(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        notehead = BoundingEllipse(((50, 50), (12, 10), 0), np.array([[44, 45], [56, 55]]))
        lower_fragment = RotatedBoundingBox(
            ((56, 44), (2, 14), 0), np.array([[55, 37], [57, 51]])
        )
        upper_fragment = RotatedBoundingBox(
            ((56, 18), (2, 24), 0), np.array([[55, 6], [57, 30]])
        )
        collector = VisualSidecar(metadata, [lower_fragment, upper_fragment])
        original = Note(
            notehead,
            position=4,
            stem=lower_fragment,
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()

        collector.add_staff_visual_notes(0, [original], [transformed])
        visual_sidecar = collector.to_json_dict()
        group = visual_sidecar["visual_groups"][0]
        contour = group["stem_contours"][0]
        height = max(point[1] for point in contour) - min(point[1] for point in contour)

        self.assertGreater(height, lower_fragment.size[1] + 15)
        self.assertEqual(group["detected_stem_contours"], [[[55, 37], [57, 51]]])
        self.assertEqual(
            [stem["contour"] for stem in visual_sidecar["raw_stem_contours"]],
            [
                [[55, 37], [57, 51]],
                [[55, 6], [57, 30]],
            ],
        )
        self.assertEqual(original.stem, lower_fragment)

    def test_visual_sidecar_stem_merge_rejects_wide_beam_fragment(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        notehead = BoundingEllipse(((50, 50), (12, 10), 0), np.array([[44, 45], [56, 55]]))
        stem = RotatedBoundingBox(((56, 40), (2, 20), 0), np.array([[55, 30], [57, 50]]))
        beam = RotatedBoundingBox(((56, 65), (42, 8), 0), np.array([[35, 61], [77, 69]]))
        collector = VisualSidecar(metadata, [beam])
        original = Note(
            notehead,
            position=4,
            stem=stem,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        xs = [point[0] for point in contour]

        self.assertLess(max(xs) - min(xs), 10)
        self.assertEqual(original.stem, stem)

    def test_visual_sidecar_stem_merge_does_not_chain_sideways(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (12, 10), 0), np.array([[44, 45], [56, 55]]))
        stem = RotatedBoundingBox(((56, 42), (2, 16), 0), np.array([[55, 34], [57, 50]]))
        near = RotatedBoundingBox(((61, 25), (2, 18), 0), np.array([[60, 16], [62, 34]]))
        drift = RotatedBoundingBox(((68, 10), (2, 18), 0), np.array([[67, 1], [69, 19]]))
        collector = VisualSidecar(metadata, [near, drift])
        original = Note(
            notehead,
            position=4,
            stem=stem,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        xs = [point[0] for point in contour]

        self.assertLess(max(xs) - min(xs), 10)
        self.assertEqual(original.stem, stem)

    def test_visual_sidecar_stem_repair_does_not_reach_distant_tempo_mark(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(
            ((50, 100), (14, 12), 0), np.array([[43, 94], [57, 106]])
        )
        detected_stem = RotatedBoundingBox(
            ((57, 95), (2, 10), 0), np.array([[56, 90], [58, 100]])
        )
        tempo_mark_stem = RotatedBoundingBox(
            ((64, 15), (2, 30), 0), np.array([[63, 0], [65, 30]])
        )
        collector = VisualSidecar(metadata, [detected_stem, tempo_mark_stem])
        original = Note(
            notehead,
            position=4,
            stem=detected_stem,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        group = collector.to_json_dict()["visual_groups"][0]
        stem_ys = [point[1] for point in group["stem_contours"][0]]

        self.assertGreater(min(stem_ys), 80)
        self.assertEqual(
            [stem["contour"] for stem in collector.to_json_dict()["raw_stem_contours"]],
            [
                [[56, 90], [58, 100]],
                [[63, 0], [65, 30]],
            ],
        )
        self.assertEqual(original.stem, detected_stem)

    def test_visual_sidecar_replaces_tiny_bad_seed_with_nearby_vertical_seed(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        tiny_bad_seed = RotatedBoundingBox(
            ((56, 57), (2, 1), 0), np.array([[55, 57], [57, 58]])
        )
        better_seed = RotatedBoundingBox(
            ((43, 59), (1, 10), 0), np.array([[43, 54], [44, 64]])
        )
        continuation = RotatedBoundingBox(
            ((43, 82), (2, 36), 0), np.array([[42, 64], [44, 100]])
        )
        collector = VisualSidecar(metadata, [tiny_bad_seed, better_seed, continuation])
        original = Note(
            notehead,
            position=4,
            stem=tiny_bad_seed,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        xs = [point[0] for point in contour]
        ys = [point[1] for point in contour]

        self.assertLess(max(xs), notehead.center[0])
        self.assertGreater(max(ys) - min(ys), 40)
        self.assertEqual(original.stem, tiny_bad_seed)

    def test_visual_sidecar_repairs_missing_downward_stem_from_nearby_chain(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        seed = RotatedBoundingBox(((43, 66), (1, 16), 0), np.array([[43, 58], [44, 74]]))
        continuation = RotatedBoundingBox(
            ((43, 88), (2, 28), 0), np.array([[42, 74], [44, 102]])
        )
        collector = VisualSidecar(metadata, [seed, continuation])
        original = Note(
            notehead,
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        xs = [point[0] for point in contour]
        ys = [point[1] for point in contour]

        self.assertLess(max(xs), notehead.center[0])
        self.assertLessEqual(min(ys), notehead.center[1])
        self.assertGreater(max(ys) - min(ys), 45)
        self.assertIsNone(original.stem)

    def test_visual_sidecar_does_not_borrow_disconnected_peer_stem(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        top_notehead = BoundingEllipse(
            ((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]])
        )
        bottom_notehead = BoundingEllipse(
            ((50, 72), (14, 12), 0), np.array([[43, 66], [57, 78]])
        )
        detected_bottom_stem = RotatedBoundingBox(
            ((43, 72), (2, 12), 0), np.array([[42, 66], [44, 78]])
        )
        bottom_continuation = RotatedBoundingBox(
            ((43, 92), (2, 28), 0), np.array([[42, 78], [44, 106]])
        )
        collector = VisualSidecar(metadata, [detected_bottom_stem, bottom_continuation])
        top_note = Note(top_notehead, 8, None, None, "vnote-top")
        bottom_note = Note(
            bottom_notehead, 4, detected_bottom_stem, None, "vnote-bottom"
        )

        collector.add_staff_visual_notes(
            0,
            [top_note, bottom_note],
            [top_note.copy(), bottom_note.copy()],
        )
        groups = {
            group["visual_group_id"]: group
            for group in collector.to_json_dict()["visual_groups"]
        }

        self.assertEqual(groups["vnote-top"]["stem_contours"], [])
        self.assertNotEqual(groups["vnote-bottom"]["stem_contours"], [])

    def test_visual_sidecar_bridges_disconnected_downward_stem_to_notehead(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        disconnected = RotatedBoundingBox(
            ((43, 72), (1, 28), 0), np.array([[43, 58], [44, 86]])
        )
        continuation = RotatedBoundingBox(
            ((43, 96), (2, 20), 0), np.array([[42, 86], [44, 106]])
        )
        collector = VisualSidecar(metadata, [disconnected, continuation])
        original = Note(
            notehead,
            position=4,
            stem=disconnected,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        ys = [point[1] for point in contour]

        self.assertLessEqual(min(ys), notehead.center[1])
        self.assertGreater(max(ys), 100)
        self.assertEqual(original.stem, disconnected)

    def test_visual_sidecar_extends_short_top_piece_to_downward_chain(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        short_piece = RotatedBoundingBox(
            ((43, 60), (1, 8), 0), np.array([[43, 56], [44, 64]])
        )
        continuation = RotatedBoundingBox(
            ((43, 86), (2, 40), 0), np.array([[42, 66], [44, 106]])
        )
        collector = VisualSidecar(metadata, [short_piece, continuation])
        original = Note(
            notehead,
            position=4,
            stem=short_piece,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        ys = [point[1] for point in contour]

        self.assertLessEqual(min(ys), notehead.center[1])
        self.assertGreater(max(ys) - min(ys), 50)
        self.assertEqual(original.stem, short_piece)

    def test_visual_sidecar_extends_short_top_piece_across_larger_aligned_gap(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(140, 140),
            autocrop_box=(0, 0, 140, 140),
            cropped_size=(140, 140),
            resized_size=(140, 140),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 140),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        short_piece = RotatedBoundingBox(
            ((43, 60), (1, 8), 0), np.array([[43, 56], [44, 64]])
        )
        continuation = RotatedBoundingBox(
            ((43, 94), (2, 16), 0), np.array([[42, 86], [44, 102]])
        )
        collector = VisualSidecar(metadata, [short_piece, continuation])
        original = Note(
            notehead,
            position=4,
            stem=short_piece,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        ys = [point[1] for point in contour]

        self.assertLessEqual(min(ys), notehead.center[1])
        self.assertGreater(max(ys), 100)
        self.assertEqual(original.stem, short_piece)

    def test_visual_sidecar_repairs_missing_upward_stem_from_nearby_chain(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        seed = RotatedBoundingBox(((57, 34), (1, 16), 0), np.array([[57, 26], [58, 42]]))
        continuation = RotatedBoundingBox(
            ((57, 12), (2, 28), 0), np.array([[56, -2], [58, 26]])
        )
        collector = VisualSidecar(metadata, [seed, continuation])
        original = Note(
            notehead,
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        xs = [point[0] for point in contour]
        ys = [point[1] for point in contour]

        self.assertGreater(min(xs), notehead.center[0])
        self.assertGreaterEqual(max(ys), notehead.center[1])
        self.assertGreater(max(ys) - min(ys), 45)
        self.assertIsNone(original.stem)

    def test_visual_sidecar_bridges_disconnected_upward_stem_to_notehead(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        disconnected = RotatedBoundingBox(
            ((57, 28), (1, 28), 0), np.array([[57, 14], [58, 42]])
        )
        continuation = RotatedBoundingBox(
            ((57, 4), (2, 20), 0), np.array([[56, -6], [58, 14]])
        )
        collector = VisualSidecar(metadata, [disconnected, continuation])
        original = Note(
            notehead,
            position=4,
            stem=disconnected,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        ys = [point[1] for point in contour]

        self.assertLess(min(ys), 0)
        self.assertGreaterEqual(max(ys), notehead.center[1])
        self.assertEqual(original.stem, disconnected)

    def test_visual_sidecar_extends_short_bottom_piece_to_upward_chain(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(140, 140),
            autocrop_box=(0, 0, 140, 140),
            cropped_size=(140, 140),
            resized_size=(140, 140),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 140),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        short_piece = RotatedBoundingBox(
            ((57, 40), (1, 8), 0), np.array([[57, 36], [58, 44]])
        )
        continuation = RotatedBoundingBox(
            ((57, 6), (2, 16), 0), np.array([[56, -2], [58, 14]])
        )
        collector = VisualSidecar(metadata, [short_piece, continuation])
        original = Note(
            notehead,
            position=4,
            stem=short_piece,
            stem_direction=None,
            visual_id="vnote-1",
        )

        collector.add_staff_visual_notes(0, [original], [original.copy()])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        ys = [point[1] for point in contour]

        self.assertLess(min(ys), 0)
        self.assertGreaterEqual(max(ys), notehead.center[1])
        self.assertGreater(max(ys) - min(ys), 50)
        self.assertEqual(original.stem, short_piece)

    def _musicxml_note_ids(self, xml: object) -> list[str]:
        ids = []

        def walk(node: object) -> None:
            if node.__class__.__name__ == "XMLNote":
                attrs = getattr(node, "_attributes", {})
                if "id" in attrs:
                    ids.append(str(attrs["id"]))
            children = []
            if hasattr(node, "get_children"):
                children = node.get_children()
            elif hasattr(node, "children"):
                children = node.children
            for child in children:
                walk(child)

        walk(xml)
        return ids


if __name__ == "__main__":
    unittest.main()
