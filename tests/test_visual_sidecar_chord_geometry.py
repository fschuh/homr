import unittest

import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import Note
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder


class TestVisualSidecarBuilderChordGeometry(unittest.TestCase):
    def test_shared_stem_components_are_exported_as_chord_identity(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
        shared_stem = RotatedBoundingBox(cv2.minAreaRect(shared_stem_contour), shared_stem_contour)
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
        builder = VisualSidecarBuilder(
            coordinate_transform, stem_fragments=[shared_stem, separate_stem]
        )
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        symbols = [
            EncodedSymbol("note_4", "C4", coordinates=(15, 15)),
            # A stray augmentation-dot prediction must not split noteheads that
            # share both a physical stem and the same base duration.
            EncodedSymbol("note_4.", "E4", coordinates=(15, 30)),
            EncodedSymbol("note_8", "F4", coordinates=(15, 42)),
            EncodedSymbol("note_4", "G4", coordinates=(65, 15)),
        ]
        builder.add_staff_matches(symbols, 0)

        groups = {
            group["visual_group_id"]: group for group in builder.to_json_dict()["visual_groups"]
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
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )

        def make_note(y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((50, y), (7, 5), -20, 0, 360, 5).reshape(-1, 1, 2)
            return Note(
                BoundingEllipse(((50, y), (14, 10), -20), contour),
                position=4,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        top = make_note(25, "top")
        bottom = make_note(55, "bottom")
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(
            0,
            [top, bottom],
            [top.copy(), bottom.copy()],
        )
        top_symbol = EncodedSymbol("note_16", "Eb6", position="upper", coordinates=(50, 25))
        bottom_symbol = EncodedSymbol("note_16", "Eb5", position="upper", coordinates=(50, 55))

        builder.add_staff_matches(
            [top_symbol, EncodedSymbol("chord"), bottom_symbol],
            0,
        )

        top_group = builder.visual_groups["top"]
        bottom_group = builder.visual_groups["bottom"]
        self.assertEqual(
            builder.matches_by_symbol_id[top_symbol.visual_match_id].alignment_method,
            "structural",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[bottom_symbol.visual_match_id].alignment_method,
            "structural",
        )
        self.assertEqual(top_group.chord_id, bottom_group.chord_id)
        self.assertIsNotNone(top_group.chord_id)
        self.assertIn("structural_chord_proven", top_group.repair_actions)
        self.assertIn("structural_chord_proven", bottom_group.repair_actions)
        self.assertEqual(top_group.visual_status, "canonical")
        self.assertEqual(bottom_group.visual_status, "canonical")

    def test_close_opposed_stem_voices_never_share_physical_chord(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
        upward_stem = RotatedBoundingBox(cv2.minAreaRect(upward_contour), upward_contour)
        downward_stem = RotatedBoundingBox(cv2.minAreaRect(downward_contour), downward_contour)

        def make_note(y: int, position: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((50, y), (10, 7), -20, 0, 360, 5).reshape(-1, 1, 2)
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
                builder = VisualSidecarBuilder(
                    coordinate_transform, stem_fragments=[upward_stem, downward_stem]
                )
                builder.add_staff_visual_notes(
                    0,
                    [upper, lower],
                    [upper.copy(), lower.copy()],
                )
                builder.add_staff_matches(
                    [
                        EncodedSymbol("note_8", "Bb5", coordinates=(50, 40)),
                        EncodedSymbol("chord"),
                        EncodedSymbol(lower_rhythm, "Gb5", coordinates=(50, 54)),
                    ],
                    0,
                )

                upper_group = builder.visual_groups["upper"]
                lower_group = builder.visual_groups["lower"]
                self.assertEqual(upper_group.moment_id, lower_group.moment_id)
                self.assertIsNotNone(upper_group.moment_id)
                self.assertIsNone(upper_group.chord_id)
                self.assertIsNone(lower_group.chord_id)
                self.assertIn(expected_action, upper_group.repair_actions)
                self.assertIn(expected_action, lower_group.repair_actions)

    def test_separate_notes_do_not_share_chord_identity_from_misassigned_stem(
        self,
    ) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        stem_contour = np.array([[[29, 15]], [[31, 15]], [[31, 70]], [[29, 70]]], dtype=np.float32)
        shared_stem = RotatedBoundingBox(cv2.minAreaRect(stem_contour), stem_contour)

        def make_note(
            x: int,
            y: int,
            visual_id: str,
            width: int = 20,
            note_stem: RotatedBoundingBox | None = shared_stem,
        ) -> Note:
            contour = cv2.ellipse2Poly((x, y), (width // 2, 7), 0, 0, 360, 10).reshape(-1, 1, 2)
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
        builder = VisualSidecarBuilder(coordinate_transform, stem_fragments=[shared_stem])
        builder.add_staff_visual_notes(0, notes, [candidate.copy() for candidate in notes])
        builder.add_staff_matches(
            [
                EncodedSymbol("note_16", "D4", coordinates=(20, 60)),
                EncodedSymbol("note_16", "G4", coordinates=(50, 40)),
                EncodedSymbol("note_16", "C4", coordinates=(90, 80)),
            ],
            0,
        )

        self.assertTrue(builder.visual_groups["first"].owned_stem_component_ids)
        self.assertEqual(
            builder.visual_groups["first"].owned_stem_component_ids,
            builder.visual_groups["second"].owned_stem_component_ids,
        )
        groups = {
            group["visual_group_id"]: group for group in builder.to_json_dict()["visual_groups"]
        }
        self.assertEqual(groups["first"]["stem_component_ids"], [])
        self.assertEqual(groups["second"]["stem_component_ids"], [])

    def test_rejoins_horizontally_split_whole_note_chord_heads(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)

        def fragment(visual_id: str, center_x: int, center_y: int, position: int) -> Note:
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
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        lower_whole = EncodedSymbol("note_1", "F3", coordinates=(44, 62))
        upper_whole = EncodedSymbol("note_1", "A3", coordinates=(56, 38))
        sequence_a = EncodedSymbol("note_16", "A5", coordinates=(90, 42))
        sequence_g = EncodedSymbol("note_16", "G5", coordinates=(75, 30))
        builder.add_staff_matches(
            [
                lower_whole,
                EncodedSymbol("chord"),
                upper_whole,
                sequence_a,
                sequence_g,
            ],
            0,
        )

        sidecar = builder.to_json_dict()
        groups = {group["visual_group_id"]: group for group in sidecar["visual_groups"]}

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
        self.assertEqual(sidecar["unmatched_visual_notes"], ["bottom-right", "top-right"])
        self.assertEqual(groups["top-right"]["visual_status"], "diagnostic")
        self.assertEqual(groups["bottom-right"]["visual_status"], "diagnostic")
        self.assertEqual(groups["bottom-left"]["provenance"], "merged_fragments")
        self.assertEqual(groups["top-left"]["provenance"], "merged_fragments")
        self.assertAlmostEqual(groups["bottom-left"]["center"][0], 50, delta=0.5)
        self.assertAlmostEqual(groups["top-left"]["center"][0], 50, delta=0.5)
        self.assertGreater(groups["bottom-left"]["notehead_ellipses"][0]["rx"], 10)
        self.assertGreater(groups["top-left"]["notehead_ellipses"][0]["rx"], 10)
        self.assertEqual(
            builder.matches_by_symbol_id[sequence_a.visual_match_id].visual_id,
            "sequence-a",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[sequence_g.visual_match_id].visual_id,
            "sequence-g",
        )

    def test_split_hollow_heads_are_consolidated_before_cross_stave_matching(
        self,
    ) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
            contour = cv2.ellipse2Poly((80, center_y), (13, 9), 0, 0, 360, 5).reshape(-1, 1, 2)
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
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        for visual_id in ("lower-left", "lower-right", "lower-bass"):
            builder.visual_groups[visual_id].stave_index = 1

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

        builder.add_staff_matches(symbols, 0)

        note_symbols = [symbol for symbol in symbols if symbol.rhythm.startswith("note")]
        matches = [builder.matches_by_symbol_id[symbol.visual_match_id] for symbol in note_symbols]
        self.assertTrue(all(match.visual_id is not None for match in matches))
        self.assertEqual({match.alignment_method for match in matches}, {"structural"})
        matched_groups = [builder.visual_groups[str(match.visual_id)] for match in matches]
        self.assertEqual(len({group.moment_id for group in matched_groups}), 1)
        self.assertEqual(len({group.chord_id for group in matched_groups}), 2)
        self.assertEqual(
            builder.visual_groups["upper-left"].provenance,
            "merged_fragments",
        )
        self.assertEqual(
            builder.visual_groups["lower-left"].provenance,
            "merged_fragments",
        )

    def test_displaced_second_stays_in_stemless_whole_note_chord_moment(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(160, 120),
            autocrop_box=(0, 0, 160, 120),
            cropped_size=(160, 120),
            resized_size=(160, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(160, 120),
        )
        image = np.full((120, 160), 255, dtype=np.uint8)

        def hollow_note(visual_id: str, center_x: int, center_y: int, position: int) -> Note:
            contour = cv2.ellipse2Poly((center_x, center_y), (12, 8), 0, 0, 360, 5).reshape(
                -1, 1, 2
            )
            cv2.ellipse(image, (center_x, center_y), (12, 8), 0, 0, 360, 2)
            return Note(
                BoundingEllipse(((center_x, center_y), (24, 16), 0), contour, position),
                position=position,
                stem=None,
                stem_direction=None,
                visual_id=visual_id,
            )

        top = hollow_note("top", 60, 35, 7)
        middle = hollow_note("middle", 60, 55, 4)
        displaced_bottom = hollow_note("displaced-bottom", 38, 56, 3)
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        builder.add_staff_visual_notes(
            0,
            [top, middle, displaced_bottom],
            [top.copy(), middle.copy(), displaced_bottom.copy()],
        )
        top_symbol = EncodedSymbol("note_1", "D5", position="upper")
        middle_symbol = EncodedSymbol("note_1", "A4", position="upper")
        bottom_symbol = EncodedSymbol("note_1", "G4", position="upper")

        builder.add_staff_matches(
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
            builder.visual_groups[builder.matches_by_symbol_id[symbol.visual_match_id].visual_id]
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
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(140, 120),
            autocrop_box=(0, 0, 140, 120),
            cropped_size=(140, 120),
            resized_size=(140, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 120),
        )

        def make_note(x: int, y: int, position: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((x, y), (10, 7), -20, 0, 360, 5).reshape(-1, 1, 2)
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
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        builder.visual_groups["bass-anchor"].stave_index = 1
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

        builder.add_staff_matches(symbols, 0)

        matched_groups = [
            builder.visual_groups[builder.matches_by_symbol_id[symbol.visual_match_id].visual_id]
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
        self.assertEqual(builder.unmatched_visual_notes, set())

    def test_dense_filled_chords_are_not_treated_as_split_whole_note_fragments(
        self,
    ) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
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

        notes = [
            make_note(30, 35, "first-top"),
            make_note(30, 55, "first-bottom"),
            make_note(50, 35, "second-top"),
            make_note(50, 55, "second-bottom"),
        ]
        builder = VisualSidecarBuilder(coordinate_transform)
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        symbols = [
            EncodedSymbol("note_16", "F5", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_16", "C5", position="upper"),
            EncodedSymbol("note_16", "F5", position="upper"),
            EncodedSymbol("chord"),
            EncodedSymbol("note_16", "C5", position="upper"),
        ]

        builder.add_staff_matches(symbols, 0)

        note_symbols = [symbol for symbol in symbols if symbol.rhythm.startswith("note")]
        self.assertEqual(
            [
                builder.matches_by_symbol_id[symbol.visual_match_id].visual_id
                for symbol in note_symbols
            ],
            ["first-top", "first-bottom", "second-top", "second-bottom"],
        )
        self.assertEqual(builder.unmatched_visual_notes, set())

    def test_discards_small_notehead_fragment_that_duplicates_a_detected_stem(
        self,
    ) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
        shared_stem = RotatedBoundingBox(cv2.minAreaRect(shared_stem_contour), shared_stem_contour)
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

        full_first_contour = cv2.ellipse2Poly((40, 60), (10, 7), 0, 0, 360, 10).reshape(-1, 1, 2)
        fragment_contour = np.array(
            [[[58, 58]], [[63, 58]], [[63, 62]], [[58, 62]]], dtype=np.float32
        )
        full_second_contour = cv2.ellipse2Poly((110, 60), (10, 7), 0, 0, 360, 10).reshape(-1, 1, 2)
        notes = [
            note("full-first", 40, shared_stem, full_first_contour),
            note("fragment", 61, shared_stem, fragment_contour),
            note("full-second", 110, separate_stem, full_second_contour),
        ]
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        builder.add_staff_visual_notes(0, notes, [candidate.copy() for candidate in notes])
        first_symbol = EncodedSymbol("note_16", "B3", coordinates=(40, 60))
        second_symbol = EncodedSymbol("note_16", "D4", coordinates=(110, 60))

        builder.add_staff_matches([first_symbol, second_symbol], 0)

        sidecar = builder.to_json_dict()
        groups = {group["visual_group_id"]: group for group in sidecar["visual_groups"]}
        self.assertEqual(set(groups), {"full-first", "fragment", "full-second"})
        self.assertEqual(groups["fragment"]["visual_status"], "diagnostic")
        self.assertIn("suspected_duplicate", groups["fragment"]["repair_actions"])
        self.assertEqual(sidecar["unmatched_visual_notes"], ["fragment"])
        self.assertEqual(
            builder.matches_by_symbol_id[first_symbol.visual_match_id].visual_id,
            "full-first",
        )
        self.assertEqual(
            builder.matches_by_symbol_id[second_symbol.visual_match_id].visual_id,
            "full-second",
        )

    def test_split_stem_across_displaced_noteheads_is_exported_as_one_chord(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
        upper_stem = RotatedBoundingBox(cv2.minAreaRect(upper_stem_contour), upper_stem_contour)
        lower_stem = RotatedBoundingBox(cv2.minAreaRect(lower_stem_contour), lower_stem_contour)

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
        builder = VisualSidecarBuilder(
            coordinate_transform, stem_fragments=[upper_stem, lower_stem]
        )
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        builder.add_staff_matches(
            [
                EncodedSymbol("note_8", "C6", coordinates=(60, 30)),
                EncodedSymbol("note_8", "G5", coordinates=(60, 50)),
                EncodedSymbol("note_8", "F5", coordinates=(40, 60)),
            ],
            0,
        )

        groups = {
            group["visual_group_id"]: group for group in builder.to_json_dict()["visual_groups"]
        }
        chord_component_ids = groups["vnote-top"]["stem_component_ids"]

        self.assertTrue(chord_component_ids)
        self.assertEqual(groups["vnote-middle"]["stem_component_ids"], chord_component_ids)
        self.assertEqual(groups["vnote-displaced"]["stem_component_ids"], chord_component_ids)
