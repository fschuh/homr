import unittest

import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import Note, Staff, StaffPoint
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.note_detection import NoteheadWithStem
from homr.visual_sidecar import PreprocessingMetadata, VisualSidecar
from homr.transformer.vocabulary import EncodedSymbol


class TestVisualSidecar(unittest.TestCase):
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

        self.assertEqual(xml_ids, visual_sidecar_ids)
        self.assertEqual(xml_ids, linked_ids)
        self.assertEqual(visual_sidecar["unmatched_musicxml_notes"], [])
        self.assertEqual(visual_sidecar["unmatched_visual_notes"], [])

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
