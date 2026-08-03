import unittest

import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import Note
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from tests.visual_sidecar_helpers import musicxml_note_ids


class TestVisualSidecarBuilderStems(unittest.TestCase):
    def test_musicxml_without_visual_sidecar_has_no_generated_ids(self) -> None:
        symbol = EncodedSymbol("note_4", "C4", "_", "_", "_", "upper")
        xml = generate_xml(XmlGeneratorArguments(), [[symbol]], "")

        self.assertEqual(musicxml_note_ids(xml), [])

    def test_split_stem_fragments_are_combined_for_visual_sidecar_geometry_only(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        notehead = BoundingEllipse(((50, 50), (12, 10), 0), np.array([[44, 45], [56, 55]]))
        lower_fragment = RotatedBoundingBox(((56, 44), (2, 14), 0), np.array([[55, 37], [57, 51]]))
        upper_fragment = RotatedBoundingBox(((56, 18), (2, 24), 0), np.array([[55, 6], [57, 30]]))
        builder = VisualSidecarBuilder(coordinate_transform, [lower_fragment, upper_fragment])
        original = Note(
            notehead,
            position=4,
            stem=lower_fragment,
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()

        builder.add_staff_visual_notes(0, [original], [transformed])
        visual_sidecar = builder.to_json_dict()
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
        coordinate_transform = PredictionCoordinateTransform(
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
        builder = VisualSidecarBuilder(coordinate_transform, [beam])
        original = Note(
            notehead,
            position=4,
            stem=stem,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        contour = builder.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        xs = [point[0] for point in contour]

        self.assertLess(max(xs) - min(xs), 10)
        self.assertEqual(original.stem, stem)

    def test_visual_sidecar_stem_merge_does_not_chain_sideways(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
        builder = VisualSidecarBuilder(coordinate_transform, [near, drift])
        original = Note(
            notehead,
            position=4,
            stem=stem,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        contour = builder.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        xs = [point[0] for point in contour]

        self.assertLess(max(xs) - min(xs), 10)
        self.assertEqual(original.stem, stem)

    def test_visual_sidecar_stem_repair_does_not_reach_distant_tempo_mark(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 100), (14, 12), 0), np.array([[43, 94], [57, 106]]))
        detected_stem = RotatedBoundingBox(((57, 95), (2, 10), 0), np.array([[56, 90], [58, 100]]))
        tempo_mark_stem = RotatedBoundingBox(((64, 15), (2, 30), 0), np.array([[63, 0], [65, 30]]))
        builder = VisualSidecarBuilder(coordinate_transform, [detected_stem, tempo_mark_stem])
        original = Note(
            notehead,
            position=4,
            stem=detected_stem,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        group = builder.to_json_dict()["visual_groups"][0]
        stem_ys = [point[1] for point in group["stem_contours"][0]]

        self.assertGreater(min(stem_ys), 80)
        self.assertEqual(
            [stem["contour"] for stem in builder.to_json_dict()["raw_stem_contours"]],
            [
                [[56, 90], [58, 100]],
                [[63, 0], [65, 30]],
            ],
        )
        self.assertEqual(original.stem, detected_stem)

    def test_visual_sidecar_replaces_tiny_bad_seed_with_nearby_vertical_seed(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        tiny_bad_seed = RotatedBoundingBox(((56, 57), (2, 1), 0), np.array([[55, 57], [57, 58]]))
        better_seed = RotatedBoundingBox(((43, 59), (1, 10), 0), np.array([[43, 54], [44, 64]]))
        continuation = RotatedBoundingBox(((43, 82), (2, 36), 0), np.array([[42, 64], [44, 100]]))
        builder = VisualSidecarBuilder(
            coordinate_transform, [tiny_bad_seed, better_seed, continuation]
        )
        original = Note(
            notehead,
            position=4,
            stem=tiny_bad_seed,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        contour = builder.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        xs = [point[0] for point in contour]
        ys = [point[1] for point in contour]

        self.assertLess(max(xs), notehead.center[0])
        self.assertGreater(max(ys) - min(ys), 40)
        self.assertEqual(original.stem, tiny_bad_seed)

    def test_visual_sidecar_repairs_missing_downward_stem_from_nearby_chain(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        seed = RotatedBoundingBox(((43, 66), (1, 16), 0), np.array([[43, 58], [44, 74]]))
        continuation = RotatedBoundingBox(((43, 88), (2, 28), 0), np.array([[42, 74], [44, 102]]))
        builder = VisualSidecarBuilder(coordinate_transform, [seed, continuation])
        original = Note(
            notehead,
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        contour = builder.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        xs = [point[0] for point in contour]
        ys = [point[1] for point in contour]

        self.assertLess(max(xs), notehead.center[0])
        self.assertLessEqual(min(ys), notehead.center[1])
        self.assertGreater(max(ys) - min(ys), 45)
        self.assertIsNone(original.stem)

    def test_visual_sidecar_does_not_borrow_disconnected_peer_stem(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        top_notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        bottom_notehead = BoundingEllipse(((50, 72), (14, 12), 0), np.array([[43, 66], [57, 78]]))
        detected_bottom_stem = RotatedBoundingBox(
            ((43, 72), (2, 12), 0), np.array([[42, 66], [44, 78]])
        )
        bottom_continuation = RotatedBoundingBox(
            ((43, 92), (2, 28), 0), np.array([[42, 78], [44, 106]])
        )
        builder = VisualSidecarBuilder(
            coordinate_transform, [detected_bottom_stem, bottom_continuation]
        )
        top_note = Note(top_notehead, 8, None, None, "vnote-top")
        bottom_note = Note(bottom_notehead, 4, detected_bottom_stem, None, "vnote-bottom")

        builder.add_staff_visual_notes(
            0,
            [top_note, bottom_note],
            [top_note.copy(), bottom_note.copy()],
        )
        groups = {
            group["visual_group_id"]: group for group in builder.to_json_dict()["visual_groups"]
        }

        self.assertEqual(groups["vnote-top"]["stem_contours"], [])
        self.assertNotEqual(groups["vnote-bottom"]["stem_contours"], [])

    def test_visual_sidecar_bridges_disconnected_downward_stem_to_notehead(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        disconnected = RotatedBoundingBox(((43, 72), (1, 28), 0), np.array([[43, 58], [44, 86]]))
        continuation = RotatedBoundingBox(((43, 96), (2, 20), 0), np.array([[42, 86], [44, 106]]))
        builder = VisualSidecarBuilder(coordinate_transform, [disconnected, continuation])
        original = Note(
            notehead,
            position=4,
            stem=disconnected,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        contour = builder.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        ys = [point[1] for point in contour]

        self.assertLessEqual(min(ys), notehead.center[1])
        self.assertGreater(max(ys), 100)
        self.assertEqual(original.stem, disconnected)

    def test_visual_sidecar_extends_short_top_piece_to_downward_chain(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        short_piece = RotatedBoundingBox(((43, 60), (1, 8), 0), np.array([[43, 56], [44, 64]]))
        continuation = RotatedBoundingBox(((43, 86), (2, 40), 0), np.array([[42, 66], [44, 106]]))
        builder = VisualSidecarBuilder(coordinate_transform, [short_piece, continuation])
        original = Note(
            notehead,
            position=4,
            stem=short_piece,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        contour = builder.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        ys = [point[1] for point in contour]

        self.assertLessEqual(min(ys), notehead.center[1])
        self.assertGreater(max(ys) - min(ys), 50)
        self.assertEqual(original.stem, short_piece)

    def test_visual_sidecar_extends_short_top_piece_across_larger_aligned_gap(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(140, 140),
            autocrop_box=(0, 0, 140, 140),
            cropped_size=(140, 140),
            resized_size=(140, 140),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 140),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        short_piece = RotatedBoundingBox(((43, 60), (1, 8), 0), np.array([[43, 56], [44, 64]]))
        continuation = RotatedBoundingBox(((43, 94), (2, 16), 0), np.array([[42, 86], [44, 102]]))
        builder = VisualSidecarBuilder(coordinate_transform, [short_piece, continuation])
        original = Note(
            notehead,
            position=4,
            stem=short_piece,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        contour = builder.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        ys = [point[1] for point in contour]

        self.assertLessEqual(min(ys), notehead.center[1])
        self.assertGreater(max(ys), 100)
        self.assertEqual(original.stem, short_piece)

    def test_visual_sidecar_repairs_missing_upward_stem_from_nearby_chain(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        seed = RotatedBoundingBox(((57, 34), (1, 16), 0), np.array([[57, 26], [58, 42]]))
        continuation = RotatedBoundingBox(((57, 12), (2, 28), 0), np.array([[56, -2], [58, 26]]))
        builder = VisualSidecarBuilder(coordinate_transform, [seed, continuation])
        original = Note(
            notehead,
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        contour = builder.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        xs = [point[0] for point in contour]
        ys = [point[1] for point in contour]

        self.assertGreater(min(xs), notehead.center[0])
        self.assertGreaterEqual(max(ys), notehead.center[1])
        self.assertGreater(max(ys) - min(ys), 45)
        self.assertIsNone(original.stem)

    def test_visual_sidecar_bridges_disconnected_upward_stem_to_notehead(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 120),
            autocrop_box=(0, 0, 120, 120),
            cropped_size=(120, 120),
            resized_size=(120, 120),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 120),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        disconnected = RotatedBoundingBox(((57, 28), (1, 28), 0), np.array([[57, 14], [58, 42]]))
        continuation = RotatedBoundingBox(((57, 4), (2, 20), 0), np.array([[56, -6], [58, 14]]))
        builder = VisualSidecarBuilder(coordinate_transform, [disconnected, continuation])
        original = Note(
            notehead,
            position=4,
            stem=disconnected,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        contour = builder.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        ys = [point[1] for point in contour]

        self.assertLess(min(ys), 0)
        self.assertGreaterEqual(max(ys), notehead.center[1])
        self.assertEqual(original.stem, disconnected)

    def test_visual_sidecar_extends_short_bottom_piece_to_upward_chain(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(140, 140),
            autocrop_box=(0, 0, 140, 140),
            cropped_size=(140, 140),
            resized_size=(140, 140),
            resize_scale=(1.0, 1.0),
            prediction_size=(140, 140),
        )
        notehead = BoundingEllipse(((50, 50), (14, 12), 0), np.array([[43, 44], [57, 56]]))
        short_piece = RotatedBoundingBox(((57, 40), (1, 8), 0), np.array([[57, 36], [58, 44]]))
        continuation = RotatedBoundingBox(((57, 6), (2, 16), 0), np.array([[56, -2], [58, 14]]))
        builder = VisualSidecarBuilder(coordinate_transform, [short_piece, continuation])
        original = Note(
            notehead,
            position=4,
            stem=short_piece,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        contour = builder.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        ys = [point[1] for point in contour]

        self.assertLess(min(ys), 0)
        self.assertGreaterEqual(max(ys), notehead.center[1])
        self.assertGreater(max(ys) - min(ys), 50)
        self.assertEqual(original.stem, short_piece)
