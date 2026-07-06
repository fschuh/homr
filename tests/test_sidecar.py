import unittest

import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import Note
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.sidecar import PreprocessingMetadata, SidecarCollector
from homr.transformer.vocabulary import EncodedSymbol


class TestSidecar(unittest.TestCase):
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

    def test_musicxml_ids_are_recorded_in_sidecar(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        collector = SidecarCollector(metadata)
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
        xml = generate_xml(XmlGeneratorArguments(), [[symbol]], "", sidecar=collector)

        xml_ids = self._musicxml_note_ids(xml)
        sidecar = collector.to_json_dict()
        sidecar_ids = [note["musicxml_id"] for note in sidecar["notes"]]
        linked_ids = sidecar["visual_groups"][0]["musicxml_ids"]

        self.assertEqual(xml_ids, sidecar_ids)
        self.assertEqual(xml_ids, linked_ids)
        self.assertEqual(sidecar["unmatched_musicxml_notes"], [])
        self.assertEqual(sidecar["unmatched_visual_notes"], [])

    def test_notehead_fitted_ellipse_is_recorded_in_source_coordinates(self) -> None:
        metadata = PreprocessingMetadata(
            source_image_size=(1000, 800),
            autocrop_box=(100, 50, 400, 300),
            cropped_size=(400, 300),
            resized_size=(800, 600),
            resize_scale=(2.0, 2.0),
            prediction_size=(400, 300),
        )
        collector = SidecarCollector(metadata)
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
        collector = SidecarCollector(metadata)
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
        collector = SidecarCollector(metadata, notehead_mask=mask)
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
        collector = SidecarCollector(metadata)
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

    def test_musicxml_without_sidecar_has_no_generated_ids(self) -> None:
        symbol = EncodedSymbol("note_4", "C4", "_", "_", "_", "upper")
        xml = generate_xml(XmlGeneratorArguments(), [[symbol]], "")

        self.assertEqual(self._musicxml_note_ids(xml), [])

    def test_split_stem_fragments_are_combined_for_sidecar_geometry_only(self) -> None:
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
        collector = SidecarCollector(metadata, [lower_fragment, upper_fragment])
        original = Note(
            notehead,
            position=4,
            stem=lower_fragment,
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()

        collector.add_staff_visual_notes(0, [original], [transformed])
        contour = collector.to_json_dict()["visual_groups"][0]["stem_contours"][0]
        height = max(point[1] for point in contour) - min(point[1] for point in contour)

        self.assertGreater(height, lower_fragment.size[1] + 15)
        self.assertEqual(original.stem, lower_fragment)

    def test_sidecar_stem_merge_rejects_wide_beam_fragment(self) -> None:
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
        collector = SidecarCollector(metadata, [beam])
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

    def test_sidecar_stem_merge_does_not_chain_sideways(self) -> None:
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
        collector = SidecarCollector(metadata, [near, drift])
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

    def test_sidecar_replaces_tiny_bad_seed_with_nearby_vertical_seed(self) -> None:
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
        collector = SidecarCollector(metadata, [tiny_bad_seed, better_seed, continuation])
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
