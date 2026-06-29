import unittest

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

    def test_musicxml_without_sidecar_has_no_generated_ids(self) -> None:
        symbol = EncodedSymbol("note_4", "C4", "_", "_", "_", "upper")
        xml = generate_xml(XmlGeneratorArguments(), [[symbol]], "")

        self.assertEqual(self._musicxml_note_ids(xml), [])

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
