import unittest

import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.model import Note
from homr.segmentation.config import model_name as segmentation_model_name
from homr.transformer.configs import model_name as transformer_model_name
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from homr.visual_sidecar.models import PRODUCER_NAME


def _minimal_sidecar() -> dict:
    coordinate_transform = PredictionCoordinateTransform(
        source_image_size=(100, 100),
        autocrop_box=(0, 0, 100, 100),
        cropped_size=(100, 100),
        resized_size=(100, 100),
        resize_scale=(1.0, 1.0),
        prediction_size=(100, 100),
    )
    notehead = BoundingEllipse(((50, 50), (12, 10), 0), np.array([[44, 45], [56, 55]]))
    stem = RotatedBoundingBox(((56, 44), (2, 14), 0), np.array([[55, 37], [57, 51]]))
    builder = VisualSidecarBuilder(coordinate_transform, [stem])
    original = Note(notehead, position=4, stem=stem, stem_direction=None, visual_id="vnote-1")
    builder.add_staff_visual_notes(0, [original], [original.copy()])
    return builder.to_json_dict()


class TestSidecarProducer(unittest.TestCase):
    """The producer block identifies what produced a sidecar's geometry.

    A sidecar outlives the checkout that wrote it, so the block has to be readable on its
    own. These assertions pin the two properties that make it so.
    """

    def test_producer_names_the_software_rather_than_relying_on_the_version(self) -> None:
        # The version is derived from the fork's visual/v* tags with the namespace
        # stripped, so it reoccupies upstream homr's release numbering: "0.1.0" is
        # indistinguishable from upstream's v0.1.0, and an untagged build reports
        # "0.0.0". Only the name separates a fork sidecar from a stock homr one.
        producer = _minimal_sidecar()["producer"]

        self.assertEqual(producer["name"], PRODUCER_NAME)
        self.assertNotEqual(producer["name"], "homr")
        self.assertTrue(producer["version"])

    def test_producer_reports_the_checkpoints_this_build_actually_runs(self) -> None:
        # Reading the same constants the inference code reads is the point: a copied
        # literal would keep reporting the old checkpoint after a model bump, and the
        # weights are what decide the geometry recorded in this file.
        models = _minimal_sidecar()["producer"]["models"]

        self.assertEqual(models["transformer"], transformer_model_name)
        self.assertEqual(models["segmentation"], segmentation_model_name)
