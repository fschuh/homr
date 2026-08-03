import unittest

import cv2
import numpy as np

from homr.bounding_boxes import BoundingEllipse
from homr.model import Note
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder


class TestVisualSidecarBuilderNoteheadGeometry(unittest.TestCase):
    def test_notehead_fitted_ellipse_is_recorded_in_source_coordinates(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(1000, 800),
            autocrop_box=(100, 50, 400, 300),
            cropped_size=(400, 300),
            resized_size=(800, 600),
            resize_scale=(2.0, 2.0),
            prediction_size=(400, 300),
        )
        builder = VisualSidecarBuilder(coordinate_transform)
        original = Note(
            BoundingEllipse(((200, 150), (40, 20), 15), np.array([[180, 140], [220, 160]]), 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()
        builder.add_staff_visual_notes(0, [original], [transformed])

        ellipse = builder.to_json_dict()["visual_groups"][0]["notehead_ellipses"][0]

        self.assertEqual(ellipse["center"], [300, 200])
        self.assertEqual(ellipse["rx"], 20)
        self.assertEqual(ellipse["ry"], 10)
        self.assertEqual(ellipse["angle"], 15)

    def test_notehead_contour_fit_uses_svg_compatible_major_axis_angle(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(300, 300),
            autocrop_box=(0, 0, 300, 300),
            cropped_size=(300, 300),
            resized_size=(300, 300),
            resize_scale=(1.0, 1.0),
            prediction_size=(300, 300),
        )
        contour = cv2.ellipse2Poly((100, 100), (40, 20), -30, 0, 360, 2).reshape(-1, 1, 2)
        builder = VisualSidecarBuilder(coordinate_transform)
        original = Note(
            BoundingEllipse(((100, 100), (80, 40), 0), contour, 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()
        builder.add_staff_visual_notes(0, [original], [transformed])

        ellipse = builder.to_json_dict()["visual_groups"][0]["notehead_ellipses"][0]

        self.assertAlmostEqual(ellipse["rx"], 40, delta=0.5)
        self.assertAlmostEqual(ellipse["ry"], 20, delta=0.5)
        self.assertAlmostEqual(ellipse["angle"], -30, delta=0.5)

    def test_detected_notehead_contour_is_exported_alongside_legacy_polygon(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
        builder = VisualSidecarBuilder(coordinate_transform, notehead_mask=mask)
        original = Note(
            BoundingEllipse(((50, 50), (12, 8), 0), detected_contour, 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        group = builder.to_json_dict()["visual_groups"][0]

        self.assertIn("notehead_contours", group)
        self.assertIn("detected_notehead_contours", group)
        self.assertNotEqual(group["notehead_contours"], group["detected_notehead_contours"])
        self.assertGreaterEqual(len(group["detected_notehead_contours"][0]), 5)

    def test_refined_notehead_contour_robustly_fits_hollow_head_across_staff_line(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        original = Note(
            BoundingEllipse(((50, 50), (20, 12), -25), contour, 3),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        group = builder.to_json_dict()["visual_groups"][0]
        refined = group["refined_notehead_contours"][0]
        xs = [point[0] for point in refined]
        ys = [point[1] for point in refined]

        self.assertAlmostEqual((min(xs) + max(xs)) / 2, 50, delta=2)
        self.assertAlmostEqual((min(ys) + max(ys)) / 2, 50, delta=2)
        self.assertGreater(max(xs) - min(xs), max(ys) - min(ys))

    def test_refined_notehead_contours_do_not_borrow_adjacent_chord_ink(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)

        def make_note(center_y: int, visual_id: str) -> Note:
            contour = cv2.ellipse2Poly((50, center_y), (9, 6), -25, 0, 360, 10).reshape(-1, 1, 2)
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
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        groups = {
            group["visual_group_id"]: group for group in builder.to_json_dict()["visual_groups"]
        }
        top_ys = [point[1] for point in groups["vnote-top"]["refined_notehead_contours"][0]]
        bottom_ys = [point[1] for point in groups["vnote-bottom"]["refined_notehead_contours"][0]]

        self.assertLessEqual(max(top_ys), 50)
        self.assertGreaterEqual(min(bottom_ys), 48)
        self.assertLess((min(top_ys) + max(top_ys)) / 2, 49)
        self.assertGreater((min(bottom_ys) + max(bottom_ys)) / 2, 49)

    def test_refined_notehead_recovers_center_before_fitting_corrupted_anchor(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(120, 100),
            autocrop_box=(0, 0, 120, 100),
            cropped_size=(120, 100),
            resized_size=(120, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(120, 100),
        )
        image = np.full((100, 120), 255, dtype=np.uint8)
        cv2.ellipse(image, ((60, 50), (18, 12), -25), 0, -1)
        cv2.line(image, (25, 50), (85, 50), 0, 1)
        corrupted = cv2.ellipse2Poly((52, 54), (16, 6), 0, 0, 360, 10).reshape(-1, 1, 2)
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        original = Note(
            BoundingEllipse(((52, 54), (32, 12), 0), corrupted, 9),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-shifted",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        group = builder.to_json_dict()["visual_groups"][0]
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
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        image = np.full((100, 100), 255, dtype=np.uint8)
        cv2.ellipse(image, ((50, 50), (18, 12), -25), 0, -1)
        cv2.line(image, (15, 50), (85, 50), 0, 2)
        contour = cv2.ellipse2Poly((50, 50), (9, 6), -25, 0, 360, 10).reshape(-1, 1, 2)
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        original = Note(
            BoundingEllipse(((50, 50), (18, 12), -25), contour, 10),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-on-line",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        refined = builder.to_json_dict()["visual_groups"][0]["refined_notehead_contours"][0]
        xs = [point[0] for point in refined]
        ys = [point[1] for point in refined]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        self.assertGreaterEqual(height, 10)
        self.assertLessEqual(width / height, 1.9)

    def test_refined_filled_notehead_does_not_expand_into_white_staff_gap(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        image = np.full((100, 100), 255, dtype=np.uint8)
        cv2.line(image, (10, 50), (90, 50), 0, 2)
        cv2.line(image, (10, 64), (90, 64), 0, 2)
        cv2.ellipse(image, ((50, 48), (18, 12), -25), 0, -1)
        cv2.line(image, (59, 48), (59, 78), 0, 2)
        contour = cv2.ellipse2Poly((50, 48), (9, 6), -25, 0, 360, 10).reshape(-1, 1, 2)
        builder = VisualSidecarBuilder(coordinate_transform, source_image=image)
        original = Note(
            BoundingEllipse(((50, 48), (18, 12), -25), contour, 11),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-filled-gap",
        )

        builder.add_staff_visual_notes(0, [original], [original.copy()])
        refined = builder.to_json_dict()["visual_groups"][0]["refined_notehead_contours"][0]
        xs = [point[0] for point in refined]
        ys = [point[1] for point in refined]

        self.assertLessEqual(max(xs) - min(xs), 21)
        self.assertLessEqual(max(ys) - min(ys), 16)
        self.assertAlmostEqual((min(ys) + max(ys)) / 2, 48, delta=2)

    def test_split_chord_notehead_keeps_split_geometry_when_mask_is_ambiguous(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
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
        builder = VisualSidecarBuilder(coordinate_transform, notehead_mask=mask)
        original = Note(
            BoundingEllipse(((100, 100), (80, 40), 0), np.array([[60, 80], [140, 120]]), 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-1",
        )
        transformed = original.copy()
        builder.add_staff_visual_notes(0, [original], [transformed])

        ellipse = builder.to_json_dict()["visual_groups"][0]["notehead_ellipses"][0]

        self.assertEqual(ellipse["center"], [100, 100])
        self.assertEqual(ellipse["rx"], 40)
        self.assertEqual(ellipse["ry"], 20)
        self.assertEqual(ellipse["angle"], 0)
        self.assertEqual(original.box.angle, 0)

    def test_low_confidence_chord_ellipse_uses_staff_typical_angle(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(300, 300),
            autocrop_box=(0, 0, 300, 300),
            cropped_size=(300, 300),
            resized_size=(300, 300),
            resize_scale=(1.0, 1.0),
            prediction_size=(300, 300),
        )
        builder = VisualSidecarBuilder(coordinate_transform)
        reliable_contour = cv2.ellipse2Poly((100, 100), (40, 20), -30, 0, 360, 2).reshape(-1, 1, 2)
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

        builder.add_staff_visual_notes(0, [reliable, fallback], [reliable.copy(), fallback.copy()])
        groups = builder.to_json_dict()["visual_groups"]
        fallback_group = next(group for group in groups if group["visual_group_id"] == "vnote-2")

        self.assertAlmostEqual(fallback_group["notehead_ellipses"][0]["angle"], -30, delta=0.5)
        self.assertEqual(fallback.box.angle, 0)

    def test_elongated_horizontal_mask_ellipse_keeps_its_hollow_notehead_angle(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(300, 300),
            autocrop_box=(0, 0, 300, 300),
            cropped_size=(300, 300),
            resized_size=(300, 300),
            resize_scale=(1.0, 1.0),
            prediction_size=(300, 300),
        )
        mask = np.zeros((300, 300), dtype=np.uint8)
        angled_contour = cv2.ellipse2Poly((100, 100), (20, 10), -30, 0, 360, 2).reshape(-1, 1, 2)
        horizontal_contour = cv2.ellipse2Poly((180, 100), (24, 10), 0, 0, 360, 2).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [angled_contour, horizontal_contour], 1)
        builder = VisualSidecarBuilder(coordinate_transform, notehead_mask=mask)
        angled = Note(
            BoundingEllipse(((100, 100), (40, 20), -30), angled_contour, 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-angled",
        )
        horizontal = Note(
            BoundingEllipse(((180, 100), (48, 20), 0), horizontal_contour, 2),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-horizontal",
        )

        notes = [angled, horizontal]
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        groups = {
            group["visual_group_id"]: group for group in builder.to_json_dict()["visual_groups"]
        }

        self.assertAlmostEqual(
            groups["vnote-horizontal"]["notehead_ellipses"][0]["angle"], 0, delta=1
        )

    def test_compact_hollow_image_notehead_keeps_its_horizontal_mask_angle(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(300, 300),
            autocrop_box=(0, 0, 300, 300),
            cropped_size=(300, 300),
            resized_size=(300, 300),
            resize_scale=(1.0, 1.0),
            prediction_size=(300, 300),
        )
        image = np.full((300, 300), 255, dtype=np.uint8)
        mask = np.zeros((300, 300), dtype=np.uint8)
        angled_contour = cv2.ellipse2Poly((100, 100), (20, 10), -30, 0, 360, 2).reshape(-1, 1, 2)
        horizontal_contour = cv2.ellipse2Poly((180, 100), (18, 12), 0, 0, 360, 2).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [angled_contour, horizontal_contour], 1)
        cv2.ellipse(image, ((100, 100), (40, 20), -30), 0, -1)
        cv2.ellipse(image, ((180, 100), (36, 24), 0), 0, 2)
        builder = VisualSidecarBuilder(coordinate_transform, notehead_mask=mask, source_image=image)
        angled = Note(
            BoundingEllipse(((100, 100), (40, 20), -30), angled_contour, 1),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-angled",
        )
        horizontal = Note(
            BoundingEllipse(((180, 100), (36, 24), 0), horizontal_contour, 2),
            position=4,
            stem=None,
            stem_direction=None,
            visual_id="vnote-horizontal",
        )

        notes = [angled, horizontal]
        builder.add_staff_visual_notes(0, notes, [note.copy() for note in notes])
        groups = {
            group["visual_group_id"]: group for group in builder.to_json_dict()["visual_groups"]
        }

        self.assertAlmostEqual(
            groups["vnote-horizontal"]["notehead_ellipses"][0]["angle"], 0, delta=1
        )
