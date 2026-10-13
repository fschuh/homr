# ruff: noqa: S101

import json
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from homr.model import Staff, StaffPoint
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.staff_canvas_transform import StaffCanvasTransform
from homr.staff_dewarping import PiecewiseAffineTransform, StaffDewarping
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from homr.visual_sidecar.rests import (
    BLOCK,
    GLYPH,
    SUPPORTED,
    UNSUPPORTED,
    UNVERIFIED,
    InkBlob,
    RestVerifier,
    SegmentationMasks,
    extract_ink_blobs,
    matches_shape,
)

UNIT = 20
UPPER = [100.0, 120.0, 140.0, 160.0, 180.0]
LOWER = [300.0, 320.0, 340.0, 360.0, 380.0]
WIDTH, HEIGHT = 600, 480


def blank_page() -> NDArray:
    return np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)


def draw_staff(image: NDArray, lines: list[float]) -> NDArray:
    """Print the staff lines and return the staff mask segnet would give for them."""
    mask = np.zeros_like(image)
    for y in lines:
        cv2.line(image, (0, int(y)), (WIDTH - 1, int(y)), 0, 2)
        cv2.line(mask, (0, int(y)), (WIDTH - 1, int(y)), 1, 3)
    return mask


def draw_eighth_rest(image: NDArray, x: int, top: int) -> None:
    """A dot and a slanted stem two staff spaces tall, crossing staff lines."""
    cv2.circle(image, (x, top + 5), 5, 0, -1)
    cv2.line(image, (x + 5, top + 5), (x - 3, top + 2 * UNIT), 0, 2)


def masks_for(staff_mask: NDArray, stems_rest: NDArray | None = None) -> SegmentationMasks:
    empty = np.zeros_like(staff_mask)
    return SegmentationMasks(
        staff=staff_mask,
        stems_rest=stems_rest if stems_rest is not None else empty,
        notehead=empty,
        clefs_keys=empty,
        symbols=empty,
    )


def identity_canvas() -> StaffCanvasTransform:
    return StaffCanvasTransform(
        region_top_left=(0.0, 0.0),
        region_scaling=1.0,
        dewarp=StaffDewarping(None),
        crop_top_left=(0.0, 0.0),
        canvas_scaling=(1.0, 1.0),
        canvas_y_offset=0.0,
    )


def staff_of(*systems: list[float]) -> Staff:
    lines = [y for system in systems for y in system]
    return Staff([StaffPoint(float(x), list(lines), 0.0) for x in range(0, WIDTH, 10)])


def lines_at_x(staff: Staff, x: float, staff_index: int) -> list[float]:
    return UPPER if staff_index == 0 else LOWER


def rest(rhythm: str, x: float | None, position: str = "upper") -> EncodedSymbol:
    coordinates = None if x is None else (x, 140.0 if position == "upper" else 340.0)
    return EncodedSymbol(rhythm, "_", position=position, coordinates=coordinates)


class TestStaffCanvasTransform(unittest.TestCase):
    def transform(self, dewarp: StaffDewarping) -> StaffCanvasTransform:
        return StaffCanvasTransform(
            region_top_left=(100.0, 50.0),
            region_scaling=0.8,
            dewarp=dewarp,
            crop_top_left=(10.0, 20.0),
            canvas_scaling=(1.2, 0.9),
            canvas_y_offset=7.0,
        )

    def test_round_trip_through_the_affine_steps(self) -> None:
        transform = self.transform(StaffDewarping(None))
        for point in [(150.0, 80.0), (420.5, 133.25), (101.0, 51.0)]:
            back = transform.to_prediction(transform.to_canvas(point))
            assert back is not None
            self.assertAlmostEqual(back[0], point[0], delta=0.05)
            self.assertAlmostEqual(back[1], point[1], delta=0.05)

    def test_round_trip_through_a_piecewise_affine_dewarp(self) -> None:
        grid = np.array([[x, y] for x in range(-50, 600, 50) for y in range(-50, 300, 50)])
        warped = grid + np.column_stack([np.zeros(len(grid)), 3 * np.sin(grid[:, 0] / 90)])
        tform = PiecewiseAffineTransform()
        tform.estimate(grid, warped)
        transform = self.transform(StaffDewarping(tform))
        for point in [(180.0, 90.0), (300.0, 120.0), (450.0, 160.0)]:
            back = transform.to_prediction(transform.to_canvas(point))
            assert back is not None
            self.assertAlmostEqual(back[0], point[0], delta=0.1)
            self.assertAlmostEqual(back[1], point[1], delta=0.1)

    def test_a_canvas_point_inside_a_dewarp_seam_has_no_preimage(self) -> None:
        class Seam:
            def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
                return (point[0] + 40.0, point[1]) if point[0] >= 50 else point

        transform = StaffCanvasTransform(
            region_top_left=(0.0, 0.0),
            region_scaling=1.0,
            dewarp=StaffDewarping(Seam()),  # type: ignore[arg-type]
            crop_top_left=(0.0, 0.0),
            canvas_scaling=(1.0, 1.0),
            canvas_y_offset=0.0,
        )
        self.assertIsNone(transform.to_prediction((70.0, 10.0)))
        self.assertEqual(transform.to_prediction((30.0, 10.0)), (30.0, 10.0))


class TestInkBlobs(unittest.TestCase):
    def blobs(
        self, image: NDArray, masks: SegmentationMasks, note_stems: NDArray | None = None
    ) -> list[InkBlob]:
        stems = note_stems if note_stems is not None else np.zeros_like(image)
        return extract_ink_blobs(image, masks, stems, (0, 0, WIDTH, 260), UNIT)

    def blob_near(self, blobs: list[InkBlob], x: float) -> InkBlob:
        return min(blobs, key=lambda blob: abs(blob.center[0] - x))

    def test_a_rest_crossing_staff_lines_stays_one_blob(self) -> None:
        image = blank_page()
        staff_mask = draw_staff(image, UPPER)
        draw_eighth_rest(image, 200, 120)

        blobs = self.blobs(image, masks_for(staff_mask))

        rest_blob = self.blob_near(blobs, 200)
        self.assertGreaterEqual(rest_blob.height, 1.8 * UNIT)
        self.assertTrue(matches_shape(rest_blob, GLYPH, UNIT, UPPER))
        self.assertTrue(all(blob.height < UNIT for blob in blobs if blob is not rest_blob))

    def test_a_rest_touching_a_beam_is_cut_free_of_it(self) -> None:
        image = blank_page()
        staff_mask = draw_staff(image, UPPER)
        cv2.rectangle(image, (240, 170), (360, 178), 0, -1)
        draw_eighth_rest(image, 300, 130)

        rest_blob = self.blob_near(self.blobs(image, masks_for(staff_mask)), 300)

        self.assertLess(rest_blob.width, UNIT)
        self.assertTrue(matches_shape(rest_blob, GLYPH, UNIT, UPPER))

    def test_printed_lines_thicker_than_the_staff_mask_do_not_join_symbols(self) -> None:
        image = blank_page()
        thin_mask = np.zeros_like(image)
        for y in UPPER:
            cv2.line(image, (0, int(y)), (WIDTH - 1, int(y)), 0, 3)
            cv2.line(thin_mask, (0, int(y)), (WIDTH - 1, int(y)), 1, 1)
        draw_eighth_rest(image, 200, 120)
        cv2.circle(image, (225, 160), 6, 0, -1)

        rest_blob = self.blob_near(self.blobs(image, masks_for(thin_mask)), 200)

        self.assertLess(rest_blob.width, UNIT)

    def test_a_flag_on_a_note_stem_touches_it(self) -> None:
        image = blank_page()
        staff_mask = draw_staff(image, UPPER)
        cv2.line(image, (100, 90), (100, 165), 0, 2)
        cv2.line(image, (101, 92), (112, 125), 0, 3)
        stems = np.zeros_like(image)
        cv2.line(stems, (100, 90), (100, 165), 1, 2)

        flag = self.blob_near(self.blobs(image, masks_for(staff_mask, stems), stems), 108)

        self.assertTrue(flag.touches_stem)

    def test_a_rest_beside_a_barline_does_not_touch_a_note_stem(self) -> None:
        image = blank_page()
        staff_mask = draw_staff(image, UPPER)
        cv2.line(image, (60, 100), (60, 180), 0, 2)
        barline = np.zeros_like(image)
        cv2.line(barline, (60, 100), (60, 180), 1, 2)
        draw_eighth_rest(image, 72, 120)

        rest_blob = self.blob_near(self.blobs(image, masks_for(staff_mask, barline)), 72)

        self.assertFalse(rest_blob.touches_stem)

    def test_rests_marked_as_stems_by_segnet_are_still_unexplained(self) -> None:
        """On scanned pages segnet puts rests in its stems_rests class too."""
        image = blank_page()
        staff_mask = draw_staff(image, UPPER)
        draw_eighth_rest(image, 200, 120)
        stems_rest = (image < 128).astype(np.uint8) & (1 - staff_mask)

        rest_blob = self.blob_near(self.blobs(image, masks_for(staff_mask, stems_rest)), 200)

        self.assertTrue(matches_shape(rest_blob, GLYPH, UNIT, UPPER))


class TestRestShapes(unittest.TestCase):
    def blob(self, left: int, top: int, right: int, bottom: int, fill: float = 1.0) -> InkBlob:
        area = int((right - left) * (bottom - top) * fill)
        return InkBlob(left, top, right, bottom, area, False)

    def test_a_block_must_hang_from_or_sit_on_a_line(self) -> None:
        on_line = self.blob(200, 120, 224, 130)
        floating = self.blob(200, 126, 224, 134)
        self.assertTrue(matches_shape(on_line, BLOCK, UNIT, UPPER))
        self.assertFalse(matches_shape(floating, BLOCK, UNIT, UPPER))

    def test_a_glyph_touching_a_note_stem_is_a_flag_not_a_rest(self) -> None:
        free = InkBlob(200, 120, 214, 160, 224, False)
        flag = InkBlob(200, 120, 214, 160, 224, True)
        self.assertTrue(matches_shape(free, GLYPH, UNIT, UPPER))
        self.assertFalse(matches_shape(flag, GLYPH, UNIT, UPPER))

    def test_a_glyph_is_taller_than_a_staff_space_and_not_too_wide(self) -> None:
        self.assertTrue(matches_shape(self.blob(200, 120, 214, 160, 0.4), GLYPH, UNIT, UPPER))
        self.assertFalse(matches_shape(self.blob(200, 120, 260, 160, 0.4), GLYPH, UNIT, UPPER))
        self.assertFalse(matches_shape(self.blob(200, 130, 214, 140, 0.4), GLYPH, UNIT, UPPER))


class TestRestVerifier(unittest.TestCase):
    def verify(
        self,
        image: NDArray,
        masks: SegmentationMasks | None,
        symbols: list[EncodedSymbol],
        staff: Staff | None = None,
    ) -> dict[str, tuple[str, str]]:
        verifier = RestVerifier(image, masks, lines_at_x)
        verdicts = verifier.verify_staff(
            symbols, 0, staff if staff is not None else staff_of(UPPER), identity_canvas()
        )
        by_id = {verdict.symbol_id: verdict for verdict in verdicts}
        return {
            f"{symbol.rhythm}@{symbol.coordinates}": (
                by_id[symbol.visual_match_id].status,
                by_id[symbol.visual_match_id].reason,
            )
            for symbol in symbols
            if symbol.visual_match_id in by_id
        }

    def page_with_rest_at(self, *xs: int) -> tuple[NDArray, SegmentationMasks]:
        image = blank_page()
        staff_mask = draw_staff(image, UPPER)
        staff_mask |= draw_staff(image, LOWER)
        for x in xs:
            draw_eighth_rest(image, x, 120)
        return image, masks_for(staff_mask)

    def test_a_printed_rest_is_supported_and_a_missing_one_is_not(self) -> None:
        image, masks = self.page_with_rest_at(200)
        printed, invented = rest("rest_8", 205), rest("rest_8", 420)

        result = self.verify(image, masks, [printed, invented])

        self.assertEqual(
            list(result.values()),
            [(SUPPORTED, "rest_shaped_ink"), (UNSUPPORTED, "no_rest_shaped_ink")],
        )

    def test_attention_may_drift_three_staff_spaces(self) -> None:
        image, masks = self.page_with_rest_at(200)
        near, far = rest("rest_8", 200 + 2.5 * UNIT), rest("rest_8", 200 + 3.5 * UNIT)

        self.assertEqual(
            self.verify(image, masks, [near])[f"rest_8@{near.coordinates}"][0], SUPPORTED
        )
        self.assertEqual(
            self.verify(image, masks, [far])[f"rest_8@{far.coordinates}"][0], UNSUPPORTED
        )

    def test_one_printed_rest_supports_only_one_token(self) -> None:
        image, masks = self.page_with_rest_at(200)
        first, second = rest("rest_8", 202), rest("rest_8", 215)

        result = self.verify(image, masks, [first, second])

        self.assertEqual(
            sorted(result.values()),
            [(SUPPORTED, "rest_shaped_ink"), (UNSUPPORTED, "ink_claimed_by_another_rest")],
        )

    def test_a_rest_displaced_below_its_staff_is_found(self) -> None:
        """A second voice's rest printed three staff spaces below the bottom line."""
        image, masks = self.page_with_rest_at()
        draw_eighth_rest(image, 200, 225)

        result = self.verify(image, masks, [rest("rest_8", 200)])

        self.assertEqual(list(result.values()), [(SUPPORTED, "rest_shaped_ink")])

    def test_a_rest_nearer_the_other_staff_belongs_to_it(self) -> None:
        image, masks = self.page_with_rest_at()
        draw_eighth_rest(image, 200, 245)
        staff = staff_of(UPPER, LOWER)

        upper = self.verify(image, masks, [rest("rest_8", 200, "upper")], staff)
        lower = self.verify(image, masks, [rest("rest_8", 200, "lower")], staff)

        self.assertEqual(list(upper.values())[0][0], UNSUPPORTED)
        self.assertEqual(list(lower.values())[0][0], SUPPORTED)

    def test_a_block_supports_a_glyph_rest_only_as_a_fallback(self) -> None:
        image, masks = self.page_with_rest_at()
        cv2.rectangle(image, (200, 120), (224, 129), 0, -1)

        result = self.verify(image, masks, [rest("rest_4.", 210)])

        self.assertEqual(list(result.values()), [(SUPPORTED, "rest_shaped_ink")])

    def test_a_block_far_from_a_glyph_rest_does_not_support_it(self) -> None:
        """A block is searched as far for a glyph token as a glyph would be."""
        image, masks = self.page_with_rest_at()
        cv2.rectangle(image, (290, 120), (314, 129), 0, -1)

        result = self.verify(image, masks, [rest("rest_4.", 220)])

        self.assertEqual(list(result.values()), [(UNSUPPORTED, "no_rest_shaped_ink")])

    def test_a_wide_whole_rest_is_not_cut_away_as_a_beam(self) -> None:
        """Some engravings print whole rests about 1.7 staff spaces wide."""
        image, masks = self.page_with_rest_at()
        cv2.rectangle(image, (283, 120), (317, 129), 0, -1)

        result = self.verify(image, masks, [rest("rest_1", 300)])

        self.assertEqual(list(result.values()), [(SUPPORTED, "rest_shaped_ink")])

    def test_a_pitchless_note_is_never_a_printed_rest(self) -> None:
        image, masks = self.page_with_rest_at(200)
        note = EncodedSymbol("note_8", ".", position="upper", coordinates=(205.0, 140.0))

        result = self.verify(image, masks, [note])

        self.assertEqual(list(result.values()), [(UNSUPPORTED, "note_without_pitch")])

    def test_unverified_reasons(self) -> None:
        image, masks = self.page_with_rest_at(200)
        lone = rest("rest_8", None)
        multi = EncodedSymbol("rest_2m", "_", position="upper", coordinates=(205.0, 140.0))

        self.assertEqual(
            list(self.verify(image, masks, [lone]).values()),
            [(UNVERIFIED, "no_attention_coordinates")],
        )
        self.assertEqual(
            list(self.verify(image, masks, [multi]).values()),
            [(UNVERIFIED, "multi_measure_rest")],
        )
        self.assertEqual(
            list(self.verify(image, None, [rest("rest_8", 205)]).values()),
            [(UNVERIFIED, "no_segmentation")],
        )

        verifier = RestVerifier(image, masks, lines_at_x)
        (no_staff,) = verifier.verify_staff([rest("rest_8", 205)], 0, None, identity_canvas())
        self.assertEqual((no_staff.status, no_staff.reason), (UNVERIFIED, "no_staff_geometry"))
        (no_canvas,) = verifier.verify_staff([rest("rest_8", 205)], 0, staff_of(UPPER), None)
        self.assertEqual((no_canvas.status, no_canvas.reason), (UNVERIFIED, "no_staff_geometry"))

        def no_lines(staff: Staff, x: float, staff_index: int) -> list[float]:
            raise ValueError("Physical staff 0 has no complete grid points")

        unlined = RestVerifier(image, masks, no_lines)
        (verdict,) = unlined.verify_staff(
            [rest("rest_8", 205)], 0, staff_of(UPPER), identity_canvas()
        )
        self.assertEqual((verdict.status, verdict.reason), (UNVERIFIED, "no_staff_lines"))

    def test_a_symbol_without_attention_is_placed_between_its_neighbours(self) -> None:
        image, masks = self.page_with_rest_at(200)
        before = EncodedSymbol("note_8", "C5", position="upper", coordinates=(150.0, 120.0))
        lost = rest("rest_8", None)
        after = EncodedSymbol("note_8", "D5", position="upper", coordinates=(250.0, 120.0))

        verifier = RestVerifier(image, masks, lines_at_x)
        (verdict,) = verifier.verify_staff(
            [before, lost, after], 0, staff_of(UPPER), identity_canvas()
        )

        self.assertTrue(verdict.position_estimated)
        self.assertEqual(verdict.status, SUPPORTED)


class TestRestRecords(unittest.TestCase):
    def test_every_written_rest_is_identified_and_reported(self) -> None:
        coordinate_transform = PredictionCoordinateTransform(
            source_image_size=(100, 100),
            autocrop_box=(0, 0, 100, 100),
            cropped_size=(100, 100),
            resized_size=(100, 100),
            resize_scale=(1.0, 1.0),
            prediction_size=(100, 100),
        )
        builder = VisualSidecarBuilder(coordinate_transform)
        voice = [
            EncodedSymbol("clef_G2", position="upper"),
            EncodedSymbol("note_4", "C4", "_", "_", "_", "upper"),
            EncodedSymbol("rest_4", "_", "_", "_", "_", "upper"),
            EncodedSymbol("note_2", ".", "_", "_", "_", "upper"),
            EncodedSymbol("barline"),
        ]

        xml = generate_xml(XmlGeneratorArguments(), [voice], "", visual_sidecar=builder)
        sidecar = json.loads(json.dumps(builder.to_json_dict()))

        root = ET.fromstring(xml.to_string())  # noqa: S314 - generated by the test itself
        written = [note.get("id") for note in root.iter("note") if note.find("rest") is not None]
        reported = [record["rest_id"] for record in sidecar["rest_verification"]["rests"]]
        self.assertEqual(written, ["homr-rest-1", "homr-rest-2"])
        self.assertEqual(reported, written)
        self.assertEqual(
            {record["status"] for record in sidecar["rest_verification"]["rests"]}, {UNVERIFIED}
        )
        self.assertEqual([note["musicxml_id"] for note in sidecar["notes"]], ["homr-note-1"])

    def test_no_rest_ids_without_a_visual_sidecar(self) -> None:
        voice = [EncodedSymbol("rest_4", "_", "_", "_", "_", "upper"), EncodedSymbol("barline")]
        xml = generate_xml(XmlGeneratorArguments(), [voice], "")
        self.assertNotIn('id="homr-rest', xml.to_string())
