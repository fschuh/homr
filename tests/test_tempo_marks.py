# ruff: noqa: S101

import json
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np

from homr.model import MultiStaff, Staff
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from homr.visual_sidecar.models import VisualGroup, VisualMatch
from homr.visual_sidecar.rests import SUPPORTED, RestVerdict
from homr.visual_sidecar.tempo_marks import (
    NOT_WRITTEN,
    PLACED,
    UNIT_PIXELS,
    UNPLACED,
    UNREAD,
    WRITTEN,
    NoteReading,
    SystemBand,
    TempoMark,
    measure_at,
    parse_number,
    read_tempo_marks,
    system_bands,
    write_tempo_directions,
)
from training.transformer.training_vocabulary import read_token_lines

# A mark drawn the size of a 300 dpi page's: digits 25 pixels tall on a staff space of
# UNIT_PIXELS, so the band the reader builds is the image itself.
DIGIT_HEIGHT = 25
BASELINE = 150
SYSTEM_TOP = 200.0


def blank() -> NDArray:
    return np.full((260, 900), 255, dtype=np.uint8)


def draw_note(
    image: NDArray, x: int, value: str = "quarter", dotted: bool = False
) -> tuple[int, int]:
    """Draws a stem-up note whose head sits on the baseline; returns its left and right."""
    head_center = (x + 9, BASELINE - 7)
    hollow = value in ("half", "whole")
    cv2.ellipse(image, head_center, (9, 6), -20, 0, 360, 0, 2 if hollow else -1)
    right = x + 18
    if value != "whole":
        cv2.line(image, (x + 17, BASELINE - 7), (x + 17, BASELINE - 48), 0, 2)
    flags = {"eighth": 1, "16th": 2}.get(value, 0)
    for flag in range(flags):
        top = BASELINE - 48 + 10 * flag
        cv2.line(image, (x + 17, top), (x + 28, top + 14), 0, 3)
        right = x + 29
    if dotted:
        cv2.circle(image, (right + 6, BASELINE - 7), 3, 0, -1)
        right += 10
    return x, right


def draw_equals(image: NDArray, x: int) -> int:
    """Draws an "=" level with the middle of the digits; returns its right edge."""
    middle = BASELINE - 11
    cv2.rectangle(image, (x, middle - 5), (x + 18, middle - 3), 0, -1)
    cv2.rectangle(image, (x, middle + 2), (x + 18, middle + 4), 0, -1)
    return x + 19


def draw_text(image: NDArray, x: int, text: str, baseline: int = BASELINE) -> int:
    """Draws text whose digits are DIGIT_HEIGHT tall; returns its right edge."""
    scale = DIGIT_HEIGHT / 22
    cv2.putText(image, text, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 2)
    (width, _height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    return x + width


def draw_mark(
    image: NDArray, x: int, value: str = "quarter", dotted: bool = False, number: str = "140"
) -> tuple[int, int]:
    """Draws "<note> = <number>"; returns the note's left edge and the number's right."""
    left, right = draw_note(image, x, value, dotted)
    after = draw_equals(image, right + 4)
    return left, draw_text(image, after + 6, number)


class Reader:
    """Stands in for OCR: answers with the given text and keeps every crop it was shown."""

    def __init__(self, text: str = "140") -> None:
        self.text = text
        self.crops: list[NDArray] = []

    def __call__(self, image: NDArray) -> str:
        self.crops.append(image)
        return self.text


def band() -> SystemBand:
    return SystemBand(top=SYSTEM_TOP, bottom=400.0, left=0.0, right=880.0, unit_size=UNIT_PIXELS)


def read(image: NDArray, reader: Reader | None = None) -> list[TempoMark]:
    return read_tempo_marks(image, [band()], reader or Reader())


class TestNumbers(unittest.TestCase):
    def test_a_number_is_read_with_what_ocr_adds_around_it(self) -> None:
        self.assertEqual(parse_number("140"), ("140", 140.0))
        self.assertEqual(parse_number("(116.)"), ("116", 116.0))
        self.assertEqual(parse_number("120）"), ("120", 120.0))
        self.assertEqual(parse_number("128 Courageously"), ("128", 128.0))

    def test_a_range_keeps_its_print_and_plays_its_first_number(self) -> None:
        self.assertEqual(parse_number("64-68"), ("64-68", 64.0))
        self.assertEqual(parse_number("64 – 68"), ("64-68", 64.0))
        self.assertIsNone(parse_number("68-64"))

    def test_only_plausible_tempos_are_numbers(self) -> None:
        self.assertEqual(parse_number("20"), ("20", 20.0))
        self.assertIsNone(parse_number("19"))
        self.assertEqual(parse_number("400"), ("400", 400.0))
        self.assertIsNone(parse_number("401"))
        self.assertIsNone(parse_number("1400"))
        self.assertIsNone(parse_number("7"))
        self.assertIsNone(parse_number("Allegro"))
        self.assertIsNone(parse_number("Op. 25"))


class TestReading(unittest.TestCase):
    def test_each_beat_unit_is_read_from_the_note(self) -> None:
        for value in ("whole", "half", "quarter", "eighth", "16th"):
            with self.subTest(value=value):
                image = blank()
                draw_mark(image, 100, value)
                marks = read(image)
                self.assertEqual(len(marks), 1)
                assert marks[0].note is not None
                self.assertEqual((marks[0].note.beat_unit, marks[0].note.dotted), (value, False))
                self.assertEqual(marks[0].status, UNPLACED)

    def test_a_dot_makes_the_beat_half_as_long_again(self) -> None:
        image = blank()
        draw_mark(image, 100, "quarter", dotted=True)
        marks = read(image)
        assert marks[0].note is not None
        self.assertTrue(marks[0].note.dotted)
        self.assertEqual(marks[0].quarter_bpm, 210.0)

    def test_the_tempo_counts_quarters_per_minute(self) -> None:
        cases = [("half", False, 200.0), ("half", True, 300.0), ("eighth", False, 50.0)]
        for value, dotted, expected in cases:
            mark = TempoMark("100", (0, 0, 1, 1), "100", 100.0, band(), None, UNPLACED, "")
            mark.note = NoteReading(value, dotted, (0, 0, 1, 1))
            self.assertEqual(mark.quarter_bpm, expected)

    def test_only_the_digits_go_to_ocr(self) -> None:
        image = blank()
        draw_text(image, 30, "Lento")
        _left, right = draw_mark(image, 140)
        draw_text(image, right + 22, "Courageously")
        reader = Reader("66")
        marks = read(image, reader)
        self.assertEqual([mark.per_minute for mark in marks], ["66"])
        self.assertEqual(len(reader.crops), 1)
        # Three digits and a margin; the word after them stays out.
        self.assertLess(reader.crops[0].shape[1], 3 * DIGIT_HEIGHT + 2 * 8)

    def test_a_range_is_read_whole(self) -> None:
        image = blank()
        draw_mark(image, 100, number="64-68")
        reader = Reader("64-68")
        marks = read(image, reader)
        self.assertEqual((marks[0].per_minute, marks[0].beats_per_minute), ("64-68", 64.0))
        self.assertGreater(reader.crops[0].shape[1], 4 * DIGIT_HEIGHT)

    def test_an_equivalence_without_a_number_is_no_mark(self) -> None:
        image = blank()
        _left, right = draw_note(image, 100, "quarter", dotted=True)
        after = draw_equals(image, right + 4)
        draw_note(image, after + 6)
        reader = Reader()
        self.assertEqual(read(image, reader), [])
        self.assertEqual(reader.crops, [])

    def test_what_ocr_cannot_read_as_a_tempo_is_no_mark(self) -> None:
        image = blank()
        draw_mark(image, 100)
        self.assertEqual(read(image, Reader("l4O")), [])

    def test_a_number_without_a_note_is_reported_unread(self) -> None:
        image = blank()
        draw_text(image, draw_equals(image, 100) + 6, "140")
        marks = read(image)
        self.assertEqual(
            [(m.status, m.reason) for m in marks], [(UNREAD, "no_note_left_of_equals")]
        )
        self.assertIsNone(marks[0].quarter_bpm)

    def test_a_glyph_that_is_not_a_note_is_reported_unread(self) -> None:
        image = blank()
        cv2.rectangle(image, (100, BASELINE - 30), (110, BASELINE), 0, -1)
        draw_text(image, draw_equals(image, 116) + 6, "140")
        marks = read(image)
        self.assertEqual([(m.status, m.reason) for m in marks], [(UNREAD, "glyph_is_not_a_note")])

    def test_a_filled_head_without_a_stem_is_not_a_whole_note(self) -> None:
        image = blank()
        cv2.ellipse(image, (111, BASELINE - 7), (11, 7), -20, 0, 360, 0, -1)
        draw_text(image, draw_equals(image, 128) + 6, "140")
        marks = read(image)
        self.assertEqual([(m.status, m.reason) for m in marks], [(UNREAD, "glyph_is_not_a_note")])

    def test_text_under_the_mark_is_not_its_note(self) -> None:
        image = blank()
        _left, right = draw_note(image, 100)
        equals_left = right + 30
        # A word below the mark, between the note and the "=".
        cv2.rectangle(image, (right + 8, BASELINE + 6), (equals_left - 2, BASELINE + 26), 0, 2)
        draw_text(image, draw_equals(image, equals_left) + 6, "140")
        marks = read(image)
        assert marks[0].note is not None
        self.assertEqual(marks[0].note.beat_unit, "quarter")

    def test_a_mark_begins_where_its_tempo_text_begins(self) -> None:
        image = blank()
        draw_text(image, 10, "Op.")
        text_end = draw_text(image, 90, "Allegro")
        draw_mark(image, text_end + 12)
        marks = read(image)
        # "Allegro" belongs to the mark; a word further off than words are spaced does not.
        self.assertAlmostEqual(marks[0].box[0], 90, delta=4)

    def test_digits_must_follow_the_equals_sign_closely(self) -> None:
        image = blank()
        _left, right = draw_note(image, 100)
        after = draw_equals(image, right + 4)
        draw_text(image, after + 6 + DIGIT_HEIGHT, "140")
        self.assertEqual(read(image), [])

    def test_a_wide_glyph_after_the_equals_sign_is_no_number(self) -> None:
        image = blank()
        _left, right = draw_note(image, 100)
        after = draw_equals(image, right + 4)
        cv2.rectangle(image, (after + 6, BASELINE - 24), (after + 46, BASELINE), 0, 2)
        reader = Reader()
        self.assertEqual(read(image, reader), [])
        self.assertEqual(reader.crops, [])

    def test_dashes_before_a_mark_are_not_its_text(self) -> None:
        image = blank()
        x = draw_text(image, 40, "accel.") + 8
        for _dash in range(4):
            cv2.rectangle(image, (x, BASELINE - 2), (x + 9, BASELINE), 0, -1)
            x += 16
        note_left, _right = draw_mark(image, x + 6)
        marks = read(image)
        self.assertAlmostEqual(marks[0].box[0], note_left, delta=2)

    def test_marks_are_found_above_each_system_in_source_pixels(self) -> None:
        image = np.full((800, 900), 255, dtype=np.uint8)
        draw_mark(image, 100)
        image[400:660, :] = image[:260, :].copy()
        systems = [
            SystemBand(top=SYSTEM_TOP, bottom=300.0, left=0.0, right=880.0, unit_size=UNIT_PIXELS),
            SystemBand(top=600.0, bottom=700.0, left=0.0, right=880.0, unit_size=UNIT_PIXELS),
        ]
        marks = read_tempo_marks(image, systems, Reader())
        self.assertEqual([mark.system.top for mark in marks], [SYSTEM_TOP, 600.0])
        self.assertAlmostEqual(marks[1].box[1] - marks[0].box[1], 400.0, delta=1.0)

    def test_a_larger_scan_is_read_at_the_same_size(self) -> None:
        image = blank()
        draw_mark(image, 100, "half", dotted=True)
        large = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        system = SystemBand(
            top=2 * SYSTEM_TOP, bottom=800.0, left=0.0, right=1760.0, unit_size=2 * UNIT_PIXELS
        )
        marks = read_tempo_marks(large, [system], Reader())
        assert marks[0].note is not None
        self.assertEqual((marks[0].note.beat_unit, marks[0].note.dotted), ("half", True))
        self.assertAlmostEqual(marks[0].box[0], 200.0, delta=4.0)

    def test_systems_come_from_the_detected_staffs(self) -> None:
        transform = PredictionCoordinateTransform(
            source_image_size=(1200, 1000),
            autocrop_box=(100, 0, 1100, 1000),
            cropped_size=(1000, 1000),
            resized_size=(500, 500),
            resize_scale=(0.5, 0.5),
            prediction_size=(500, 500),
        )
        # A grand staff detected as one staff of ten lines, 10 apart, 40 between staves;
        # its first point lies 2 higher than the rest, as on a slightly rotated page.
        grand = [*range(50, 100, 10), *range(130, 180, 10)]
        grid = [SimpleNamespace(y=[float(y) for y in grand]) for _point in range(3)]
        grid[0].y = [y - 2 for y in grid[0].y]
        piano = cast(
            Staff, SimpleNamespace(min_x=10.0, max_x=400.0, min_y=48.0, max_y=170.0, grid=grid)
        )
        voice = cast(
            Staff,
            SimpleNamespace(
                min_x=12.0,
                max_x=410.0,
                min_y=200.0,
                max_y=240.0,
                grid=[SimpleNamespace(y=[200.0, 210.0, 220.0, 230.0, 240.0])],
            ),
        )
        system = cast(MultiStaff, SimpleNamespace(staffs=[piano, voice]))
        self.assertEqual(
            system_bands([system], transform),
            [SystemBand(top=96.0, bottom=480.0, left=120.0, right=920.0, unit_size=20.0)],
        )


class TestPlacement(unittest.TestCase):
    POSITIONS = [(1, 100.0), (1, 180.0), (2, 260.0), (2, 330.0), (3, 420.0)]

    def test_a_mark_over_a_measure_applies_to_it(self) -> None:
        self.assertEqual(measure_at(90.0, self.POSITIONS, 10.0), 1)
        self.assertEqual(measure_at(300.0, self.POSITIONS, 10.0), 2)

    def test_a_mark_between_measures_applies_to_the_next(self) -> None:
        self.assertEqual(measure_at(200.0, self.POSITIONS, 10.0), 2)
        self.assertEqual(measure_at(250.0, self.POSITIONS, 10.0), 2)

    def test_a_mark_just_after_a_measures_last_symbol_stays_in_it(self) -> None:
        self.assertEqual(measure_at(189.0, self.POSITIONS, 10.0), 1)
        self.assertEqual(measure_at(191.0, self.POSITIONS, 10.0), 2)

    def test_a_mark_after_the_last_symbol_has_no_measure(self) -> None:
        self.assertEqual(measure_at(429.0, self.POSITIONS, 10.0), 3)
        self.assertIsNone(measure_at(431.0, self.POSITIONS, 10.0))
        self.assertIsNone(measure_at(10.0, [], 10.0))


def tokens(lines: list[str]) -> list[EncodedSymbol]:
    return read_token_lines(lines)


TWO_BARS = [
    "clef_G2 _ _ _ _ upper",
    "keySignature_0 . . . . .",
    "timeSignature/4 . . . . .",
    "note_2 C5 _ _ _ upper",
    "note_2 D5 _ _ _ upper",
    "barline . . . . .",
    "timeSignature/8 . . . . .",
    "note_4. E5 _ _ _ upper",
    "note_4. F5 _ _ _ upper",
    "barline . . . . .",
]


def placed(measure: int, x: float, value: str = "quarter", dotted: bool = False) -> TempoMark:
    return TempoMark(
        "140",
        (x, 0.0, x + 50.0, 20.0),
        "140",
        140.0,
        band(),
        NoteReading(value, dotted, (x, 0.0, x + 18.0, 20.0)),
        PLACED,
        "",
        measure,
    )


def measure_children(xml: Any, index: int) -> list[ET.Element]:
    root = ET.fromstring(xml.to_string())  # noqa: S314 - generated by the test itself
    part = root.find("part")
    assert part is not None
    return list(part.findall("measure")[index])


def metronomes(xml: Any) -> list[tuple[str, str | None, bool, str | None, str | None]]:
    """(measure, beat unit, dotted, per minute, sound tempo) of every tempo direction."""
    root = ET.fromstring(xml.to_string())  # noqa: S314 - generated by the test itself
    result = []
    for measure in root.iter("measure"):
        for direction in measure.iter("direction"):
            metronome = direction.find("direction-type/metronome")
            sound = direction.find("sound")
            assert metronome is not None and sound is not None
            result.append(
                (
                    measure.get("number", ""),
                    metronome.findtext("beat-unit"),
                    metronome.find("beat-unit-dot") is not None,
                    metronome.findtext("per-minute"),
                    sound.get("tempo"),
                )
            )
    return result


class TestWriting(unittest.TestCase):
    def test_a_mark_is_written_before_its_measures_first_note(self) -> None:
        xml = generate_xml(XmlGeneratorArguments(), [tokens(TWO_BARS)], "")
        mark = placed(2, 300.0, dotted=True)
        write_tempo_directions(xml, [mark])
        self.assertEqual(mark.status, WRITTEN)
        self.assertEqual(metronomes(xml), [("2", "quarter", True, "140", "210")])
        self.assertEqual(
            [child.tag for child in measure_children(xml, 1)][:3],
            ["attributes", "direction", "note"],
        )

    def test_a_tempo_that_is_not_whole_keeps_its_fraction(self) -> None:
        xml = generate_xml(XmlGeneratorArguments(), [tokens(TWO_BARS)], "")
        mark = placed(1, 10.0, "eighth", dotted=True)
        mark.per_minute, mark.beats_per_minute = "93", 93.0
        write_tempo_directions(xml, [mark])
        self.assertEqual(metronomes(xml), [("1", "eighth", True, "93", "69.75")])

    def test_a_measure_takes_the_first_of_its_marks(self) -> None:
        xml = generate_xml(XmlGeneratorArguments(), [tokens(TWO_BARS)], "")
        second = placed(1, 200.0, "eighth")
        second.per_minute, second.beats_per_minute = "360", 360.0
        first = placed(1, 100.0)
        write_tempo_directions(xml, [second, first])
        self.assertEqual((first.status, second.status), (WRITTEN, NOT_WRITTEN))
        self.assertEqual(second.reason, "measure_has_an_earlier_mark")
        self.assertEqual(metronomes(xml), [("1", "quarter", False, "140", "140")])

    def test_marks_not_placed_in_a_written_measure_are_left_out(self) -> None:
        xml = generate_xml(XmlGeneratorArguments(), [tokens(TWO_BARS)], "")
        missing = placed(7, 10.0)
        unplaced = placed(1, 10.0)
        unplaced.status = UNPLACED
        unread = placed(1, 10.0)
        unread.status, unread.note = UNREAD, None
        write_tempo_directions(xml, [missing, unplaced, unread])
        self.assertEqual(
            (missing.status, missing.reason), (NOT_WRITTEN, "measure_not_in_first_part")
        )
        self.assertEqual((unplaced.status, unread.status), (UNPLACED, UNREAD))
        self.assertEqual(metronomes(xml), [])


def identity_transform() -> PredictionCoordinateTransform:
    return PredictionCoordinateTransform(
        source_image_size=(900, 600),
        autocrop_box=(0, 0, 900, 600),
        cropped_size=(900, 600),
        resized_size=(900, 600),
        resize_scale=(1.0, 1.0),
        prediction_size=(900, 600),
    )


def visual_group(visual_id: str, x: float, status: str = "canonical") -> VisualGroup:
    return VisualGroup(
        visual_id=visual_id,
        staff_group_index=0,
        staff_index=0,
        staff_position=0,
        prediction_center=(x, SYSTEM_TOP + 20.0),
        prediction_notehead_size=(18.0, 12.0),
        transformer_center=None,
        transformer_notehead_size=None,
        notehead_ellipses=[],
        notehead_contours=[],
        detected_notehead_contours=[],
        refined_notehead_contours=[],
        detected_stem_contours=[],
        stem_contours=[],
        owned_stem_component_ids=[],
        is_hollow_notehead=False,
        visual_status=status,
        provenance="test",
    )


def linked_sidecar(xs: dict[str, float]) -> tuple[VisualSidecarBuilder, list[EncodedSymbol]]:
    """A sidecar for TWO_BARS with the notes of the given pitches linked at those x."""
    sidecar = VisualSidecarBuilder(identity_transform())
    voice = tokens(TWO_BARS)
    for symbol in voice:
        if symbol.pitch in xs:
            visual_id = f"v-{symbol.pitch}"
            sidecar.visual_groups[visual_id] = visual_group(visual_id, xs[symbol.pitch])
            sidecar.matches_by_symbol_id[symbol.visual_match_id] = VisualMatch(
                symbol, visual_id, 0.9, "structural"
            )
    sidecar.state.source_staffs[0] = cast(Staff, SimpleNamespace(min_x=0.0, min_y=SYSTEM_TOP))
    sidecar.state.system_index_by_staff_group[0] = 0
    return sidecar, voice


class TestSidecar(unittest.TestCase):
    def test_marks_are_placed_by_the_linked_notes_and_reported(self) -> None:
        sidecar, voice = linked_sidecar({"C5": 100.0, "D5": 180.0, "E5": 300.0, "F5": 380.0})
        xml = generate_xml(XmlGeneratorArguments(), [voice], "", visual_sidecar=sidecar)
        opening, change = placed(0, 60.0, dotted=True), placed(0, 240.0)
        opening.measure = change.measure = None
        sidecar.write_tempo_marks(xml, [opening, change])
        self.assertEqual((opening.measure, change.measure), (1, 2))
        self.assertEqual(
            metronomes(xml),
            [("1", "quarter", True, "140", "210"), ("2", "quarter", False, "140", "140")],
        )
        block = json.loads(json.dumps(sidecar.to_json_dict()))["tempo_marks"]
        self.assertEqual(block["version"], 1)
        self.assertEqual(
            block["marks"][1],
            {
                "text": "140",
                "box": [240.0, 0.0, 290.0, 20.0],
                "per_minute": "140",
                "beat_unit": "quarter",
                "dotted": False,
                "quarter_bpm": 140.0,
                "measure": 2,
                "status": WRITTEN,
                "reason": "",
            },
        )

    def test_a_measure_of_printed_rests_places_a_mark_too(self) -> None:
        def measure_with_rest(status: str, part: int = 1) -> int | None:
            sidecar, voice = linked_sidecar({"E5": 300.0, "F5": 380.0})
            xml = generate_xml(XmlGeneratorArguments(), [voice], "", visual_sidecar=sidecar)
            rest = SimpleNamespace(
                part=part,
                measure=1,
                verdict=RestVerdict(0, 0, 0, "rest_1", status, "", center=(140.0, 230.0)),
            )
            sidecar.state.musicxml_rests.append(cast(Any, rest))
            mark = placed(0, 120.0)
            sidecar.write_tempo_marks(xml, [mark])
            return mark.measure

        self.assertEqual(measure_with_rest(SUPPORTED), 1)
        # A rest the page does not print sits where attention put it, if anywhere, and
        # a rest of another part counts its own measures.
        self.assertEqual(measure_with_rest("unsupported"), 2)
        self.assertEqual(measure_with_rest(SUPPORTED, part=2), 2)

    def test_notes_withdrawn_as_diagnostic_do_not_place_marks(self) -> None:
        sidecar, voice = linked_sidecar({"C5": 100.0, "E5": 300.0})
        sidecar.visual_groups["v-C5"].visual_status = "diagnostic"
        xml = generate_xml(XmlGeneratorArguments(), [voice], "", visual_sidecar=sidecar)
        mark = placed(0, 60.0)
        sidecar.write_tempo_marks(xml, [mark])
        self.assertEqual(mark.measure, 2)

    def test_an_explicit_tempo_wins_over_the_printed_marks(self) -> None:
        sidecar, voice = linked_sidecar({"C5": 100.0, "E5": 300.0})
        xml = generate_xml(XmlGeneratorArguments(), [voice], "", visual_sidecar=sidecar)
        mark = placed(0, 60.0)
        sidecar.write_tempo_marks(xml, [mark], explicit_tempo=True)
        self.assertEqual(
            (mark.measure, mark.status, mark.reason), (1, NOT_WRITTEN, "tempo_given_explicitly")
        )
        self.assertEqual(metronomes(xml), [])

    def test_marks_above_no_parsed_system_or_measure_stay_unplaced(self) -> None:
        sidecar, voice = linked_sidecar({"C5": 100.0, "E5": 300.0})
        xml = generate_xml(XmlGeneratorArguments(), [voice], "", visual_sidecar=sidecar)
        elsewhere = placed(0, 60.0)
        elsewhere.system = SystemBand(
            top=SYSTEM_TOP + 2.5 * UNIT_PIXELS,
            bottom=500.0,
            left=0.0,
            right=880.0,
            unit_size=UNIT_PIXELS,
        )
        nearby = placed(0, 60.0)
        nearby.system = SystemBand(
            top=SYSTEM_TOP + 1.5 * UNIT_PIXELS,
            bottom=500.0,
            left=0.0,
            right=880.0,
            unit_size=UNIT_PIXELS,
        )
        beyond = placed(0, 600.0)
        unread = placed(0, 60.0)
        unread.note, unread.status, unread.reason = None, UNREAD, "glyph_is_not_a_note"
        sidecar.write_tempo_marks(xml, [elsewhere, nearby, beyond, unread])
        self.assertEqual(
            [
                (mark.status, mark.reason, mark.measure)
                for mark in (elsewhere, nearby, beyond, unread)
            ],
            [
                (UNPLACED, "system_not_parsed", None),
                (WRITTEN, "", 1),
                (UNPLACED, "no_measure_under_mark", None),
                (UNREAD, "glyph_is_not_a_note", None),
            ],
        )

    def test_the_block_is_absent_when_marks_were_never_asked_for(self) -> None:
        self.assertNotIn("tempo_marks", VisualSidecarBuilder(identity_transform()).to_json_dict())
        sidecar, voice = linked_sidecar({})
        xml = generate_xml(XmlGeneratorArguments(), [voice], "", visual_sidecar=sidecar)
        sidecar.write_tempo_marks(xml, [])
        self.assertEqual(sidecar.to_json_dict()["tempo_marks"], {"version": 1, "marks": []})


if __name__ == "__main__":
    unittest.main()
