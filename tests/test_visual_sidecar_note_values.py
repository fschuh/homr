# ruff: noqa: S101

import json
import unittest

import cv2
import numpy as np

from homr.model import Staff, StaffPoint
from homr.visual_sidecar import PredictionCoordinateTransform, VisualSidecarBuilder
from homr.visual_sidecar.models import MusicXmlNoteRecord, VisualGroup
from homr.visual_sidecar.note_values import (
    AGREES,
    DISAGREES,
    UNKNOWN,
    ChordMember,
    NoteValueReader,
    NoteValueReading,
    SharedNoteheadReading,
    recognized_value,
)
from homr.visual_sidecar.rests import SegmentationMasks

UNIT = 20
LINES = [100.0, 120.0, 140.0, 160.0, 180.0]
WIDTH, HEIGHT = 600, 400
HEAD = (13, 9)  # notehead half axes: 1.3 by 0.9 staff spaces
STEM_LENGTH = 70  # 3.5 staff spaces
BEAM = 10  # half a staff space thick
BEAM_STEP = 15


def identity_transform(width: int = WIDTH, height: int = HEIGHT) -> PredictionCoordinateTransform:
    return PredictionCoordinateTransform(
        source_image_size=(width, height),
        autocrop_box=(0, 0, width, height),
        cropped_size=(width, height),
        resized_size=(width, height),
        resize_scale=(1.0, 1.0),
        prediction_size=(width, height),
    )


def staff() -> Staff:
    return Staff([StaffPoint(float(x), list(LINES), 0.0) for x in range(0, WIDTH, 10)])


def lines_at_x(staff: Staff, x: float, staff_index: int) -> list[float]:
    return LINES


class Page:
    """A printed staff with the masks segnet would give for what is drawn on it."""

    def __init__(self) -> None:
        self.image = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        self.staff = np.zeros_like(self.image)
        self.notehead = np.zeros_like(self.image)
        self.stems = np.zeros_like(self.image)
        self.accidentals = np.zeros_like(self.image)
        for y in LINES:
            cv2.line(self.image, (0, int(y)), (WIDTH - 1, int(y)), 0, 2)
            cv2.line(self.staff, (0, int(y)), (WIDTH - 1, int(y)), 1, 3)
        self.count = 0

    def masks(self) -> SegmentationMasks:
        return SegmentationMasks(
            staff=self.staff,
            stems_rest=self.stems,
            notehead=self.notehead,
            clefs_keys=self.accidentals,
            symbols=np.zeros_like(self.image),
        )

    def head(
        self, x: int, y: int, hollow: bool = False, chord_id: str | None = None
    ) -> VisualGroup:
        cv2.ellipse(self.image, (x, y), HEAD, -20, 0, 360, 0, -1)
        if hollow:
            cv2.ellipse(self.image, (x, y), (9, 4), -20, 0, 360, 255, -1)
        cv2.ellipse(self.notehead, (x, y), HEAD, -20, 0, 360, 1, -1)
        self.count += 1
        return VisualGroup(
            visual_id=f"vnote-{self.count}",
            staff_group_index=0,
            staff_index=0,
            staff_position=0,
            prediction_center=(float(x), float(y)),
            prediction_notehead_size=(2.0 * HEAD[0], 2.0 * HEAD[1]),
            transformer_center=None,
            transformer_notehead_size=None,
            notehead_ellipses=[],
            notehead_contours=[],
            detected_notehead_contours=[],
            refined_notehead_contours=[],
            detected_stem_contours=[],
            stem_contours=[],
            owned_stem_component_ids=[],
            is_hollow_notehead=hollow,
            visual_status="canonical",
            provenance="test",
            chord_id=chord_id,
        )

    def stem(self, x: int, top: int, bottom: int, *recorded_on: VisualGroup) -> None:
        """Print a stem; record it on the groups segmentation would give it to."""
        cv2.line(self.image, (x, top), (x, bottom), 0, 2)
        cv2.line(self.stems, (x, top), (x, bottom), 1, 3)
        for group in recorded_on:
            group.stem_contours.append([[float(x), float(top)], [float(x), float(bottom)]])

    def up_stem(self, group: VisualGroup, length: int = STEM_LENGTH) -> tuple[int, int]:
        """A stem rising from the head's right edge; returns its x and free end."""
        x, y = int(group.prediction_center[0]) + HEAD[0] - 1, int(group.prediction_center[1])
        self.stem(x, y - length, y, group)
        return x, y - length

    def down_stem(self, group: VisualGroup, length: int = STEM_LENGTH) -> tuple[int, int]:
        x, y = int(group.prediction_center[0]) - HEAD[0] + 1, int(group.prediction_center[1])
        self.stem(x, y, y + length, group)
        return x, y + length

    def beams(
        self,
        x0: int,
        x1: int,
        free_end: int,
        count: int,
        up: bool = True,
        thickness: int = BEAM,
        step: int = BEAM_STEP,
    ) -> None:
        """Beams hanging inward from a stem's free end, between two xs."""
        for index in range(count):
            offset = index * step
            top = free_end + offset if up else free_end - offset - thickness
            cv2.rectangle(self.image, (x0, top), (x1, top + thickness - 1), 0, -1)

    def dot(self, x: int, y: int) -> None:
        cv2.circle(self.image, (x, y), 4, 0, -1)

    def read(self, *members: tuple[VisualGroup, str]) -> list[NoteValueReading]:
        reader = NoteValueReader(self.image, self.masks(), identity_transform(), lines_at_x)
        return reader.read_chord(
            [
                ChordMember(f"homr-note-{index + 1}", group, rhythm)
                for index, (group, rhythm) in enumerate(members)
            ],
            staff(),
        )

    def read_one(self, group: VisualGroup, rhythm: str) -> NoteValueReading:
        return self.read((group, rhythm))[0]


def outcome(reading: NoteValueReading) -> tuple[str | None, bool | None, str]:
    return reading.printed, reading.dotted, reading.status


class TestRecognizedValue(unittest.TestCase):
    def test_plain_values_dots_and_tuplets(self) -> None:
        self.assertEqual(recognized_value("note_8"), ("eighth", False))
        self.assertEqual(recognized_value("note_4."), ("quarter", True))
        self.assertEqual(recognized_value("note_12"), ("eighth", False))  # triplet eighth
        self.assertEqual(recognized_value("note_1"), ("whole", False))
        self.assertEqual(recognized_value("note_8G"), (None, False))  # grace note
        self.assertEqual(recognized_value("rest_8"), (None, False))


class TestPrintedValue(unittest.TestCase):
    def test_flags_or_beams_at_the_free_end_give_the_value(self) -> None:
        for count, value in enumerate(("quarter", "eighth", "16th", "32nd")):
            page = Page()
            note = page.head(200, 130)
            x, free_end = page.up_stem(note)
            page.beams(x, x + 40, free_end, count)
            reading = page.read_one(note, "note_16")
            self.assertEqual(reading.printed, value)
            self.assertEqual(reading.status, AGREES if value == "16th" else DISAGREES)

    def test_more_bands_than_a_32nd_are_not_read(self) -> None:
        page = Page()
        note = page.head(200, 160)
        x, free_end = page.up_stem(note, length=90)
        # A 64th's four beams only fit the window when packed tighter than usual.
        page.beams(x, x + 40, free_end, 4, thickness=7, step=11)
        self.assertEqual(page.read_one(note, "note_32").reason, "too_many_bands")

    def test_a_hollow_head_with_a_stem_is_a_half_and_without_one_a_whole(self) -> None:
        page = Page()
        half = page.head(200, 130, hollow=True)
        page.up_stem(half)
        whole = page.head(400, 130, hollow=True)
        self.assertEqual(outcome(page.read_one(half, "note_2")), ("half", False, AGREES))
        self.assertEqual(outcome(page.read_one(whole, "note_2")), ("whole", False, DISAGREES))

    def test_a_hollow_head_beside_an_unrecorded_stem_is_not_a_whole(self) -> None:
        page = Page()
        note = page.head(200, 130, hollow=True)
        page.up_stem(note)
        note.stem_contours.clear()  # segnet missed the stem
        self.assertEqual(page.read_one(note, "note_2").reason, "stem_unclear")

    def test_a_stem_just_off_the_estimated_head_edge_is_still_seen(self) -> None:
        page = Page()
        note = page.head(200, 130, hollow=True)
        x = 200 + HEAD[0] + 4
        cv2.line(page.image, (x, 130 - STEM_LENGTH), (x, 130), 0, 2)  # not recorded
        self.assertEqual(page.read_one(note, "note_2").reason, "stem_unclear")

    def test_a_short_stem_hidden_in_part_by_the_notehead_mask_is_still_seen(self) -> None:
        page = Page()
        note = page.head(200, 130, hollow=True)
        x = 200 - HEAD[0] + 1
        cv2.line(page.image, (x, 130 + HEAD[1]), (x, 130 + HEAD[1] + 40), 0, 2)  # not recorded
        self.assertEqual(page.read_one(note, "note_2").reason, "stem_unclear")

    def test_a_hollow_head_with_an_unclear_recorded_stem_is_not_a_whole(self) -> None:
        page = Page()
        note = page.head(200, 130, hollow=True)
        page.stem(200 + HEAD[0] + 9, 130 - STEM_LENGTH, 130, note)  # beyond the edge check
        self.assertEqual(page.read_one(note, "note_2").reason, "stem_unclear")

    def test_a_head_neither_clearly_filled_nor_hollow_is_not_read(self) -> None:
        page = Page()
        holed = page.head(200, 130)
        cv2.circle(page.image, (200, 130), 2, 255, -1)  # a speck of paper in the core
        page.up_stem(holed)
        disputed = page.head(400, 130, hollow=True)
        disputed.is_hollow_notehead = False  # the sidecar's own test calls it filled
        page.up_stem(disputed)
        self.assertEqual(page.read_one(holed, "note_4").reason, "notehead_unclear")
        self.assertEqual(page.read_one(disputed, "note_4").reason, "notehead_unclear")

    def test_a_scanned_hollow_head_with_a_gray_hole_is_not_filled(self) -> None:
        page = Page()
        note = page.head(200, 130)
        cv2.ellipse(page.image, (200, 130), (9, 4), -20, 0, 360, 70, -1)
        page.up_stem(note)
        self.assertEqual(page.read_one(note, "note_2").reason, "notehead_unclear")

    def test_a_non_note_token_is_not_compared(self) -> None:
        page = Page()
        note = page.head(200, 130)
        page.up_stem(note)
        reading = page.read_one(note, "note_8G")
        self.assertEqual((reading.status, reading.reason), (UNKNOWN, "not_a_plain_note"))


class TestStem(unittest.TestCase):
    def test_the_stem_is_traced_past_where_segmentation_stops(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = 200 + HEAD[0] - 1, 130 - STEM_LENGTH
        cv2.line(page.image, (x, free_end), (x, 130), 0, 2)
        page.stem(x, 110, 130, note)  # segnet saw only the stem's first staff space
        page.beams(x, x + 40, free_end, 2)
        self.assertEqual(page.read_one(note, "note_16").printed, "16th")

    def test_a_chord_shares_the_stem_recorded_on_one_of_its_notes(self) -> None:
        page = Page()
        top = page.head(200, 110, chord_id="c")
        middle = page.head(200, 130, chord_id="c")
        bottom = page.head(200, 150, chord_id="c")
        x, free_end = page.down_stem(top, length=40 + STEM_LENGTH)
        page.beams(x, x + 40, free_end, 2, up=False)
        readings = page.read((top, "note_16"), (middle, "note_16"), (bottom, "note_8"))
        self.assertEqual([r.printed for r in readings], ["16th", "16th", "16th"])
        self.assertEqual([r.status for r in readings], [AGREES, AGREES, DISAGREES])

    def test_a_stem_running_through_the_chord_both_ways_is_not_read(self) -> None:
        page = Page()
        note = page.head(200, 140)
        x = 200 + HEAD[0] - 1
        page.stem(x, 140 - STEM_LENGTH, 140 + STEM_LENGTH, note)
        self.assertEqual(page.read_one(note, "note_4").reason, "stem_unclear")

    def test_stems_recorded_apart_are_two_voices(self) -> None:
        page = Page()
        upper = page.head(200, 120, chord_id="c")
        lower = page.head(200, 140, chord_id="c")
        stray = page.head(176, 130, chord_id="c")
        page.up_stem(upper)
        page.stem(200 + HEAD[0] - 1, 50, 140, lower)
        page.stem(260, 130, 210, stray)  # a neighbour's stem given to this note
        readings = page.read((upper, "note_4"), (lower, "note_4"), (stray, "note_4"))
        self.assertEqual({r.reason for r in readings}, {"stem_unclear"})

    def test_a_head_the_stem_does_not_reach_is_another_voices(self) -> None:
        page = Page()
        whole = page.head(200, 110, hollow=True, chord_id="c")
        half = page.head(200, 150, hollow=True, chord_id="c")
        page.down_stem(half)
        whole.stem_contours.extend(half.stem_contours)  # segnet gave both heads the stem
        readings = page.read((whole, "note_1"), (half, "note_2"))
        self.assertEqual({r.reason for r in readings}, {"stem_unclear"})

    def test_a_barline_beside_the_heads_is_not_their_stem(self) -> None:
        page = Page()
        note = page.head(200, 130)
        page.stem(160, 125, 260, note)
        self.assertEqual(page.read_one(note, "note_4").reason, "stem_unclear")

    def test_a_stem_rising_from_a_heads_left_edge_is_another_voices(self) -> None:
        page = Page()
        note = page.head(200, 130)
        page.stem(200 - HEAD[0] + 1, 130 - STEM_LENGTH, 130, note)
        self.assertEqual(page.read_one(note, "note_4").reason, "stem_unclear")

    def test_a_second_stem_leaving_the_other_way_is_another_voices(self) -> None:
        page = Page()
        note = page.head(200, 130)
        page.up_stem(note)
        left = 200 - HEAD[0] + 1
        cv2.line(page.image, (left, 130), (left, 130 + STEM_LENGTH), 0, 2)  # not recorded
        self.assertEqual(page.read_one(note, "note_4").reason, "stem_unclear")


class TestBands(unittest.TestCase):
    def test_staff_line_leftovers_are_not_bands(self) -> None:
        page = Page()
        note = page.head(200, 190)
        x, _ = page.up_stem(note)
        # The staff mask misses a thickened stretch of a line beside the stem.
        cv2.line(page.image, (x - 15, 140), (x + 15, 140), 0, 4)
        page.staff[136:145, x - 15 : x + 16] = 0
        self.assertEqual(page.read_one(note, "note_4").printed, "quarter")

    def test_a_thick_run_beside_the_free_end_is_another_symbol(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.up_stem(note)
        cv2.rectangle(page.image, (x + 2, free_end + 10), (x + 30, free_end + 30), 0, -1)
        self.assertEqual(page.read_one(note, "note_4").reason, "bands_unclear")

    def test_ink_just_past_the_stems_end_is_not_a_band(self) -> None:
        page = Page()
        note = page.head(200, 140)
        x, free_end = page.down_stem(note)
        # A fingering numeral printed just under the stem, beside its column: a bar and
        # a down stroke, about as tall as a staff space.
        cv2.rectangle(page.image, (x + 3, free_end + 3), (x + 14, free_end + 6), 0, -1)
        cv2.rectangle(page.image, (x + 12, free_end + 3), (x + 14, free_end + 21), 0, -1)
        self.assertEqual(page.read_one(note, "note_4").reason, "bands_unclear")

    def test_a_beam_just_past_a_stem_that_stops_short_is_not_missed(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.down_stem(note)
        page.beams(x - 40, x + 40, free_end + 7 + BEAM, 1, up=False)  # a gap above the beam
        self.assertEqual(page.read_one(note, "note_8").reason, "bands_unclear")

    def test_a_staccato_dot_past_the_stems_end_is_not_a_band(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.down_stem(note)
        page.beams(x - 40, x, free_end, 2, up=False)
        page.dot(x + 3, free_end + 10)
        self.assertEqual(page.read_one(note, "note_16").printed, "16th")

    def test_columns_on_one_side_must_agree(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.up_stem(note)
        page.beams(x, x + 8, free_end, 1)  # crosses the near column only
        self.assertEqual(page.read_one(note, "note_8").reason, "bands_unclear")

    def test_the_side_showing_more_bands_gives_the_count(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.up_stem(note)
        page.beams(x - 60, x, free_end, 2)  # the last note of its group
        self.assertEqual(page.read_one(note, "note_16").printed, "16th")

    def test_a_notehead_the_chord_group_left_out_is_not_a_band(self) -> None:
        page = Page()
        note = page.head(200, 330)
        x, _ = page.up_stem(note)
        # Another head on the same stem, small enough to pass for a beam by thickness.
        cv2.ellipse(page.image, (x, 290), (12, 6), 0, 0, 360, 0, -1)
        cv2.ellipse(page.notehead, (x, 290), (12, 6), 0, 0, 360, 1, -1)
        self.assertEqual(page.read_one(note, "note_4").reason, "bands_unclear")

    def test_a_beam_end_labelled_notehead_is_not_miscounted(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.up_stem(note)
        page.beams(x, x + 40, free_end, 2)
        # Erased with the noteheads, the beams would read as none: a quarter.
        page.notehead[free_end : free_end + 30, x - 2 : x + 14] = 1
        self.assertEqual(page.read_one(note, "note_16").reason, "bands_unclear")

    def test_a_beam_only_grazing_a_notehead_still_counts(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.up_stem(note)
        page.beams(x, x + 40, free_end, 2)
        page.notehead[free_end : free_end + 3, x - 2 : x + 14] = 1  # 3 of the first beam's 10 rows
        self.assertEqual(page.read_one(note, "note_16").printed, "16th")

    def test_an_accidental_beside_the_free_end_is_not_a_band(self) -> None:
        page = Page()
        note = page.head(200, 160)
        x, free_end = page.down_stem(note)
        # A natural's thick stroke, thicker than any staff line leftover.
        cv2.rectangle(page.image, (x - 14, free_end - 32), (x - 4, free_end - 25), 0, -1)
        page.accidentals[free_end - 33 : free_end - 22, x - 17 : x - 1] = 1
        self.assertEqual(page.read_one(note, "note_4").printed, "quarter")


class TestDots(unittest.TestCase):
    def dotted_note(self, dot: tuple[int, int] | None, y: int = 130) -> tuple[Page, VisualGroup]:
        page = Page()
        note = page.head(200, y)
        page.up_stem(note)
        if dot is not None:
            page.dot(*dot)
        return page, note

    def test_a_round_dot_in_a_space_dots_the_note(self) -> None:
        page, note = self.dotted_note((200 + HEAD[0] + 10, 130))
        self.assertEqual(outcome(page.read_one(note, "note_4.")), ("quarter", True, AGREES))
        self.assertEqual(outcome(page.read_one(note, "note_4")), ("quarter", True, DISAGREES))

    def test_clean_paper_beside_the_head_means_no_dot(self) -> None:
        page, note = self.dotted_note(None)
        self.assertEqual(outcome(page.read_one(note, "note_4.")), ("quarter", False, DISAGREES))

    def test_a_dot_below_a_note_on_a_line(self) -> None:
        page, note = self.dotted_note((200 + HEAD[0] + 10, 152), y=140)
        self.assertTrue(page.read_one(note, "note_4.").dotted)

    def test_a_dot_on_a_line_is_not_read(self) -> None:
        page, note = self.dotted_note((200 + HEAD[0] + 10, 140))
        self.assertIsNone(page.read_one(note, "note_4.").dotted)

    def test_the_knob_of_a_rest_is_not_a_dot(self) -> None:
        page, note = self.dotted_note((200 + HEAD[0] + 10, 130))
        knob_x = 200 + HEAD[0] + 10
        cv2.line(page.image, (knob_x + 3, 132), (knob_x + 12, 170), 0, 2)
        cv2.line(page.stems, (knob_x + 3, 136), (knob_x + 12, 170), 1, 4)
        self.assertIsNone(page.read_one(note, "note_4").dotted)

    def test_the_staccato_of_the_next_note_is_not_a_dot(self) -> None:
        page, note = self.dotted_note((232, 130))
        page.head(232, 152)
        self.assertIsNone(page.read_one(note, "note_4").dotted)

    def test_the_dots_of_a_repeat_sign_are_not_read(self) -> None:
        page, note = self.dotted_note((200 + HEAD[0] + 10, 130))
        page.stem(240, 100, 180)
        self.assertIsNone(page.read_one(note, "note_4").dotted)

    def test_a_dot_beside_any_head_dots_the_chord(self) -> None:
        page = Page()
        upper = page.head(200, 110, chord_id="c")
        lower = page.head(200, 150, chord_id="c")
        page.stem(200 + HEAD[0] - 1, 150 - 40 - STEM_LENGTH, 150, upper)
        page.dot(200 + HEAD[0] + 10, 110)
        readings = page.read((upper, "note_4."), (lower, "note_4."))
        self.assertEqual([r.dotted for r in readings], [True, True])

    def test_a_chords_dot_is_measured_from_its_rightmost_head(self) -> None:
        page = Page()
        lower = page.head(200, 150, chord_id="c")
        displaced = page.head(224, 130, chord_id="c")  # a second, beside the stem
        page.stem(200 + HEAD[0] - 1, 150 - STEM_LENGTH, 150, lower)
        page.dot(224 + HEAD[0] + 10, 150)
        readings = page.read((lower, "note_4."), (displaced, "note_4."))
        self.assertEqual([r.dotted for r in readings], [True, True])


class TestTwoVoices(unittest.TestCase):
    def test_hollow_and_filled_heads_at_one_moment_are_read_apart(self) -> None:
        page = Page()
        eighth = page.head(200, 130, chord_id="c")
        hollow = page.head(200, 170, hollow=True, chord_id="c")
        x = 200 + HEAD[0] - 1
        page.stem(x, 130 - STEM_LENGTH, 130, eighth, hollow)
        page.beams(x, x + 40, 130 - STEM_LENGTH, 1)
        readings = page.read((eighth, "note_8"), (hollow, "note_1"))
        self.assertEqual(outcome(readings[0]), ("eighth", False, AGREES))
        self.assertEqual((readings[1].printed, readings[1].reason), (None, "two_voices"))


class TestSharedNotehead(unittest.TestCase):
    """A head two voices share: an eighth rising from it, a 16th run falling from it."""

    def shared_head(self, y: int = 130, hollow: bool = False) -> tuple[Page, VisualGroup]:
        page = Page()
        note = page.head(200, y, hollow=hollow)
        x, free_end = page.up_stem(note)
        page.beams(x, x + 40, free_end, 1)
        x, free_end = page.down_stem(note)
        page.beams(x, x + 40, free_end, 2, up=False)
        return page, note

    def read(self, page: Page, note: VisualGroup) -> SharedNoteheadReading:
        reader = NoteValueReader(page.image, page.masks(), identity_transform(), lines_at_x)
        return reader.read_shared_notehead(note, staff())

    def summary(self, reading: SharedNoteheadReading) -> tuple[str | None, str | None, str]:
        return reading.up, reading.down, reading.reason

    def test_each_stem_of_a_shared_head_is_read(self) -> None:
        page, note = self.shared_head()
        reading = self.read(page, note)
        self.assertEqual(self.summary(reading), ("eighth", "16th", "two_stems"))
        self.assertIs(reading.dotted, False)

    def test_a_dot_beside_a_shared_head_is_read(self) -> None:
        page, note = self.shared_head()
        page.dot(200 + HEAD[0] + 10, 125)
        self.assertIs(self.read(page, note).dotted, True)

    def test_a_head_with_one_stem_is_not_shared(self) -> None:
        for draw in (Page.up_stem, Page.down_stem):
            page = Page()
            note = page.head(200, 130)
            draw(page, note)
            self.assertEqual(self.summary(self.read(page, note)), (None, None, "one_stem"))

    def test_a_stem_too_short_to_carry_bands_is_not_a_second_stem(self) -> None:
        page = Page()
        note = page.head(200, 130)
        page.up_stem(note)
        page.down_stem(note, length=30)
        self.assertEqual(self.read(page, note).reason, "one_stem")

    def test_a_line_through_the_heads_middle_is_not_its_stem(self) -> None:
        page = Page()
        note = page.head(200, 130)
        page.stem(200, 130 - STEM_LENGTH, 130)
        page.down_stem(note)
        # Segmentation underestimates the head, so its middle lies where a stem is looked for.
        note.prediction_notehead_size = (12.0, 18.0)
        self.assertEqual(self.read(page, note).reason, "one_stem")

    def test_a_stem_through_another_head_belongs_to_a_chord(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.up_stem(note)
        page.beams(x, x + 40, free_end, 1)
        page.head(200, 170)
        x, free_end = page.down_stem(note, length=90)
        page.beams(x, x + 40, free_end, 2, up=False)
        self.assertEqual(self.summary(self.read(page, note)), (None, None, "stem_unclear"))

    def test_a_hollow_head_is_not_read(self) -> None:
        page, note = self.shared_head(hollow=True)
        self.assertEqual(self.read(page, note).reason, "notehead_unclear")

    def test_a_flag_too_thick_to_read_leaves_only_its_own_value_unread(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.up_stem(note)
        page.beams(x, x + 40, free_end, 1, thickness=18)
        x, free_end = page.down_stem(note)
        page.beams(x, x + 40, free_end, 2, up=False)
        self.assertEqual(self.summary(self.read(page, note)), (None, "16th", "two_stems"))

    def test_a_thick_stem_is_read_from_its_middle(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x = 200 + HEAD[0] - 2
        cv2.line(page.image, (x, 130 - STEM_LENGTH), (x, 130), 0, 5)
        page.beams(x, x + 40, 130 - STEM_LENGTH, 1)
        x, free_end = page.down_stem(note)
        page.beams(x, x + 40, free_end, 2, up=False)
        self.assertEqual(self.summary(self.read(page, note)), ("eighth", "16th", "two_stems"))

    def test_more_bands_than_a_32nd_leave_that_value_unread(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.up_stem(note)
        page.beams(x, x + 40, free_end, 1)
        x, free_end = page.down_stem(note, length=90)
        page.beams(x, x + 40, free_end, 4, up=False, thickness=7, step=11)
        self.assertEqual(self.summary(self.read(page, note)), ("eighth", None, "two_stems"))

    def test_without_segmentation_nothing_is_read(self) -> None:
        page, note = self.shared_head()
        reader = NoteValueReader(page.image, None, identity_transform(), lines_at_x)
        reading = reader.read_shared_notehead(note, staff())
        self.assertEqual(self.summary(reading), (None, None, "no_segmentation"))


class TestSidecarBlock(unittest.TestCase):
    def test_every_linked_note_is_reported_and_diagnostic_ones_are_not(self) -> None:
        page = Page()
        note = page.head(200, 130)
        x, free_end = page.up_stem(note)
        page.beams(x, x + 40, free_end, 1)
        diagnostic = page.head(400, 130)
        diagnostic.visual_status = "diagnostic"
        builder = VisualSidecarBuilder(
            identity_transform(),
            source_image=cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR),
            segmentation_masks=page.masks(),
        )
        builder.state.source_staffs[0] = staff()
        for index, group in enumerate((note, diagnostic)):
            builder.visual_groups[group.visual_id] = group
            builder.musicxml_notes.append(
                MusicXmlNoteRecord(
                    musicxml_id=f"homr-note-{index + 1}",
                    part=1,
                    measure=1,
                    musicxml_staff_number=1,
                    voice=1,
                    pitch="C5",
                    duration="note_16",
                    match_confidence=0.9,
                    visual_group_id=group.visual_id,
                    alignment_method="structural",
                )
            )
        sidecar = json.loads(json.dumps(builder.to_json_dict()))
        self.assertEqual(
            sidecar["note_value_verification"],
            {
                "version": 1,
                "notes": [
                    {
                        "musicxml_id": "homr-note-1",
                        "printed": "eighth",
                        "dotted": False,
                        "status": DISAGREES,
                        "reason": "1_bands",
                    }
                ],
            },
        )
