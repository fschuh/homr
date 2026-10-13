# ruff: noqa: S101

import io
import sys
import unittest
from collections.abc import Iterable
from concurrent.futures import Future
from unittest import mock

import cv2
import numpy as np

from homr import main as homr_main
from homr.music_xml_generator import XmlGeneratorArguments
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray
from homr.visual_sidecar import VisualSidecarBuilder
from homr.visual_sidecar import builder as builder_module
from homr.visual_sidecar.models import VisualMatch
from homr.visual_sidecar.time_signatures import (
    BEAT_TYPE_DIFFERS,
    NO_READ,
    READ,
    UNREAD_GLYPH,
    Glyph,
    StaffLines,
    TimeSignatureLinks,
    TimeSignatureRead,
    ink_without_lines,
    is_meter,
    read_time_signatures,
    validate_meter_options,
)
from tests.test_timing_repairs import identity_transform, linked_group

SPACE = 20
LINES = [100, 120, 140, 160, 180]
WIDTH, HEIGHT = 600, 300


def staff(system_index: int = 0, offset: int = 0) -> StaffLines:
    lines = [np.array([y + offset, y + offset], dtype=float) for y in LINES]
    return StaffLines(
        system_index, np.array([0.0, WIDTH - 1.0]), lines, float(SPACE), (100.0, 180.0)
    )


class Page:
    """A printed staff, with time signatures drawn on it as solid glyphs."""

    def __init__(self, staves: int = 1) -> None:
        self.gray = np.full((HEIGHT * staves, WIDTH), 255, dtype=np.uint8)
        self.staffs = []
        for index in range(staves):
            for y in LINES:
                cv2.line(self.gray, (0, y + index * HEIGHT), (WIDTH - 1, y + index * HEIGHT), 0, 2)
            self.staffs.append(staff(0, index * HEIGHT))

    def numbers(self, x: int, offset: int = 0) -> None:
        """A top number between the top and middle lines, a bottom one below it."""
        cv2.rectangle(self.gray, (x, 103 + offset), (x + 28, 137 + offset), 0, -1)
        cv2.rectangle(self.gray, (x, 144 + offset), (x + 28, 178 + offset), 0, -1)

    def common_time(self, x: int) -> None:
        cv2.ellipse(self.gray, (x, 140), (15, 20), 0, 40, 320, 0, 6)

    def cut_time(self, x: int) -> None:
        self.common_time(x)
        cv2.line(self.gray, (x, 110), (x, 170), 0, 2)

    def read(self, *texts: str) -> tuple[list[TimeSignatureRead], list[Glyph], list[NDArray]]:
        reader = Script(texts)
        glyphs: list[Glyph] = []
        return read_time_signatures(self.gray, self.staffs, reader, glyphs), glyphs, reader.crops


class Script:
    """OCR that reads each crop as the next scripted text, in turn."""

    def __init__(self, texts: Iterable[str]) -> None:
        self.texts = list(texts)
        self.crops: list[NDArray] = []

    def __call__(self, crop: NDArray) -> str:
        self.crops.append(crop)
        return self.texts[(len(self.crops) - 1) % len(self.texts)]


def meters(reads: list[TimeSignatureRead]) -> list[tuple[int, int, str | None]]:
    return [(r.beats, r.beat_type, r.symbol) for r in reads]


class TestReadingSignatures(unittest.TestCase):
    def test_stacked_numbers_are_read_as_a_meter(self) -> None:
        page = Page()
        page.numbers(300)
        reads, glyphs, crops = page.read("3", "4")
        self.assertEqual(meters(reads), [(3, 4, None)])
        self.assertAlmostEqual(reads[0].x, 314.5, delta=1)
        # Each number is read on its own, in both passes over the page.
        self.assertEqual(len(crops), 4)
        self.assertEqual([round(g.x) for g in glyphs], [314, 314])

    def test_look_alike_letters_are_read_as_digits(self) -> None:
        page = Page()
        page.numbers(300)
        self.assertEqual(meters(page.read("l2", "O8")[0]), [(12, 8, None)])

    def test_both_passes_must_agree(self) -> None:
        page = Page()
        page.numbers(300)
        reads, glyphs, _ = page.read("3", "4", "3", "8")
        self.assertEqual(reads, [])
        self.assertEqual(len(glyphs), 2)  # still something printed there

    def test_every_staff_of_the_system_must_agree(self) -> None:
        page = Page(staves=2)
        page.numbers(300)
        page.numbers(300, HEIGHT)
        self.assertEqual(meters(page.read("3", "4")[0]), [(3, 4, None)])
        # The second staff reads 2/4 in both passes.
        self.assertEqual(page.read("3", "4", "2", "4")[0], [])

    def test_only_meters_are_read(self) -> None:
        page = Page()
        page.numbers(300)
        self.assertEqual(page.read("3", "3")[0], [])  # no beat type of 3
        self.assertEqual(page.read("33", "4")[0], [])  # more beats than any meter
        self.assertEqual(page.read("0", "4")[0], [])
        self.assertEqual(page.read("3", "x")[0], [])
        self.assertEqual(meters(page.read("32", "32")[0]), [(32, 32, None)])

    def test_common_time_is_four_four(self) -> None:
        page = Page()
        page.common_time(300)
        self.assertEqual(meters(page.read("C")[0]), [(4, 4, "common")])
        self.assertEqual(page.read("D")[0], [])

    def test_cut_time_is_read_without_its_stroke(self) -> None:
        page = Page()
        page.cut_time(300)
        reads, _, crops = page.read("C")
        self.assertEqual(meters(reads), [(2, 2, "cut")])
        # OCR sees the C alone: no column of the crop is inked most of its height, as
        # the stroke would be (about 0.8 of it with margins; the C's side about 0.56).
        crop = cv2.cvtColor(crops[0], cv2.COLOR_BGR2GRAY) < 128
        self.assertLess(crop.sum(axis=0).max(), 0.7 * crop.shape[0])

    def test_a_tall_glyph_without_a_stroke_is_not_cut_time(self) -> None:
        page = Page()
        cv2.ellipse(page.gray, (300, 140), (15, 30), 0, 40, 320, 0, 6)
        self.assertEqual(page.read("C")[0], [])

    def test_glyphs_outside_the_staff_are_not_candidates(self) -> None:
        page = Page()
        cv2.rectangle(page.gray, (300, 30), (328, 64), 0, -1)
        cv2.rectangle(page.gray, (300, 70), (328, 98), 0, -1)
        reads, glyphs, crops = page.read("3", "4")
        self.assertEqual((reads, glyphs, crops), ([], [], []))


class TestStaffLines(unittest.TestCase):
    def test_lines_go_where_nothing_crosses_them(self) -> None:
        page = Page()
        cv2.line(page.gray, (300, 90), (300, 190), 0, 3)  # a stroke across every line
        ink = ink_without_lines(page.gray, page.staffs)
        self.assertEqual(int(ink[100:102, 50:250].sum()), 0)
        self.assertTrue(ink[90:191, 300].all())

    def test_a_thin_stroke_crossing_a_line_keeps_the_line_there(self) -> None:
        # The stroke's ink stops a pixel short of the line on both sides, as a hairline
        # often does on a scan; the line between joins its two halves.
        page = Page()
        cv2.rectangle(page.gray, (300, 85), (301, 97), 0, -1)
        cv2.rectangle(page.gray, (300, 103), (301, 115), 0, -1)
        ink = ink_without_lines(page.gray, page.staffs)
        self.assertTrue(ink[99:102, 300:302].all())
        self.assertEqual(int(ink[99:102, 250:290].sum()), 0)

    def test_a_glyph_lying_along_a_line_stays(self) -> None:
        page = Page()
        cv2.rectangle(page.gray, (300, 96), (340, 104), 0, -1)  # thicker than the line
        ink = ink_without_lines(page.gray, page.staffs)
        self.assertTrue(ink[96:105, 300:341].all())

    def test_slack_takes_out_a_line_where_a_scan_prints_it_thicker(self) -> None:
        page = Page()
        cv2.line(page.gray, (100, 122), (200, 122), 0, 1)  # a pixel thicker over a stretch
        exact = ink_without_lines(page.gray, page.staffs, 0)
        loose = ink_without_lines(page.gray, page.staffs, 1)
        self.assertGreater(int(exact[117:125, 100:201].sum()), 0)
        self.assertEqual(int(loose[117:125, 100:201].sum()), 0)


def read(x: float, beats: int = 3, beat_type: int = 4) -> TimeSignatureRead:
    return TimeSignatureRead(0, x, float(SPACE), (100.0, 180.0), beats, beat_type, None)


def links(reads: list[TimeSignatureRead], glyphs: list[Glyph] | None = None) -> TimeSignatureLinks:
    future: Future[tuple[list[TimeSignatureRead], list[Glyph]]] = Future()
    future.set_result((reads, glyphs or []))
    result = TimeSignatureLinks()
    result.set_reads(future)
    return result


class TestLinks(unittest.TestCase):
    def test_a_token_takes_the_read_beside_it(self) -> None:
        tied = links([read(300)])
        tied.add_token(1, (310.0, 140.0), 4)
        self.assertEqual((tied.status_of(1), tied.meter_of(1)), (READ, (3, 4)))

    def test_a_read_beyond_reach_is_not_the_tokens(self) -> None:
        # Six staff spaces is the reach, across or along the system.
        tied = links([read(300)])
        tied.add_token(1, (300.0 + 6.5 * SPACE, 140.0), 4)
        tied.add_token(2, (300.0, 180.0 + 2.5 * SPACE), 4)
        tied.add_token(3, (300.0 + 5.5 * SPACE, 140.0), 4)
        self.assertEqual([tied.status_of(i) for i in (1, 2, 3)], [NO_READ, NO_READ, READ])

    def test_a_printed_glyph_without_a_read_is_told_apart_from_nothing(self) -> None:
        glyph = Glyph(300.0, float(SPACE), (100.0, 180.0))
        tied = links([], [glyph])
        tied.add_token(1, (305.0, 140.0), 4)
        tied.add_token(2, (500.0, 140.0), 4)
        tied.add_token(3, None, 4)
        self.assertEqual([tied.status_of(i) for i in (1, 2, 3)], [UNREAD_GLYPH, NO_READ, NO_READ])
        self.assertIsNone(tied.meter_of(1))

    def test_a_symbol_is_reported_only_from_a_read_that_lands(self) -> None:
        common = TimeSignatureRead(0, 300.0, float(SPACE), (100.0, 180.0), 4, 4, "common")
        cut = TimeSignatureRead(0, 500.0, float(SPACE), (100.0, 180.0), 2, 2, "cut")
        tied = links([common, cut, read(700, 3, 4)])
        tied.add_token(1, (300.0, 140.0), 4)
        tied.add_token(2, (500.0, 140.0), 4)  # cut time read where the token says quarters
        tied.add_token(3, (700.0, 140.0), 4)
        self.assertEqual([tied.symbol_of(i) for i in (1, 2, 3, 4)], ["common", None, None, None])

    def test_a_read_with_another_beat_type_is_not_used(self) -> None:
        tied = links([read(300, 6, 8)])
        tied.add_token(1, (300.0, 140.0), 4)
        self.assertEqual((tied.status_of(1), tied.meter_of(1)), (BEAT_TYPE_DIFFERS, None))

    def test_two_tokens_never_share_a_read(self) -> None:
        tied = links([read(300)])
        tied.add_token(1, (340.0, 140.0), 4)
        tied.add_token(2, (310.0, 140.0), 4)
        self.assertEqual([tied.status_of(i) for i in (1, 2)], [NO_READ, READ])
        # A token added after the reads were tied cannot take one already held.
        tied.add_token(3, (300.0, 140.0), 4)
        self.assertEqual(tied.status_of(3), NO_READ)

    def test_the_block_reports_tokens_unmatched_reads_and_meters(self) -> None:
        tied = links([read(300), read(500, 2, 4)])
        tied.carried_in = (2, 4)
        tied.carried_out = (3, 4)
        tied.add_token(1, (300.0, 140.0), 4)
        block = tied.to_json_dict()
        self.assertEqual(block["carried_in"], [2, 4])
        self.assertEqual(block["carried_out"], [3, 4])
        self.assertEqual([t["status"] for t in block["tokens"]], [READ])
        self.assertEqual([r["x"] for r in block["unmatched_reads"]], [500.0])

    def test_reads_without_tokens_are_reported(self) -> None:
        self.assertEqual(len(links([read(300)]).to_json_dict()["unmatched_reads"]), 1)

    def test_without_reads_no_token_is_decided(self) -> None:
        tied = TimeSignatureLinks()
        tied.add_token(1, (300.0, 140.0), 4)
        self.assertFalse(tied.enabled)
        self.assertEqual((tied.status_of(1), tied.meter_of(1)), (None, None))


class TestOptions(unittest.TestCase):
    def test_meters(self) -> None:
        for meter in [(1, 1), (3, 4), (6, 8), (32, 32)]:
            self.assertTrue(is_meter(meter), meter)
        for value in [(0, 4), (33, 4), (3, 3), (3, 64), (3.0, 4), (True, 4), (3, 4, 1), [3, 4]]:
            self.assertFalse(is_meter(value), value)

    def test_tuplets_need_the_time_signatures_read(self) -> None:
        with self.assertRaises(ValueError):
            validate_meter_options(False, True, None)
        with self.assertRaises(ValueError):
            XmlGeneratorArguments(repair_tuplets=True)
        XmlGeneratorArguments(read_time_signatures=True, repair_tuplets=True)

    def test_a_carried_meter_needs_them_read_and_must_be_a_meter(self) -> None:
        with self.assertRaises(ValueError):
            XmlGeneratorArguments(time_signature_from_previous_page=(3, 4))
        with self.assertRaises(ValueError):
            XmlGeneratorArguments(
                read_time_signatures=True, time_signature_from_previous_page=(3, 5)
            )
        args = XmlGeneratorArguments(
            read_time_signatures=True, time_signature_from_previous_page=(3, 4)
        )
        self.assertEqual(args.time_signature_from_previous_page, (3, 4))

    def test_everything_is_off_by_default(self) -> None:
        args = XmlGeneratorArguments()
        self.assertEqual(
            (
                args.read_time_signatures,
                args.repair_tuplets,
                args.time_signature_from_previous_page,
            ),
            (False, False, None),
        )


if __name__ == "__main__":
    unittest.main()


def done(value: object) -> Future[object]:
    future: Future[object] = Future()
    future.set_result(value)
    return future


class TestPageReads(unittest.TestCase):
    """What the sidecar reads from the page, and reports, for each option."""

    def builder(self) -> tuple[VisualSidecarBuilder, list[str]]:
        started: list[str] = []

        def start(name: str, result: object) -> Future[object]:
            started.append(name)
            return done(result)

        builder = VisualSidecarBuilder(identity_transform())
        patches = [
            mock.patch.object(
                builder_module,
                "read_time_signatures_in_background",
                lambda *_: start("time signatures", ([], [])),
            ),
            mock.patch.object(
                builder_module,
                "read_tuplet_numerals_in_background",
                lambda *_: start("numerals", []),
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return builder, started

    def test_nothing_is_read_by_default(self) -> None:
        builder, started = self.builder()
        builder.start_meter_reads("page.png", [], XmlGeneratorArguments())
        self.assertEqual(started, [])
        self.assertFalse(builder.time_signatures.enabled)
        sidecar = builder.to_json_dict()
        self.assertNotIn("time_signatures", sidecar)
        self.assertNotIn("tuplet_repairs", sidecar)

    def test_each_option_starts_its_read(self) -> None:
        builder, started = self.builder()
        args = XmlGeneratorArguments(
            read_time_signatures=True, time_signature_from_previous_page=(6, 8)
        )
        builder.start_meter_reads("page.png", [], args)
        self.assertEqual(started, ["time signatures"])
        self.assertEqual(builder.carried_meter(), (6, 8))
        self.assertEqual(builder.to_json_dict()["time_signatures"]["carried_in"], [6, 8])
        builder, started = self.builder()
        args = XmlGeneratorArguments(read_time_signatures=True, repair_tuplets=True)
        builder.start_meter_reads("page.png", [], args)
        self.assertEqual(started, ["time signatures", "numerals"])

    def test_the_meter_at_the_end_is_only_known_where_the_page_was_read(self) -> None:
        builder, _ = self.builder()
        builder.record_final_meter((3, 4))
        builder.set_carried_time_signature((2, 4))
        self.assertIsNone(builder.time_signature_at_end())
        self.assertIsNone(builder.carried_meter())
        builder.start_meter_reads("page.png", [], XmlGeneratorArguments(read_time_signatures=True))
        builder.record_final_meter((3, 4))
        self.assertEqual(builder.time_signature_at_end(), (3, 4))
        builder.set_carried_time_signature((2, 4))
        self.assertEqual(builder.carried_meter(), (2, 4))

    def test_the_tuplet_block_reports_each_note_by_its_musicxml_id(self) -> None:
        builder, _ = self.builder()
        symbol = EncodedSymbol("note_8", "C5", position="upper")
        builder.visual_groups["vnote-1"] = linked_group("vnote-1")
        builder.visual_groups["vnote-1"].musicxml_id = "homr-note-7"
        builder.matches_by_symbol_id[symbol.visual_match_id] = VisualMatch(
            symbol, "vnote-1", 0.9, "structural"
        )
        record = {"bar": 2, "source": "beam", "recognized": "note_8", "repaired": "note_12"}
        builder.record_tuplet_repairs([{"outcome": "candidate"}], [(symbol, record)])
        block = builder.to_json_dict()["tuplet_repairs"]
        self.assertEqual(block["groups"], [{"outcome": "candidate"}])
        self.assertEqual(
            block["notes"], [{"musicxml_id": "homr-note-7", "visual_group_id": "vnote-1", **record}]
        )
        self.assertEqual(block["numerals"], [])


class TestEntryPoints(unittest.TestCase):
    def test_reading_needs_the_sidecar(self) -> None:
        config = homr_main.ProcessingConfig(
            False, False, False, False, -1, False, False, False, False
        )
        with self.assertRaisesRegex(ValueError, "visual sidecar"):
            homr_main.process_image(
                "page.png", config, XmlGeneratorArguments(read_time_signatures=True)
            )

    def test_the_command_line_refuses_options_that_cannot_work(self) -> None:
        for argv, message in (
            (["homr", "page.png", "--read-time-signatures"], "needs --output-visual-sidecar"),
            (
                ["homr", "page.png", "--output-visual-sidecar", "--repair-tuplets"],
                "needs --read-time-signatures",
            ),
        ):
            stderr = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("sys.stderr", stderr),
                self.assertRaises(SystemExit) as exit_,
            ):
                homr_main.main()
            self.assertEqual(exit_.exception.code, 2)
            self.assertIn(message, stderr.getvalue())
