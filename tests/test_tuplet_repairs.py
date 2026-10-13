# ruff: noqa: S101

import unittest
from collections.abc import Iterable
from fractions import Fraction
from typing import Any

import cv2
import numpy as np

from homr.bar_placement import AFTER_SHORTEST, EARLIEST_END, staff_ends
from homr.music_xml_generator import (
    SymbolChord,
    add_tuplet_start_stop,
    group_into_chords,
)
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar.note_values import NoteValueReading, StemEnd
from homr.visual_sidecar.time_signatures import NO_READ, READ, UNREAD_GLYPH, StaffLines
from homr.visual_sidecar.tuplet_numerals import Numeral, find_numerals
from homr.visual_sidecar.tuplet_repairs import (
    A_STAFF_GETS_WORSE,
    ADDS_UP_AS_RECOGNIZED,
    CANDIDATE,
    HEAD_OFF_STEM,
    LONE_SHORT_NEIGHBOUR,
    NO_TRUSTED_METER,
    NOT_EXACT,
    NOT_PLAIN,
    REST_OR_SINGLE,
    TOO_FEW_SUPPORTING_BARS,
    UNEVEN_SPACING,
    UNITS_DIFFER,
    VALUE_DISAGREES,
    _written_ends,
    repair_tuplets,
    trusted_meters,
)
from tests.test_bar_placement import GRAND_STAFF_IN_A, bars_of, corridors_bar
from training.transformer.training_vocabulary import read_token_lines

#: Stems stand this far apart, one chord group after another; a staff space is SPACE.
STEP = 30.0
SPACE = 10.0
#: Where noteheads of each staff sit, in source pixels.
UPPER_Y, LOWER_Y = 100.0, 200.0
BAR = "barline . . . . ."


def up(rhythm: str, pitch: str = "C5") -> str:
    return f"{rhythm} {pitch} _ _ _ upper"


def low(rhythm: str, pitch: str = "C3") -> str:
    return f"{rhythm} {pitch} _ _ _ lower"


def voice(*bars: list[str]) -> list[SymbolChord]:
    lines: list[str] = []
    for bar in bars:
        lines.extend([*bar, BAR])
    return add_tuplet_start_stop(group_into_chords(read_token_lines(lines)))


#: A 2/4 bar the transformer reads right.
PLAIN = [up("note_4", "E5"), up("note_4", "F5")]
#: A 2/4 bar of two eighth-note triplets, read as six plain eighths.
TRIPLETS = [up("note_8", pitch) for pitch in ("C5", "D5", "E5", "F5", "G5", "A5")]


class Page:
    """What the tuplet repair reads from the page, given directly.

    Each chord's stem stands at STEP times the index of its chord group, or where
    ``stem_x`` puts it; upper stems point up and lower stems down. ``beams`` lists, per
    staff, the group indexes one beam joins. Noteheads lie on their stem, except the
    pitches in ``off_stem``. A stem's free end holds its group index, for the beams.
    """

    def __init__(
        self,
        groups: list[SymbolChord],
        beams: Iterable[tuple[str, Iterable[int]]] = (),
        carried: tuple[int, int] | None = None,
        time_signatures: Iterable[tuple[tuple[int, int] | None, str]] = (),
        stem_x: dict[int, float] | None = None,
        off_stem: Iterable[str] = (),
        readings: dict[str, NoteValueReading] | None = None,
        numerals: Iterable[Numeral] = (),
    ) -> None:
        self.index = {s.visual_match_id: i for i, group in enumerate(groups) for s in group.symbols}
        self.beams = [(staff, set(indexes)) for staff, indexes in beams]
        self.carried = carried
        self.scripted = list(time_signatures)
        self.time_signatures: dict[int, tuple[tuple[int, int] | None, str]] = {}
        self.stem_x = stem_x or {}
        self.off_stem = set(off_stem)
        self.readings = readings or {}
        self.numerals = list(numerals)
        self.groups: list[dict[str, Any]] = []
        self.notes: list[tuple[EncodedSymbol, dict[str, Any]]] = []

    def x(self, symbol: EncodedSymbol) -> float:
        index = self.index[symbol.visual_match_id]
        return self.stem_x.get(index, STEP * index)

    def stem_ends(self, chords: list[list[EncodedSymbol]]) -> list[StemEnd | None]:
        return [
            StemEnd(
                self.x(chord[0]),
                float(self.index[chord[0].visual_match_id]),
                chord[0].position == "upper",
                SPACE,
            )
            for chord in chords
        ]

    def beamed_together(self, first: StemEnd, second: StemEnd) -> bool:
        staff = "upper" if first.up else "lower"
        pair = {int(first.free_end), int(second.free_end)}
        return any(on == staff and pair <= beam for on, beam in self.beams)

    def notehead_prediction_point(self, symbol: EncodedSymbol) -> tuple[float, float]:
        shift = 2 * SPACE if symbol.pitch in self.off_stem else 0.0
        return self.x(symbol) + shift, UPPER_Y if symbol.position == "upper" else LOWER_Y

    def notehead_point(self, symbol: EncodedSymbol) -> tuple[float, float]:
        return self.x(symbol), UPPER_Y if symbol.position == "upper" else LOWER_Y

    def read_token_values(self, symbols: list[EncodedSymbol]) -> dict[int, NoteValueReading]:
        return {id(s): self.readings[s.pitch] for s in symbols if s.pitch in self.readings}

    def tuplet_numerals(self) -> list[Numeral]:
        return self.numerals

    def _time_signature(self, symbol: EncodedSymbol) -> tuple[tuple[int, int] | None, str]:
        if symbol.visual_match_id not in self.time_signatures:
            self.time_signatures[symbol.visual_match_id] = self.scripted.pop(0)
        return self.time_signatures[symbol.visual_match_id]

    def read_time_signature(self, symbol: EncodedSymbol) -> tuple[int, int] | None:
        return self._time_signature(symbol)[0]

    def time_signature_status(self, symbol: EncodedSymbol) -> str | None:
        return self._time_signature(symbol)[1]

    def carried_meter(self) -> tuple[int, int] | None:
        return self.carried

    def record_tuplet_repairs(
        self, groups: list[dict[str, Any]], notes: list[tuple[EncodedSymbol, dict[str, Any]]]
    ) -> None:
        self.groups.extend(groups)
        self.notes.extend(notes)


def rhythms(groups: list[SymbolChord]) -> list[str]:
    return [s.rhythm for group in groups for s in group.symbols if s.rhythm.startswith("note")]


def reasons(page: Page) -> set[str]:
    return {record["reason"] for _, record in page.notes}


def outcomes(page: Page) -> list[str]:
    return [group["outcome"] for group in page.groups]


def triplet_page(*bars: list[str], **options: Any) -> tuple[list[SymbolChord], Page]:
    """The bars, with the six eighths of each TRIPLETS bar in two beam groups."""
    groups = voice(*bars)
    beams = []
    index = 0
    for bar in bars:
        if bar is TRIPLETS:
            beams += [("upper", range(index, index + 3)), ("upper", range(index + 3, index + 6))]
        index += len(bar) + 1
    options.setdefault("carried", (2, 4))
    return groups, Page(groups, beams=beams + list(options.pop("beams", [])), **options)


def reading(
    status: str, printed: str | None = None, dotted: bool | None = False
) -> NoteValueReading:
    return NoteValueReading("", printed, dotted, status, "")


class TestBeamGroups(unittest.TestCase):
    def test_a_beamed_triplet_is_repaired_where_the_bar_then_adds_up(self) -> None:
        groups, page = triplet_page(PLAIN, PLAIN, PLAIN, TRIPLETS)
        repaired = repair_tuplets(groups, page)
        self.assertEqual(rhythms(repaired), ["note_4"] * 6 + ["note_12"] * 6)
        self.assertEqual(outcomes(page), [CANDIDATE, CANDIDATE])
        self.assertEqual({r["ratio"] for r in page.groups}, {"3:2"})
        self.assertEqual(
            [
                (r["bar"], r["source"], r["recognized"], r["repaired"], r["applied"])
                for _, r in page.notes
            ],
            [(4, "beam", "note_8", "note_12", True)] * 6,
        )
        self.assertEqual(reasons(page), {""})

    def test_repaired_groups_get_tuplet_marks(self) -> None:
        groups, page = triplet_page(PLAIN, PLAIN, PLAIN, TRIPLETS)
        marks = [
            g.tuplet_mark for g in repair_tuplets(groups, page) if g.symbols[0].rhythm == "note_12"
        ]
        self.assertEqual(marks, ["start", "", "stop", "start", "", "stop"])

    def test_the_input_groups_are_not_changed(self) -> None:
        groups, page = triplet_page(PLAIN, PLAIN, PLAIN, TRIPLETS)
        repair_tuplets(groups, page)
        self.assertEqual(rhythms(groups), ["note_4"] * 6 + ["note_8"] * 6)

    def test_a_beam_alone_implies_only_a_triplet(self) -> None:
        # Five 16ths under one beam and a dotted quarter: as a quintuplet the bar would
        # add up, but a beam does not say quintuplet.
        bar = [up("note_16", p) for p in ("C5", "D5", "E5", "F5", "G5")] + [up("note_4.", "A5")]
        groups = voice(PLAIN, PLAIN, PLAIN, bar)
        page = Page(groups, beams=[("upper", range(9, 14))], carried=(2, 4))
        self.assertEqual(rhythms(repair_tuplets(groups, page))[-6:], ["note_16"] * 5 + ["note_4."])
        self.assertEqual(outcomes(page), [])

    def test_plain_lengths_and_existing_tuplets_are_no_candidates(self) -> None:
        four = [up("note_8", p) for p in ("C5", "D5", "E5", "F5")]
        already = [up("note_12", p) for p in ("C5", "D5", "E5")] + [up("note_4", "F5")]
        groups = voice(PLAIN, PLAIN, PLAIN, four, already)
        page = Page(
            groups, beams=[("upper", range(9, 13)), ("upper", range(14, 17))], carried=(2, 4)
        )
        self.assertEqual(rhythms(repair_tuplets(groups, page)), rhythms(groups))
        self.assertEqual(page.notes, [])

    def test_a_bar_that_adds_up_as_recognized_is_not_repaired(self) -> None:
        # Three beamed eighths and an eighth rest fill a 2/4 bar as read.
        bar = [up("note_8", p) for p in ("C5", "D5", "E5")] + [up("rest_8", "_")]
        groups = voice(PLAIN, PLAIN, PLAIN, bar)
        page = Page(groups, beams=[("upper", range(9, 12))], carried=(2, 4))
        self.assertEqual(rhythms(repair_tuplets(groups, page))[-3:], ["note_8"] * 3)
        self.assertEqual(reasons(page), {ADDS_UP_AS_RECOGNIZED})

    def test_only_the_staves_a_repair_touches_must_be_broken(self) -> None:
        # The lower staff overflows, but the upper one, which the triplet would change,
        # adds up as read: the lower staff's error is no evidence for the upper one.
        bar = [
            up("note_8", "C5") + "&" + low("note_2", "C3"),
            up("note_8", "D5"),
            up("note_8", "E5") + "&" + low("note_2", "E3"),
            up("rest_8", "_"),
        ]
        groups = voice(PLAIN, PLAIN, PLAIN, bar)
        page = Page(groups, beams=[("upper", range(9, 12))], carried=(2, 4))
        repair_tuplets(groups, page)
        self.assertEqual(reasons(page), {ADDS_UP_AS_RECOGNIZED})

    def test_a_staff_that_adds_up_as_read_placed_either_way_is_not_repaired(self) -> None:
        # By the earliest end only, as in Chopin's Op. 25 No. 7: three beamed 16ths and a
        # 16th rest open the bass under a treble quarter, and later a bass dotted eighth
        # outlasts the treble's eighth. A 32nd at the end leaves the treble past the
        # barline, so the writer places the bar after the shortest note, which starts the
        # bass's last 16th late: the bass is late for the treble's sake, not a triplet's.
        by_the_earliest_end = [
            up("note_4", "E5") + "&" + low("note_16", "C3"),
            low("note_16", "D3"),
            low("note_16", "E3"),
            low("rest_16", "_"),
            up("note_8", "F5") + "&" + up("note_8.", "A5") + "&" + low("note_8.", "G3"),
            up("note_8", "G5"),
            low("note_16", "A3"),
            up("note_32", "B5"),
        ]
        # After the shortest note only: the bass's dotted eighth ends under the treble's
        # second eighth, where the earliest end would start the third.
        after_the_shortest_note = [
            up("note_8", "C5") + "&" + low("note_8.", "G2"),
            up("note_8", "D5"),
            up("note_8", "E5") + "&" + low("note_4", "C3"),
            up("rest_8", "_"),
        ]
        cases = [
            ("by the earliest end", by_the_earliest_end, "lower", slice(7, 10)),
            ("after the shortest note", after_the_shortest_note, "upper", slice(6, 9)),
        ]
        for name, bar, staff, candidates in cases:
            with self.subTest(placed=name):
                groups = voice(PLAIN, PLAIN, PLAIN, bar)
                page = Page(groups, beams=[(staff, range(9, 12))], carried=(2, 4))
                written = rhythms(repair_tuplets(groups, page))
                self.assertEqual(written[candidates], rhythms(groups)[candidates])
                self.assertNotIn("note_12", written)
                self.assertNotIn("note_24", written)
                self.assertEqual(reasons(page), {ADDS_UP_AS_RECOGNIZED})

    def test_a_bar_still_broken_with_the_repairs_is_not_repaired(self) -> None:
        bar = [up("note_8", p) for p in ("C5", "D5", "E5")] + [up("note_2", "F5")]
        groups = voice(PLAIN, PLAIN, PLAIN, bar)
        page = Page(groups, beams=[("upper", range(9, 12))], carried=(2, 4))
        repair_tuplets(groups, page)
        self.assertEqual(reasons(page), {NOT_EXACT})

    def test_a_repair_that_moves_the_other_staff_off_its_length_is_not_applied(self) -> None:
        # The lower quarter starts on the upper staff's third eighth and ends on the
        # barline as read; with the triplets it would start early and end short of it.
        bar = list(TRIPLETS)
        bar[2] = up("note_8", "E5") + "&" + low("note_4", "C3")
        groups = voice(PLAIN, PLAIN, PLAIN, bar)
        page = Page(
            groups,
            beams=[("upper", range(9, 12)), ("upper", range(12, 15))],
            carried=(2, 4),
        )
        repair_tuplets(groups, page)
        self.assertEqual(reasons(page), {A_STAFF_GETS_WORSE})

    def test_bars_without_a_trusted_meter_are_not_repaired(self) -> None:
        groups, page = triplet_page(PLAIN, PLAIN, PLAIN, TRIPLETS, carried=None)
        self.assertEqual(rhythms(repair_tuplets(groups, page))[-6:], ["note_8"] * 6)
        self.assertEqual(reasons(page), {NO_TRUSTED_METER})

    def test_the_bar_length_needs_three_bars_that_add_up(self) -> None:
        # The repaired bar supports its own length: with two plain bars it would pass.
        groups, page = triplet_page(PLAIN, TRIPLETS)
        self.assertEqual(rhythms(repair_tuplets(groups, page))[-6:], ["note_8"] * 6)
        self.assertEqual(reasons(page), {TOO_FEW_SUPPORTING_BARS})

    def test_bars_that_add_up_with_their_own_repairs_support_the_length(self) -> None:
        groups, page = triplet_page(TRIPLETS, TRIPLETS, TRIPLETS)
        self.assertEqual(rhythms(repair_tuplets(groups, page)), ["note_12"] * 18)
        groups, page = triplet_page(TRIPLETS, TRIPLETS)
        self.assertEqual(rhythms(repair_tuplets(groups, page)), ["note_8"] * 12)

    def test_a_short_note_beside_the_group_keeps_it_unrepaired(self) -> None:
        # A lone eighth may be a note the beam detector cut off the group.
        bar = (
            [up("note_8", "B4")]
            + [up("note_8", p) for p in ("C5", "D5", "E5")]
            + [up("note_8", "F5")]
        )
        groups = voice(PLAIN, PLAIN, PLAIN, bar)
        page = Page(groups, beams=[("upper", range(10, 13))], carried=(2, 4))
        repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [LONE_SHORT_NEIGHBOUR])
        self.assertEqual(page.notes, [])

    def test_a_longer_note_beside_the_group_does_not(self) -> None:
        bar = [up("note_8", p) for p in ("C5", "D5", "E5")] + [up("note_4", "F5")]
        groups = voice(PLAIN, PLAIN, PLAIN, bar)
        page = Page(groups, beams=[("upper", range(9, 12))], carried=(2, 4))
        self.assertEqual(rhythms(repair_tuplets(groups, page))[-4:], ["note_12"] * 3 + ["note_4"])

    def test_unevenly_spaced_stems_hold_a_lost_note(self) -> None:
        stems = {12: 360.0, 13: 380.0, 14: 450.0}  # gaps 20 and 70
        groups, page = triplet_page(PLAIN, PLAIN, PLAIN, TRIPLETS, stem_x=stems)
        repaired = repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [CANDIDATE, UNEVEN_SPACING])
        self.assertEqual(rhythms(repaired)[-6:], ["note_8"] * 6)

    def test_spacing_within_the_limit_is_even(self) -> None:
        stems = {12: 360.0, 13: 380.0, 14: 411.0}  # gaps 20 and 31: under 1.6 times
        groups, page = triplet_page(PLAIN, PLAIN, PLAIN, TRIPLETS, stem_x=stems)
        self.assertEqual(rhythms(repair_tuplets(groups, page))[-6:], ["note_12"] * 6)
        stems = {12: 360.0, 13: 380.0, 14: 420.0}  # gaps 20 and 40: twice
        groups, page = triplet_page(PLAIN, PLAIN, PLAIN, TRIPLETS, stem_x=stems)
        repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [CANDIDATE, UNEVEN_SPACING])

    def test_mixed_values_may_be_spaced_wider(self) -> None:
        # A dotted eighth, a 16th and an eighth under a triplet beam: the dotted note's
        # gap is wider.
        bar = [up("note_8.", "C5"), up("note_16", "D5"), up("note_8", "E5"), up("note_4", "F5")]
        groups = voice(PLAIN, PLAIN, PLAIN, bar)
        beams = [("upper", range(9, 12))]
        stems = {9: 270.0, 10: 310.0, 11: 330.0}  # gaps 40 and 20: twice, under 2.2 times
        page = Page(groups, beams=beams, carried=(2, 4), stem_x=stems)
        self.assertEqual(
            rhythms(repair_tuplets(groups, page))[-4:], ["note_12.", "note_24", "note_12", "note_4"]
        )
        stems = {9: 270.0, 10: 320.0, 11: 340.0}  # gaps 50 and 20: 2.5 times
        page = Page(groups, beams=beams, carried=(2, 4), stem_x=stems)
        repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [UNEVEN_SPACING])

    def test_a_notehead_off_its_stem_is_a_merged_note(self) -> None:
        groups, page = triplet_page(PLAIN, PLAIN, PLAIN, TRIPLETS, off_stem={"D5"})
        repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [HEAD_OFF_STEM, CANDIDATE])

    def test_printed_values_that_disagree_rule_out_the_group(self) -> None:
        readings = {"E5": reading("disagrees", "16th"), "F5": reading("agrees", "eighth")}
        groups, page = triplet_page(PLAIN, PLAIN, PLAIN, TRIPLETS, readings=readings)
        repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [VALUE_DISAGREES, CANDIDATE])

    def test_dotted_beats_of_a_compound_meter_are_plain(self) -> None:
        six_eight = [up("note_8", p) for p in ("C5", "D5", "E5", "F5", "G5", "A5")]
        groups = voice(six_eight, six_eight, six_eight, six_eight)
        beams = [("upper", range(21, 24)), ("upper", range(24, 27))]
        page = Page(groups, beams=beams, carried=(6, 8))
        repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [])

    def test_a_beam_over_the_whole_bar_is_plain(self) -> None:
        three_eight = [up("note_8", p) for p in ("C5", "D5", "E5")]
        groups = voice(three_eight, three_eight, three_eight, three_eight)
        page = Page(groups, beams=[("upper", range(12, 15))], carried=(3, 8))
        repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [])


class TestWrittenEnds(unittest.TestCase):
    def test_bars_are_measured_as_the_writer_places_them_under_the_trusted_meter(
        self,
    ) -> None:
        """By the earliest end where that alone ends every staff on the bar length, as the
        printed bar of Corridors of Time does, else after the shortest note."""
        bar = Fraction(1)
        for name, sixteenth, rule in (
            ("printed", "note_16", EARLIEST_END),
            ("misread", "note_32", AFTER_SHORTEST),
        ):
            with self.subTest(bar=name):
                _, bars, lone = bars_of([*GRAND_STAFF_IN_A, *corridors_bar(sixteenth)])
                expected = staff_ends(bars[0], rule, bar, lone)
                self.assertEqual(_written_ends(bars[0], bar, lone), expected)
                other = AFTER_SHORTEST if rule == EARLIEST_END else EARLIEST_END
                self.assertNotEqual(staff_ends(bars[0], other, bar, lone), expected)


class TestTrustedMeters(unittest.TestCase):
    def meters(self, carried: tuple[int, int] | None, *scripted: Any) -> list[Any]:
        ts = "timeSignature/4 . . . . ."
        groups = voice(PLAIN, [ts, *PLAIN], [ts, *PLAIN], PLAIN)
        return trusted_meters(groups, Page(groups, carried=carried, time_signatures=scripted))

    def test_the_carried_meter_holds_until_a_read_changes_it(self) -> None:
        self.assertEqual(
            self.meters((2, 4), ((3, 4), READ), ((4, 4), READ)),
            [(2, 4), (3, 4), (4, 4), (4, 4), (4, 4)],
        )

    def test_a_token_with_nothing_printed_keeps_the_trusted_meter(self) -> None:
        self.assertEqual(
            self.meters((2, 4), (None, NO_READ), ((3, 4), READ)),
            [(2, 4), (2, 4), (3, 4), (3, 4), (3, 4)],
        )

    def test_a_printed_glyph_without_a_read_ends_it(self) -> None:
        self.assertEqual(
            self.meters((2, 4), (None, UNREAD_GLYPH), (None, NO_READ)),
            [(2, 4), None, None, None, None],
        )

    def test_without_a_carried_meter_nothing_is_trusted_before_a_read(self) -> None:
        self.assertEqual(
            self.meters(None, (None, NO_READ), ((2, 4), READ)),
            [None, None, (2, 4), (2, 4), (2, 4)],
        )


def numeral(x0: float, x1: float, number: int = 3, y: float = 60.0) -> Numeral:
    return Numeral(0, 0, (x0 + x1) / 2, y, number, True, (x0, x1), SPACE)


#: A 2/4 bar with a quarter-note triplet on its first beat: a quarter and an eighth under
#: a "3", then a quarter. As read the bar is an eighth too long.
QUARTER_TRIPLET = [up("note_4", "C5"), up("note_8", "D5"), up("note_4", "E5")]


class TestNumerals(unittest.TestCase):
    def page(
        self, bar: list[str], *numerals: Numeral, **options: Any
    ) -> tuple[list[SymbolChord], Page]:
        groups = voice(PLAIN, PLAIN, PLAIN, bar)
        return groups, Page(groups, carried=(2, 4), numerals=numerals, **options)

    def test_the_notes_under_a_bracket_take_its_ratio(self) -> None:
        groups, page = self.page(QUARTER_TRIPLET, numeral(265, 305))
        self.assertEqual(
            rhythms(repair_tuplets(groups, page))[-3:], ["note_6", "note_12", "note_4"]
        )
        self.assertEqual(outcomes(page), [CANDIDATE])
        self.assertEqual({r["source"] for _, r in page.notes}, {"numeral"})

    def test_printed_values_correct_misreads_under_the_bracket(self) -> None:
        # The quarter is read as an eighth and the eighth as a 16th; the page prints
        # what the bracket holds.
        bar = [up("note_8", "C5"), up("note_16", "D5"), up("note_4", "E5")]
        readings = {"C5": reading("disagrees", "quarter"), "D5": reading("disagrees", "eighth")}
        groups, page = self.page(bar, numeral(265, 305), readings=readings)
        self.assertEqual(
            rhythms(repair_tuplets(groups, page))[-3:], ["note_6", "note_12", "note_4"]
        )
        groups, page = self.page(bar, numeral(265, 305))
        self.assertEqual(
            rhythms(repair_tuplets(groups, page))[-3:], ["note_8", "note_16", "note_4"]
        )
        self.assertEqual(reasons(page), {NOT_EXACT})

    def test_notes_that_do_not_add_up_to_the_numeral_are_not_repaired(self) -> None:
        bar = [up("note_8", "C5"), up("note_8", "D5"), up("note_4", "E5"), up("note_8", "F5")]
        groups, page = self.page(bar, numeral(265, 305))
        repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [UNITS_DIFFER])

    def test_a_bracket_over_a_rest_or_one_note_is_not_used(self) -> None:
        bar = [up("note_4", "C5"), up("rest_8", "_"), up("note_4", "E5")]
        groups, page = self.page(bar, numeral(265, 335), numeral(325, 335))
        repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [REST_OR_SINGLE, REST_OR_SINGLE])

    def test_notes_already_in_a_tuplet_are_not_repaired(self) -> None:
        bar = [up("note_6", "C5"), up("note_12", "D5"), up("note_4", "E5")]
        groups, page = self.page(bar, numeral(265, 305))
        repair_tuplets(groups, page)
        self.assertEqual(outcomes(page), [NOT_PLAIN])

    def test_the_staff_nearer_the_numeral_takes_it(self) -> None:
        bar = [
            up("note_4", "C5") + "&" + low("note_4", "C3"),
            up("note_8", "D5") + "&" + low("note_8", "D3"),
            up("note_4", "E5") + "&" + low("note_4", "E3"),
        ]
        groups, page = self.page(bar, numeral(265, 305, y=160.0))
        repair_tuplets(groups, page)
        self.assertEqual([g["staff"] for g in page.groups], ["lower"])

    def test_a_numeral_takes_precedence_over_a_beam_on_the_same_notes(self) -> None:
        groups, page = self.page(
            QUARTER_TRIPLET, numeral(265, 305), beams=[("upper", range(9, 11))]
        )
        repair_tuplets(groups, page)
        self.assertEqual({r["source"] for _, r in page.notes}, {"numeral"})


if __name__ == "__main__":
    unittest.main()


class NumeralPage:
    """A staff with a tuplet numeral drawn above or below it, its bracket optional."""

    LINES = (100, 120, 140, 160, 180)

    def __init__(self) -> None:
        self.gray = np.full((300, 600), 255, dtype=np.uint8)
        for y in self.LINES:
            cv2.line(self.gray, (0, y), (599, y), 0, 2)
        self.staffs = [
            StaffLines(
                0,
                np.array([0.0, 599.0]),
                [np.array([y, y], dtype=float) for y in self.LINES],
                20.0,
                (100.0, 180.0),
            )
        ]

    def numeral(self, x: int, above: bool = True, bracket: bool = True, hooks: bool = True) -> None:
        top = 40 if above else 216
        cv2.rectangle(self.gray, (x, top), (x + 15, top + 23), 0, -1)
        if not bracket:
            return
        middle, toward = top + 12, 1 if above else -1
        for start, end in ((x - 4, x - 30), (x + 19, x + 45)):
            cv2.line(self.gray, (start, middle), (end, middle), 0, 1)
            if hooks:
                cv2.line(self.gray, (end, middle), (end, middle + toward * 10), 0, 1)

    def find(self, text: str = "3") -> list[Numeral]:
        return find_numerals(self.gray, self.staffs, lambda _crop: text)


class TestNumeralReader(unittest.TestCase):
    def test_a_bracketed_numeral_is_read_with_its_bracket(self) -> None:
        page = NumeralPage()
        page.numeral(300)
        [found] = page.find()
        self.assertEqual((found.number, found.above, found.bracket), (3, True, (270.0, 345.0)))
        page = NumeralPage()
        page.numeral(300, above=False)
        [found] = page.find("5")
        self.assertEqual((found.number, found.above), (5, False))

    def test_a_digit_without_a_bracket_is_not_a_numeral(self) -> None:
        page = NumeralPage()
        page.numeral(300, bracket=False)
        self.assertEqual(page.find(), [])

    def test_a_line_beside_the_digit_without_hooks_is_no_bracket(self) -> None:
        page = NumeralPage()
        page.numeral(300, hooks=False)
        self.assertEqual(page.find(), [])

    def test_only_tuplet_digits_count(self) -> None:
        page = NumeralPage()
        page.numeral(300)
        self.assertEqual(page.find("4"), [])
        self.assertEqual([n.number for n in page.find("з")], [3])  # noqa: RUF001

    def test_a_number_at_the_start_of_the_system_is_a_measure_number(self) -> None:
        page = NumeralPage()
        page.numeral(40)
        self.assertEqual(page.find(), [])
