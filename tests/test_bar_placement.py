# ruff: noqa: S101

import unittest
import xml.etree.ElementTree as ET
from fractions import Fraction
from typing import Any

from homr.bar_placement import (
    AFTER_SHORTEST,
    EARLIEST_END,
    BarPlacement,
    after_shortest_advance,
    earliest_end_preferred,
    staff_ends,
)
from homr.music_xml_generator import (
    SymbolChord,
    XmlGeneratorArguments,
    _bar_length_evidence,
    _lone_rest_ids,
    _written_bar_length,
    add_tuplet_start_stop,
    generate_xml,
    group_into_chords,
    split_mixed_chords,
)
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar.time_signatures import (
    BEAT_TYPE_DIFFERS,
    NO_READ,
    READ,
    UNREAD_GLYPH,
)
from homr.visual_sidecar.timing_repairs import TimingRepairs
from tests.test_timing_repairs import (
    CLEAN_BAR,
    EIGHTH_OVER_16TH,
    MeterSidecar,
    StubbedSidecar,
    butterfly_bar,
)
from training.transformer.training_vocabulary import read_token_lines

GRAND_STAFF_IN_A = [
    "clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower",
    "keySignature_3 . . . . .",
    "timeSignature/4 . . . . .",
]


def corridors_bar(sixteenth: str) -> list[str]:
    """Bar 33 of Chrono Trigger's Corridors of Time: a whole-note chord held over a bass
    arpeggio in eighths, under which a lower voice plays a dotted eighth and a 16th. The
    transformer reads the bar like this, but with the 16th as ``sixteenth``."""
    return [
        "note_1 F4 _ _ _ upper&note_1 C4 _ _ _ upper&note_1 A3 _ _ _ upper"
        + "&note_8 D3 _ _ _ lower&note_8. D2 _ _ _ lower",
        "note_8 F3 _ _ _ lower",
        f"{sixteenth} D2 _ _ _ lower",
        "note_8 B3 _ _ _ lower&rest_8 _ _ _ _ lower",
        "note_8 C4 _ _ _ upper&note_8 D2 _ _ _ lower",
        "note_8 F4 _ _ _ upper&note_4 D2 _ _ _ lower",
        "note_8 F3 _ _ _ lower",
        "note_8 B3 _ _ _ lower&note_8 A2 _ _ _ lower",
        "note_8 C4 _ _ _ lower&note_8 D3 _ _ _ lower",
        "barline . . . . .",
    ]


#: Bar 33 as printed, in quarter notes: what the earliest-end rule makes of the right tokens.
PRINTED_ONSETS = sorted(
    [
        ("1", "F4", 0.0),
        ("1", "C4", 0.0),
        ("1", "A3", 0.0),
        ("1", "C4", 1.5),
        ("1", "F4", 2.0),
        ("2", "D3", 0.0),
        ("2", "D2", 0.0),
        ("2", "F3", 0.5),
        ("2", "D2", 0.75),
        ("2", "B3", 1.0),
        ("2", "rest", 1.0),
        ("2", "D2", 1.5),
        ("2", "D2", 2.0),
        ("2", "F3", 2.5),
        ("2", "B3", 3.0),
        ("2", "A2", 3.0),
        ("2", "C4", 3.5),
        ("2", "D3", 3.5),
    ]
)

#: Bar 33 as printed, each group starting after the shortest note of the group before: the
#: 16th starts when the bass's eighth ends instead of when the dotted eighth does, and every
#: note after it a quarter beat late.
PRINTED_BAR_LATE_ONSETS = sorted(
    [
        ("1", "F4", 0.0),
        ("1", "C4", 0.0),
        ("1", "A3", 0.0),
        ("1", "C4", 1.75),
        ("1", "F4", 2.25),
        ("2", "D3", 0.0),
        ("2", "D2", 0.0),
        ("2", "F3", 0.5),
        ("2", "D2", 1.0),
        ("2", "B3", 1.25),
        ("2", "rest", 1.25),
        ("2", "D2", 1.75),
        ("2", "D2", 2.25),
        ("2", "F3", 2.75),
        ("2", "B3", 3.25),
        ("2", "A2", 3.25),
        ("2", "C4", 3.75),
        ("2", "D3", 3.75),
    ]
)

#: Bar 33 with the 16th read as a 32nd, each group starting after the shortest note of the
#: group before: the misread only moves the notes after it an eighth of a beat.
SHIFTED_ONSETS = sorted(
    [
        ("1", "F4", 0.0),
        ("1", "C4", 0.0),
        ("1", "A3", 0.0),
        ("1", "C4", 1.625),
        ("1", "F4", 2.125),
        ("2", "D3", 0.0),
        ("2", "D2", 0.0),
        ("2", "F3", 0.5),
        ("2", "D2", 1.0),
        ("2", "B3", 1.125),
        ("2", "rest", 1.125),
        ("2", "D2", 1.625),
        ("2", "D2", 2.125),
        ("2", "F3", 2.625),
        ("2", "B3", 3.125),
        ("2", "A2", 3.125),
        ("2", "C4", 3.625),
        ("2", "D3", 3.625),
    ]
)

#: Bar 25 of Corridors of Time as the transformer reads it: bar 33's misread under a
#: five-note chord, with the second half of the arpeggio written on the treble staff.
CORRIDORS_BAR_25 = [
    "note_8. D2 _ _ _ lower&note_8 D3 _ _ _ lower&note_1 G5 # _ _ upper&note_1 G4 # _ _ upper"
    + "&note_1 C5 # _ _ upper&note_1 A5 _ arpeggiate _ upper&note_1 A4 _ _ _ upper",
    "note_8 F3 # _ _ lower",
    "note_32 D2 _ staccato _ lower",
    "rest_8 _ _ _ _ lower&note_8 B3 _ _ _ lower",
    "note_8 D2 _ _ slurStart lower&note_8 C4 # _ _ upper",
    "note_8 F4 # _ _ upper&note_4 D2 _ _ slurStop lower",
    "note_8 F3 # _ _ upper",
    "note_8 B3 _ _ _ upper&note_8 A2 _ _ _ lower",
    "note_8 D3 _ _ _ lower&note_8 C4 # _ _ upper",
    "barline . . . . .",
]

#: Bar 25 with each group starting after the shortest note of the group before.
BAR_25_SHIFTED_ONSETS = sorted(
    [
        *[("1", pitch, 0.0) for pitch in ("G5", "G4", "C5", "A5", "A4")],
        ("1", "C4", 1.625),
        ("1", "F4", 2.125),
        ("1", "F3", 2.625),
        ("1", "B3", 3.125),
        ("1", "C4", 3.625),
        ("2", "D2", 0.0),
        ("2", "D3", 0.0),
        ("2", "F3", 0.5),
        ("2", "D2", 1.0),
        ("2", "rest", 1.125),
        ("2", "B3", 1.125),
        ("2", "D2", 1.625),
        ("2", "D2", 2.125),
        ("2", "A2", 3.125),
        ("2", "D3", 3.625),
    ]
)

#: Bar 26 of Corridors of Time as the transformer reads it: the 16th is read right, but the
#: dotted eighth before it is read as a plain eighth, under two half-note chords.
CORRIDORS_BAR_26 = [
    "note_8 D2 _ _ _ lower&note_2 F4 # _ _ upper&note_2 E4 _ _ _ upper&note_2 B4 _ _ _ upper",
    "note_8 F3 # _ _ lower",
    "note_16 D2 _ staccato _ lower",
    "rest_8 _ _ _ _ lower&note_8 B3 _ _ _ lower",
    "note_8 D2 _ _ slurStart lower&note_8 C4 # _ _ upper",
    "note_4 D2 _ _ slurStop lower&note_2 F4 # _ _ upper&note_2 E5 _ _ _ upper"
    + "&note_2 B4 _ _ _ upper",
    "note_8 F3 # _ _ lower",
    "note_8 B3 _ _ _ lower&note_8 A2 _ _ _ lower",
    "note_8 D3 _ _ _ lower&note_8 C4 # _ _ lower",
    "barline . . . . .",
]

#: Bar 26 with each group starting after the shortest note of the group before: the second
#: chord a quarter beat late, the bar overflowing.
BAR_26_SHIFTED_ONSETS = sorted(
    [
        *[("1", pitch, 0.0) for pitch in ("F4", "E4", "B4")],
        ("1", "C4", 1.75),
        *[("1", pitch, 2.25) for pitch in ("F4", "E5", "B4")],
        ("2", "D2", 0.0),
        ("2", "F3", 0.5),
        ("2", "D2", 1.0),
        ("2", "rest", 1.25),
        ("2", "B3", 1.25),
        ("2", "D2", 1.75),
        ("2", "D2", 2.25),
        ("2", "F3", 3.25),
        ("2", "B3", 3.75),
        ("2", "A2", 3.75),
        ("2", "D3", 4.25),
        ("2", "C4", 4.25),
    ]
)


def write(lines: list[str], sidecar: MeterSidecar | None = None) -> Any:
    return generate_xml(XmlGeneratorArguments(), [read_token_lines(lines)], "", sidecar)


def read_meter(lines: list[str], meter: tuple[int, int] = (4, 4)) -> MeterSidecar:
    """A sidecar that reads ``meter`` at every time signature of ``lines``: a trusted meter."""
    count = sum(line.startswith("timeSignature") for line in lines)
    return MeterSidecar([(meter, READ)] * count)


def _measures(xml: Any) -> list[ET.Element]:
    root = ET.fromstring(xml.to_string())  # noqa: S314 - generated by the test itself
    part = root.find("part")
    assert part is not None
    return part.findall("measure")


def _divisions(xml: Any) -> int:
    return int(_measures(xml)[0].findtext("attributes/divisions", "1"))


def _timed_notes(xml: Any, index: int) -> list[tuple[str, str, Fraction, Fraction]]:
    """(staff, pitch, onset, duration) in quarter notes for every note of one measure."""
    divisions = _divisions(xml)
    time, onset, result = 0, 0, []
    for element in _measures(xml)[index]:
        if element.tag == "backup":
            time -= int(element.findtext("duration", "0"))
        elif element.tag == "forward":
            time += int(element.findtext("duration", "0"))
        elif element.tag == "note":
            duration = int(element.findtext("duration", "0"))
            if element.find("chord") is None:
                onset = time
                time += duration
            pitch = element.find("pitch")
            name = (
                "rest"
                if pitch is None
                else pitch.findtext("step", "") + pitch.findtext("octave", "")
            )
            result.append(
                (
                    element.findtext("staff", "1"),
                    name,
                    Fraction(onset, divisions),
                    Fraction(duration, divisions),
                )
            )
    return result


def onsets(xml: Any, index: int = 0) -> list[tuple[str, str, float]]:
    return sorted((s, n, float(o)) for s, n, o, _ in _timed_notes(xml, index))


def written_bar_length(xml: Any) -> Fraction:
    """The bar length of the time signature the writer wrote, in whole notes."""
    beats = _measures(xml)[0].findtext("attributes/time/beats")
    beat_type = _measures(xml)[0].findtext("attributes/time/beat-type")
    assert beats is not None and beat_type is not None
    return Fraction(int(beats), int(beat_type))


def written_staff_ends(xml: Any, index: int = 0) -> dict[str, Fraction]:
    """When each staff's last note ends in one written measure, in whole notes."""
    ends: dict[str, Fraction] = {}
    for staff, _, onset, duration in _timed_notes(xml, index):
        key = "upper" if staff == "1" else "lower"
        ends[key] = max(ends.get(key, Fraction(0)), (onset + duration) / 4)
    return ends


def bars_of(
    lines: list[str],
) -> tuple[list[SymbolChord], list[list[SymbolChord]], dict[int, Fraction]]:
    """The writer's groups for these tokens, the groups of each bar, and the lone rests."""
    groups = split_mixed_chords(add_tuplet_start_stop(group_into_chords(read_token_lines(lines))))
    bars: list[list[SymbolChord]] = [[]]
    for group in groups:
        if group.is_barline():
            bars.append([])
        else:
            bars[-1].append(group)
    return groups, bars, _lone_rest_ids(groups)


def first_timed_group(groups: list[SymbolChord]) -> int:
    return next(i for i, g in enumerate(groups) if g.symbols[0].rhythm.startswith("note"))


#: A 2/4 bar whose quarter C5 shares its head with a 16th the transformer does not read,
#: under which a second treble voice moves in eighths.
LOST_16TH_BAR = [
    "note_4 C5 _ _ _ upper&note_2 C3 _ _ _ lower",
    "note_8 D5 _ _ _ upper",
    "note_8 E5 _ _ _ upper",
    "note_4 G5 _ _ _ upper",
    "note_8 F5 _ _ _ upper",
    "barline . . . . .",
]


#: Super Mario Bros' Ground Theme, a 2/2 bar as the transformer reads it: the bass holds a
#: quarter under the treble's quarter chord.
STAGGERED_BAR = [
    "note_8 C5 _ _ _ upper&note_4 A2 _ _ _ lower",
    "note_4 C5 _ _ _ upper",
    "rest_8 _ _ _ _ lower",
    "note_8 C5 _ _ _ upper&note_8 E3 _ _ _ lower",
    "rest_8 _ _ _ _ upper&rest_4 _ _ _ _ lower",
    "note_8 C5 _ _ _ upper",
    "note_4 D5 _ _ _ upper&note_4 A3 _ _ _ lower",
    "barline . . . . .",
]


class NoRepairs:
    """Timing repairs with nothing applied, or with lost 16ths in the given groups."""

    def __init__(self, repaired: list[SymbolChord] | None = None) -> None:
        self.repaired = [id(group.symbols[0]) for group in repaired or []]

    def lost_note_lengths(self, group: SymbolChord) -> list[Fraction]:
        return [Fraction(1, 16)] if id(group.symbols[0]) in self.repaired else []


class TestBarPlacement(unittest.TestCase):
    def test_without_a_trusted_meter_every_bar_is_placed_after_the_shortest_note(self) -> None:
        """
        The written 4/4 is inferred from the page's bars, so no bar can prove the
        earliest-end placement: the printed bar overflows by a 16th, one late hiccup, and
        the misread bar shifts its notes after the misread instead of rushing them.
        """
        xml = write([*GRAND_STAFF_IN_A, *corridors_bar("note_16"), *corridors_bar("note_32")])
        self.assertEqual(onsets(xml, 0), PRINTED_BAR_LATE_ONSETS)
        self.assertEqual(onsets(xml, 1), SHIFTED_ONSETS)

    def test_a_bar_the_earliest_end_ends_on_a_trusted_meter_is_written_that_way(self) -> None:
        lines = [*GRAND_STAFF_IN_A, *corridors_bar("note_16")]
        self.assertEqual(onsets(write(lines, read_meter(lines))), PRINTED_ONSETS)

    def test_a_misread_that_would_rush_the_bar_moves_the_rest_a_little_instead(self) -> None:
        """
        With the 16th read as a 32nd, starting groups when the earliest note ends puts the
        next group off the beat; each later group then starts at whichever of two ends comes
        first, so the bass rushes and stops at beat 2.5 while the chord holds to 4. That
        does not end every staff on the trusted 4/4, so the bar is placed after the
        shortest note: the notes after the misread start an eighth of a beat late and keep
        their spacing.
        """
        lines = [*GRAND_STAFF_IN_A, *corridors_bar("note_32")]
        _, bars, lone = bars_of(lines)
        earliest_end = staff_ends(bars[0], EARLIEST_END, Fraction(1), lone)
        self.assertEqual(earliest_end, {"upper": Fraction(1), "lower": Fraction(5, 8)})
        self.assertEqual(onsets(write(lines, read_meter(lines))), SHIFTED_ONSETS)

    def test_each_bar_is_placed_by_its_own_rule(self) -> None:
        lines = [*GRAND_STAFF_IN_A, *corridors_bar("note_32"), *corridors_bar("note_16")]
        xml = write(lines, read_meter(lines))
        self.assertEqual(onsets(xml, 0), SHIFTED_ONSETS)
        self.assertEqual(onsets(xml, 1), PRINTED_ONSETS)

    def test_a_rushed_bar_whose_shift_ends_a_staff_just_past_the_barline_is_shifted(
        self,
    ) -> None:
        """
        Bar 25 has bar 33's misread, but the second half of its arpeggio is on the treble
        staff. Starting groups after the shortest note then ends both staves a 32nd past
        the barline, and the earliest-end placement stops the bass a beat and a half short.
        """
        lines = [*GRAND_STAFF_IN_A, *CORRIDORS_BAR_25]
        _, bars, lone = bars_of(lines)
        bar = Fraction(1)
        self.assertEqual(
            staff_ends(bars[0], EARLIEST_END, bar, lone),
            {"upper": Fraction(1), "lower": Fraction(5, 8)},
        )
        self.assertEqual(
            staff_ends(bars[0], AFTER_SHORTEST, bar, lone),
            {"upper": Fraction(33, 32), "lower": Fraction(33, 32)},
        )
        self.assertEqual(onsets(write(lines, read_meter(lines))), BAR_25_SHIFTED_ONSETS)

    def test_a_rushed_bar_whose_shift_overflows_is_shifted(self) -> None:
        """
        Bar 26 reads its dotted eighth as a plain eighth, and the earliest-end placement
        stops the bass an eighth note short. Starting groups after the shortest note moves
        the second chord a quarter beat late and overflows both staves: a late hiccup
        instead of a rush.
        """
        lines = [*GRAND_STAFF_IN_A, *CORRIDORS_BAR_26]
        _, bars, lone = bars_of(lines)
        bar = Fraction(1)
        self.assertEqual(
            staff_ends(bars[0], EARLIEST_END, bar, lone),
            {"upper": Fraction(1), "lower": Fraction(7, 8)},
        )
        self.assertEqual(
            staff_ends(bars[0], AFTER_SHORTEST, bar, lone),
            {"upper": Fraction(17, 16), "lower": Fraction(19, 16)},
        )
        self.assertEqual(onsets(write(lines, read_meter(lines))), BAR_26_SHIFTED_ONSETS)

    def test_the_earliest_end_is_preferred_only_where_it_alone_ends_every_staff_on_the_bar(
        self,
    ) -> None:
        bar = Fraction(1)
        on = {"upper": bar, "lower": bar}
        cases = [
            ("the other overflows", on, {"upper": bar, "lower": Fraction(17, 16)}, True),
            ("the other stops short", on, {"upper": Fraction(7, 8), "lower": bar}, True),
            ("both end on it", on, on, False),
            ("a staff stops short", {"upper": bar, "lower": Fraction(7, 8)}, {}, False),
            ("a staff overflows", {"upper": bar, "lower": Fraction(9, 8)}, {}, False),
            ("its only staff", {"upper": bar}, {"upper": Fraction(9, 8)}, True),
            ("nothing measured", {}, {}, False),
        ]
        for name, earliest_end, after_shortest, preferred in cases:
            with self.subTest(case=name):
                self.assertEqual(
                    earliest_end_preferred(earliest_end, after_shortest, bar), preferred
                )

    def test_only_a_read_or_carried_meter_is_trusted(self) -> None:
        """The printed bar is placed by the earliest end exactly where its meter is trusted."""
        printed = corridors_bar("note_16")
        with_token = [*GRAND_STAFF_IN_A, *printed]
        without_token = [*GRAND_STAFF_IN_A[:2], *printed]
        cases = [
            ("inferred", with_token, None, PRINTED_BAR_LATE_ONSETS),
            ("read", with_token, MeterSidecar([((4, 4), READ)]), PRINTED_ONSETS),
            (
                "read with another beat type",
                with_token,
                MeterSidecar([(None, BEAT_TYPE_DIFFERS)]),
                PRINTED_BAR_LATE_ONSETS,
            ),
            ("carried", without_token, MeterSidecar(carried=(4, 4)), PRINTED_ONSETS),
            (
                "carried past a token with nothing printed",
                with_token,
                MeterSidecar([(None, NO_READ)], carried=(4, 4)),
                PRINTED_ONSETS,
            ),
            (
                "carried up to a glyph no read made out",
                with_token,
                MeterSidecar([(None, UNREAD_GLYPH)], carried=(4, 4)),
                PRINTED_BAR_LATE_ONSETS,
            ),
        ]
        for name, lines, sidecar, expected in cases:
            with self.subTest(meter=name):
                self.assertEqual(onsets(write(lines, sidecar)), expected)

    def test_without_a_trusted_bar_length_the_bar_is_placed_after_the_shortest_note(
        self,
    ) -> None:
        groups, _, lone = bars_of([*GRAND_STAFF_IN_A, *corridors_bar("note_16")])
        first = first_timed_group(groups)
        repairs: Any = NoRepairs()
        self.assertEqual(BarPlacement(groups, lone, repairs).rule(first, None), AFTER_SHORTEST)
        self.assertEqual(BarPlacement(groups, lone, repairs).rule(first, Fraction(1)), EARLIEST_END)

    def test_a_lost_shorter_note_counts_among_its_groups_notes_under_either_rule(
        self,
    ) -> None:
        """
        The second treble voice starts when the lost 16th ends. Counted, it ends the bar on
        the barline by the earliest end, so the bar is placed that way; uncounted, neither
        rule ends it there and it is placed after the shortest note.
        """
        lines = [*GRAND_STAFF_IN_A, *LOST_16TH_BAR]
        groups, bars, lone = bars_of(lines)
        first = first_timed_group(groups)
        repaired: Any = NoRepairs([groups[first]])
        bar = Fraction(1, 2)
        self.assertEqual(
            staff_ends(bars[0], EARLIEST_END, bar, lone, repaired.lost_note_lengths),
            {"upper": bar, "lower": bar},
        )
        self.assertEqual(
            staff_ends(bars[0], AFTER_SHORTEST, bar, lone, repaired.lost_note_lengths),
            {"upper": Fraction(11, 16), "lower": bar},
        )
        self.assertEqual(
            staff_ends(bars[0], EARLIEST_END, bar, lone),
            {"upper": Fraction(7, 8), "lower": bar},
        )
        self.assertEqual(BarPlacement(groups, lone, repaired).rule(first, bar), EARLIEST_END)
        unrepaired: Any = NoRepairs()
        self.assertEqual(BarPlacement(groups, lone, unrepaired).rule(first, bar), AFTER_SHORTEST)

    def test_a_bar_is_measured_whole_for_the_meter_and_as_written_for_repairs(self) -> None:
        """
        The typical bar length, which gives a guessed meter its beat count, takes the
        staggered bar by the earliest end: one 2/2 bar. A shared-notehead repair is checked
        against the bar as the writer places it: after the shortest note it overflows by
        an eighth, unless a trusted 2/2 lets the earliest end prove the bar.
        """
        _, bars, _ = bars_of(STAGGERED_BAR)
        chords = [group.symbols for group in bars[0]]
        self.assertEqual(_bar_length_evidence(chords), Fraction(1))
        self.assertEqual(_written_bar_length(chords, None), Fraction(9, 8))
        self.assertEqual(_written_bar_length(chords, Fraction(1)), Fraction(1))

    def test_lost_note_lengths_are_the_applied_repairs_and_record_nothing(self) -> None:
        lines = [
            "clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower",
            "keySignature_0 . . . . .",
            "timeSignature/4 . . . . .",
            *butterfly_bar("B4", "D5"),
            *CLEAN_BAR,
        ]
        sidecar = StubbedSidecar({"B4": EIGHTH_OVER_16TH, "D5": EIGHTH_OVER_16TH})
        groups = add_tuplet_start_stop(group_into_chords(read_token_lines(lines)))
        repairs = TimingRepairs(groups, sidecar)
        lengths = [repairs.lost_note_lengths(group) for group in groups]
        self.assertEqual(
            [
                [s.pitch for s in group.symbols if s.position == "upper"]
                for group, found in zip(groups, lengths, strict=True)
                if found
            ],
            [["B4"], ["D5"]],
        )
        self.assertEqual([found for found in lengths if found], [[Fraction(1, 16)]] * 2)
        self.assertEqual(sidecar.state.shared_notehead_records, [])

    def test_grace_notes_rests_and_lost_notes_in_the_shortest_note(self) -> None:
        def chord(*rhythms: str) -> SymbolChord:
            return SymbolChord([EncodedSymbol(r, "C4", position="upper") for r in rhythms])

        self.assertEqual(after_shortest_advance(chord("note_4", "note_8G")), Fraction(1, 4))
        self.assertEqual(after_shortest_advance(chord("note_8G")), Fraction(0))
        self.assertEqual(after_shortest_advance(chord("note_2", "rest_8")), Fraction(1, 8))
        self.assertEqual(after_shortest_advance(chord("note_12", "note_4")), Fraction(1, 12))
        self.assertEqual(
            after_shortest_advance(chord("note_4"), [Fraction(1, 16)]), Fraction(1, 16)
        )
        self.assertEqual(
            after_shortest_advance(chord("note_16"), [Fraction(1, 8)]), Fraction(1, 16)
        )

    def test_the_bar_is_measured_as_the_writer_writes_it(self) -> None:
        """The rule is chosen by where each staff ends; that must be where it ends written."""
        bars = {
            "misread 32nd": corridors_bar("note_32"),
            "printed 16th": corridors_bar("note_16"),
            "lone rest": [
                "note_4 C5 _ _ _ upper&rest_2 _ _ _ _ lower",
                "note_4 D5 _ _ _ upper",
                "note_2 E5 _ _ _ upper",
                "barline . . . . .",
            ],
            "lone rest read later": [
                "note_4 C5 _ _ _ upper",
                "note_4 D5 _ _ _ upper&rest_2 _ _ _ _ lower",
                "note_2 E5 _ _ _ upper",
                "barline . . . . .",
            ],
            "grace note and triplets": [
                "note_4 C5 _ _ _ upper&note_2 C3 _ _ _ lower",
                "note_8G D5 _ _ _ upper",
                "note_12 E5 _ _ _ upper",
                "note_12 F5 _ _ _ upper",
                "note_12 G5 _ _ _ upper",
                "note_2 F5 _ _ _ upper&note_2 G2 _ _ _ lower",
                "barline . . . . .",
            ],
            # The writer splits a time signature chorded with notes out of the group, and
            # writes the notes.
            "note under a time signature": [
                "note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower",
                "timeSignature/4 . . . . .&note_4 E3 _ _ _ lower",
                "note_4 D5 _ _ _ upper&note_8 E3 _ _ _ lower",
                "note_2 E5 _ _ _ upper&note_2 G3 _ _ _ lower",
                "barline . . . . .",
            ],
        }
        multirest = ["rest_2m _ _ _ _ upper", "barline . . . . ."]
        for name, bar in bars.items():
            for trusted in (False, True):
                with self.subTest(bar=name, trusted=trusted):
                    lines = [*GRAND_STAFF_IN_A, *bar, *multirest]
                    xml = write(lines, read_meter(lines) if trusted else None)
                    groups, measured, lone = bars_of(lines)
                    bar_length = written_bar_length(xml)
                    placement = BarPlacement(groups, lone, NoRepairs())  # type: ignore[arg-type]
                    trusted_length = bar_length if trusted else None
                    rule = placement.rule(first_timed_group(groups), trusted_length)
                    self.assertEqual(
                        staff_ends(measured[0], rule, bar_length, lone), written_staff_ends(xml)
                    )
                    self.assertEqual(staff_ends(measured[1], rule, bar_length, lone), {})
