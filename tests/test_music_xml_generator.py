# ruff: noqa: E501, S101

import re
import unittest
import xml.etree.ElementTree as ET
from fractions import Fraction
from typing import Any

import musicxml.xmlelement.xmlelement as mxl

from homr.music_xml_generator import (
    SymbolChord,
    XmlGeneratorArguments,
    _split_mixed_chord,
    find_division_and_time_signature_nominator,
    generate_xml,
    rebalance_measure_voices,
)
from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar.time_signatures import READ
from tests.test_timing_repairs import MeterSidecar
from training.transformer.training_vocabulary import (
    read_token_lines,
)


def _written(tokens: list[EncodedSymbol]) -> ET.Element:
    """The MusicXML generate_xml writes for one staff, parsed for ElementTree lookups."""
    written = generate_xml(XmlGeneratorArguments(), [tokens], "").to_string()
    return ET.fromstring(written)  # noqa: S314 - local HOMR artifact


def _notes(measure: ET.Element) -> list[ET.Element]:
    return [c for c in measure if c.tag == "note"]


def _pitch(note: ET.Element) -> str:
    p = note.find("pitch")
    if p is None:
        return "rest"
    step = p.findtext("step", "")
    return step


def _duration(note: ET.Element) -> int:
    d = note.findtext("duration")
    return int(d) if d is not None else 0


def _staff(note: ET.Element) -> str:
    return note.findtext("staff", "")


def _first_measure(xml: ET.Element) -> ET.Element:
    part = xml.find("part")
    assert part is not None
    m = part.find("measure")
    assert m is not None
    return m


def _onsets(xml: Any, rests: bool = True) -> list[tuple[str, str, float]]:
    """(staff, pitch with octave, onset in quarter notes) for every note of the first measure."""
    root = xml if isinstance(xml, ET.Element) else ET.fromstring(xml.to_string())  # noqa: S314
    part = root.find("part")
    assert part is not None
    divisions = int(part.findtext("measure/attributes/divisions", "1"))
    measure = part.find("measure")
    assert measure is not None
    time, onset, result = 0, 0, []
    for element in measure:
        if element.tag == "backup":
            time -= int(element.findtext("duration", "0"))
        elif element.tag == "forward":
            time += int(element.findtext("duration", "0"))
        elif element.tag == "note":
            if element.find("chord") is None:
                onset = time
                time += int(element.findtext("duration", "0"))
            pitch = element.find("pitch")
            name = (
                "rest"
                if pitch is None
                else pitch.findtext("step", "") + pitch.findtext("octave", "")
            )
            if rests or pitch is not None:
                result.append((element.findtext("staff", "1"), name, onset / divisions))
    return sorted(result)


def _write_under_read_meter(tokens: list[EncodedSymbol], meter: tuple[int, int]) -> Any:
    """Writes tokens whose time signature a read of the page confirms as ``meter``. Under
    such a trusted meter a bar may prove the earliest-end placement; see
    homr.bar_placement."""
    sidecar = MeterSidecar([(meter, READ)])
    return generate_xml(XmlGeneratorArguments(), [tokens], "", visual_sidecar=sidecar)


class TestMusicXmlGenerator(unittest.TestCase):
    """
    MusicXML testing is mostly covered by training/validate_music_xml_conversion.py
    This script requires that the data sets are downloaded and converted and uses
    the data sets to check that back and forth conversion works.
    """

    def test_chord_with_different_duratons(self) -> None:
        tabi_measure_18_upper = """clef_G2 . . . . upper
keySignature_4 . . . . .
timeSignature/8 . . . . .
note_4. G3 # _ _ upper &note_4. C4 # _ _ upper&note_16 E4 # _ _ upper
note_16 F4 # _ _ upper
note_4 E4 # _ _ upper
note_8 E4 # _ _ upper
note_8 C4 # _ _ upper
note_8 D4 # _ _ upper
barline . . . . ."""
        tokens = read_token_lines(tabi_measure_18_upper.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        actual = self._xml_to_str(xml)
        expected = """XMLScorePartwise([XMLWork([XMLWorkTitle()]),
XMLPart([XMLMeasure([XMLAttributes([XMLDivisions(value: 4)]),
XMLAttributes([XMLKey([XMLFifths(value: 4)]),
XMLTime([XMLBeats(value: 6),
XMLBeatType(value: 8)]),
XMLClef([XMLSign(value: G),
XMLLine(value: 2)])]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 1),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 1)]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 6),
XMLVoice(value: 2),
XMLDot(),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLChord(),
XMLPitch([XMLStep(value: C)]),
XMLDuration(value: 6),
XMLVoice(value: 2),
XMLDot(),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 5)]),
XMLNote([XMLPitch([XMLStep(value: F)]),
XMLDuration(value: 1),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 4),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: C)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: D)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()])])])])"""
        self.assertEqual(self._norm_expected(expected), actual)

    def test_grand_staff_generation(self) -> None:
        grandstaff = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_1 . . . . .
timeSignature/4 . . . . .
note_1 G4 _ _ _ upper&note_1 A3 # _ _ upper&rest_2 _ _ _ _ upper&note_4 G3 _ _ _ lower
rest_4 _ _ _ _ lower
note_2 E4 _ _ _ upper&note_2 C2 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(grandstaff.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        actual = self._xml_to_str(xml)
        expected = """XMLScorePartwise([XMLWork([XMLWorkTitle()]),
XMLPart([XMLMeasure([XMLAttributes([XMLDivisions(value: 1)]),
XMLAttributes([XMLKey([XMLFifths(value: 1)]),
XMLTime([XMLBeats(value: 4),
XMLBeatType(value: 4)]),
XMLClef([XMLSign(value: G),
XMLLine(value: 2)]),
XMLClef([XMLSign(value: F),
XMLLine(value: 4)])]),
XMLNote([XMLRest(),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 2)]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 4),
XMLVoice(value: 2),
XMLStaff(value: 1),
XMLNotations()]),
XMLNote([XMLChord(),
XMLPitch([XMLStep(value: A)]),
XMLDuration(value: 4),
XMLVoice(value: 2),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 4)]),
XMLNote([XMLPitch([XMLStep(value: G)]),
XMLDuration(value: 1),
XMLVoice(value: 5),
XMLStaff(value: 2),
XMLNotations()]),
XMLNote([XMLRest(),
XMLDuration(value: 1),
XMLVoice(value: 5),
XMLStaff(value: 2),
XMLNotations()]),
XMLNote([XMLPitch([XMLStep(value: E)]),
XMLDuration(value: 2),
XMLVoice(value: 1),
XMLStaff(value: 1),
XMLNotations()]),
XMLBackup([XMLDuration(value: 2)]),
XMLNote([XMLPitch([XMLStep(value: C)]),
XMLDuration(value: 2),
XMLVoice(value: 5),
XMLStaff(value: 2),
XMLNotations()])])])])"""
        self.assertEqual(self._norm_expected(expected), actual)

    def test_begin_chord_with_standalone_rests(self) -> None:
        """
        If the lower position consists of a standalone rest then start the
        chord with this. That fixes an issue where the upper position
        consists of tuplets because in that case backups must not be used.

        See tabi.jpg measure 9 for an example.
        """
        chord = SymbolChord(
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("rest_8", position="lower"),
            ]
        )
        first, second = chord.into_positions()

        self.assertEqual(first.symbols, [EncodedSymbol("rest_8", position="lower")])
        self.assertEqual(
            second.symbols,
            [
                EncodedSymbol("note_12", position="upper"),
                EncodedSymbol("note_12", position="upper"),
            ],
        )

    def test_rebalance_measure_voices_assigns_stable_voices_per_staff(self) -> None:
        measure = mxl.XMLMeasure()

        note1 = self._build_test_note(duration=4, staff=1, voice=1)
        measure.add_child(note1)
        measure.add_child(self._build_test_backup(duration=4))

        # Overlaps note1, so this must move to voice 2.
        note2 = self._build_test_note(duration=2, staff=1, voice=1)
        measure.add_child(note2)

        # Starts later but still overlaps with note1, reusing voice 2.
        note3 = self._build_test_note(duration=2, staff=1, voice=1)
        measure.add_child(note3)

        # Chord tone sharing note3's timing; should get the same voice as note3.
        note4 = self._build_test_note(duration=2, staff=1, voice=1, is_chord=True)
        measure.add_child(note4)

        measure.add_child(self._build_test_backup(duration=4))
        note5 = self._build_test_note(duration=4, staff=2, voice=1)
        measure.add_child(note5)
        measure.add_child(self._build_test_backup(duration=4))

        # Overlaps note5 on staff 2.
        note6 = self._build_test_note(duration=2, staff=2, voice=1)
        measure.add_child(note6)

        rebalance_measure_voices(measure)

        # Same-start events are processed by (start, end), so shorter durations
        # receive lower local voice numbers first.
        self.assertEqual(self._read_note_voice(note1), "2")
        self.assertEqual(self._read_note_voice(note2), "1")
        self.assertEqual(self._read_note_voice(note3), "1")
        self.assertEqual(self._read_note_voice(note4), "1")
        self.assertEqual(self._read_note_voice(note5), "6")
        self.assertEqual(self._read_note_voice(note6), "5")

    def test_next_group_starts_when_the_earliest_sounding_note_ends(self) -> None:
        """
        Tokens tell which notes start together, not when each group starts. Here the left
        hand's quarter note E3 starts under the right hand's quarter note C5, so the right
        hand's D5 starts when C5 ends (beat 1), while E3 is still sounding. Advancing by the
        shortest note of the group just written would put D5 at beat 1.5 instead.
        """
        hands_moving_independently = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_8 C3 _ _ _ lower
note_4 E3 _ _ _ lower
note_4 D5 _ _ _ upper
note_8 G3 _ _ _ lower
note_2 E5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(hands_moving_independently.splitlines())
        xml = _write_under_read_meter(tokens, (4, 4))
        self.assertEqual(
            _onsets(xml),
            sorted(
                [
                    ("1", "C5", 0.0),
                    ("1", "D5", 1.0),
                    ("1", "E5", 2.0),
                    ("2", "C3", 0.0),
                    ("2", "E3", 0.5),
                    ("2", "G3", 1.5),
                    ("2", "C3", 2.0),
                ]
            ),
        )

    def test_independent_hands_in_elijah_measure(self) -> None:
        """
        Measure 6 of the Elijah page from upstream issue #142: the right hand plays a
        quarter chord and dotted rhythms while the left hand holds A2 and moves in eighths
        and quarters. Every note must start on its beat, so that no staff runs past the end
        of the 4/4 measure. Upstream writes the held A2 as a second lower voice
        (``lower2``); this vocabulary has one lower position, so it shares ``lower``.
        """
        elijah_measure_6 = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_-4 . . . . .
timeSignature/4 . . . . .
note_4 A4 b _ _ upper&note_4 E4 b _ _ upper&note_4 C4 _ _ _ upper&note_2 A2 b _ _ lower&note_8 A2 b _ _ lower
note_4 A3 b _ _ lower
note_8. A4 b _ _ upper&note_8. E4 b _ _ upper&note_8. C4 _ _ _ upper
note_8 A3 b _ _ lower
note_16 C5 _ _ _ upper&note_16 A4 b _ _ upper&note_16 E4 b _ _ upper
note_8. C5 _ _ _ upper&note_8. A4 b _ _ upper&note_8. G4 b _ _ upper&note_8. E4 b _ _ upper&note_8 A2 b _ _ lower
note_4 C4 _ _ _ lower&note_4 A3 b _ _ lower
note_16 E5 b _ _ upper&note_16 A4 b _ _ upper&note_16 G4 b _ _ upper
note_8. E5 b _ _ upper&note_8. A4 b _ _ upper&note_8. G4 b _ _ upper
note_8 E4 b _ _ lower&note_8 C4 _ _ _ lower&note_8 A3 b _ _ lower
note_16 E5 b _ _ upper&note_16 A4 b _ _ upper&note_16 G4 b _ _ upper
barline . . . . ."""
        tokens = read_token_lines(elijah_measure_6.splitlines())
        xml = _write_under_read_meter(tokens, (4, 4))
        onsets = _onsets(xml)
        upper = sorted({onset for staff, _, onset in onsets if staff == "1"})
        lower = sorted({onset for staff, _, onset in onsets if staff == "2"})
        self.assertEqual(upper, [0.0, 1.0, 1.75, 2.0, 2.75, 3.0, 3.75])
        self.assertEqual(lower, [0.0, 0.5, 1.5, 2.0, 2.5, 3.5])

    def test_grace_notes_take_no_time(self) -> None:
        """A group of only grace notes must not move later notes (the next note stays on beat 1)."""
        with_grace_note = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_2 C3 _ _ _ lower
note_8G D5 _ _ _ upper
note_4 E5 _ _ _ upper
note_2 F5 _ _ _ upper&note_2 G2 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(with_grace_note.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        by_pitch = {name: onset for _, name, onset in _onsets(xml)}
        self.assertEqual(by_pitch["E5"], 1.0)
        self.assertEqual(by_pitch["F5"], 2.0)
        self.assertEqual(by_pitch["G2"], 2.0)

    def test_rest_under_a_held_note_starts_when_the_note_ends(self) -> None:
        """
        Super Mario Bros Ground Theme, measure 19, as the transformer reads it. The bass
        quarter A2 is still sounding when the treble's quarter chord starts on the second
        eighth, so the bass eighth rest starts when A2 ends (beat 1), not when the treble
        chord does (beat 1.5). The late start left a gap that renderers filled with an
        extra rest in each staff and overfilled the 2/2 measure.
        """
        ground_theme_measure_19 = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/2 . . . . .
note_8 C5 _ _ _ upper&note_8 A4 b _ _ upper&note_4 A2 b _ _ lower
note_4 C5 _ _ _ upper&note_4 A4 b _ _ upper
rest_8 _ _ _ _ lower
note_8 C5 _ _ _ upper&note_8 A4 b _ _ upper&note_8 E3 b _ _ lower
rest_8 _ _ _ _ upper&rest_4 _ _ _ _ lower
note_8 C5 _ _ _ upper&note_8 A4 b _ _ upper
note_4 D5 _ _ _ upper&note_4 B4 b _ _ upper&note_4 A3 b _ _ lower
barline . . . . ."""
        tokens = read_token_lines(ground_theme_measure_19.splitlines())
        xml = _write_under_read_meter(tokens, (2, 2))
        onsets = _onsets(xml)
        self.assertIn(("2", "rest", 1.0), onsets)
        self.assertIn(("2", "E3", 1.5), onsets)
        self.assertIn(("2", "A3", 3.0), onsets)
        self.assertIn(("1", "D5", 3.0), onsets)
        self.assertEqual(max(onset for _, _, onset in onsets), 3.0)

    def test_divisions_represent_notes_that_are_not_the_shortest_of_their_group(self) -> None:
        """
        The triplet quarter is never the shortest note of its group, yet its length and
        the backup after it must be whole numbers of divisions. Counting only each group's
        shortest note left the triplet's denominator out of the LCM, which truncated
        durations to zero.
        """
        groups = [
            SymbolChord(
                [
                    EncodedSymbol("note_6", "C5", position="upper"),
                    EncodedSymbol("note_8", "C3", position="lower"),
                ]
            ),
            SymbolChord([EncodedSymbol("note_8", "E3", position="lower")]),
            SymbolChord([EncodedSymbol("barline")]),
        ]
        division, _ = find_division_and_time_signature_nominator(groups)
        self.assertEqual(division % 3, 0)

    def test_time_signature_counts_held_notes_once(self) -> None:
        """
        The time signature token carries only the beat type; the beat count comes from
        the typical bar length. A bar in which the bass holds a quarter under the treble's
        quarter chord is one 2/2 bar, not 9/8 of one.
        """
        staggered_bar = [
            [
                EncodedSymbol("note_8", "C5", position="upper"),
                EncodedSymbol("note_4", "A2", position="lower"),
            ],
            [EncodedSymbol("note_4", "C5", position="upper")],
            [EncodedSymbol("rest_8", "_", position="lower")],
            [
                EncodedSymbol("note_8", "C5", position="upper"),
                EncodedSymbol("note_8", "E3", position="lower"),
            ],
            [
                EncodedSymbol("rest_8", "_", position="upper"),
                EncodedSymbol("rest_4", "_", position="lower"),
            ],
            [EncodedSymbol("note_8", "C5", position="upper")],
            [
                EncodedSymbol("note_4", "D5", position="upper"),
                EncodedSymbol("note_4", "A3", position="lower"),
            ],
            [EncodedSymbol("barline")],
        ]
        groups = [SymbolChord(symbols) for symbols in staggered_bar * 3]
        _, nominator = find_division_and_time_signature_nominator(groups)
        self.assertEqual(nominator, Fraction(1))

    def test_lone_rest_does_not_set_the_beat_count(self) -> None:
        """
        Super Mario Bros Coin: two treble eighths over a bass whose bar is a single rest,
        read as a half rest. That rest fills the 1/4 bar whatever glyph was read, so only
        the treble says how long the bar is.
        """
        coin = [
            SymbolChord(
                [
                    EncodedSymbol("note_8", "B4", position="upper"),
                    EncodedSymbol("rest_2", "_", position="lower"),
                ]
            ),
            SymbolChord([EncodedSymbol("note_8", "E5", position="upper")]),
            SymbolChord([EncodedSymbol("bolddoublebarline")]),
        ]
        _, nominator = find_division_and_time_signature_nominator(coin)
        self.assertEqual(nominator, Fraction(1, 4))

    def test_bars_of_only_rests_do_not_vote_on_the_beat_count(self) -> None:
        """
        A bar in which every staff holds a single rest says nothing about the bar length;
        it must neither count as a bar nor run into the next one.
        """
        full_bar = [
            SymbolChord(
                [
                    EncodedSymbol("note_2", "C5", position="upper"),
                    EncodedSymbol("note_2", "C3", position="lower"),
                ]
            ),
            SymbolChord(
                [
                    EncodedSymbol("note_2", "D5", position="upper"),
                    EncodedSymbol("note_2", "D3", position="lower"),
                ]
            ),
            SymbolChord([EncodedSymbol("barline")]),
        ]
        rest_bar = [
            SymbolChord(
                [
                    EncodedSymbol("rest_8", "_", position="upper"),
                    EncodedSymbol("rest_8", "_", position="lower"),
                ]
            ),
            SymbolChord([EncodedSymbol("barline")]),
        ]
        _, nominator = find_division_and_time_signature_nominator(rest_bar + rest_bar + full_bar)
        self.assertEqual(nominator, Fraction(1))

    def test_lone_rest_is_written_as_a_measure_rest(self) -> None:
        """
        Super Mario Bros Coin: the bass's only symbol in the 1/4 bar is a rest read as a
        half rest. Written as read it overfills the bar; as a measure rest it fills it.
        """
        coin = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_8 B4 _ _ _ upper&rest_2 _ _ _ _ lower
note_8 E5 _ _ _ upper
bolddoublebarline . . . . ."""
        root = self._generate_root(coin)
        divisions = int(root.findtext("part/measure/attributes/divisions", "0"))
        rests = [n for n in root.iter("note") if n.find("rest") is not None]
        self.assertEqual(len(rests), 1)
        self.assertEqual(rests[0].find("rest").get("measure"), "yes")  # type: ignore[union-attr]
        self.assertEqual(int(rests[0].findtext("duration", "0")), divisions)
        self.assertEqual(root.findtext("part/measure/attributes/time/beats"), "1")

    def test_lone_rest_after_the_bar_start_stays_as_read(self) -> None:
        """A measure rest starts its bar, so a staff's only rest read mid-bar keeps its value."""
        late_rest = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper
note_2 D5 _ _ _ upper&rest_2 _ _ _ _ lower
barline . . . . ."""
        root = self._generate_root(late_rest)
        rests = [n for n in root.iter("note") if n.find("rest") is not None]
        self.assertEqual(len(rests), 1)
        self.assertIsNone(rests[0].find("rest").get("measure"))  # type: ignore[union-attr]
        self.assertEqual(rests[0].findtext("type"), "half")

    def test_lone_rest_as_long_as_the_other_staff_stays_as_read(self) -> None:
        """
        Für Elise opens with a pickup: two treble sixteenths over a bass eighth rest. The
        staves agree on an eighth, so the rest keeps its value; filling the 3/8 bar would
        add two eighths of silence before the first full bar.
        """
        pickup = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/8 . . . . .
note_16 E5 _ _ _ upper&rest_8 _ _ _ _ lower
note_16 D5 _ _ _ upper
barline . . . . .
note_4. A4 _ _ _ upper&note_4. A2 _ _ _ lower
barline . . . . .
note_4. B4 _ _ _ upper&note_4. B2 _ _ _ lower
barline . . . . ."""
        root = self._generate_root(pickup)
        divisions = int(root.findtext("part/measure/attributes/divisions", "0"))
        self.assertEqual(root.findtext("part/measure/attributes/time/beats"), "3")
        rest = next(n for n in root.iter("note") if n.find("rest") is not None)
        self.assertIsNone(rest.find("rest").get("measure"))  # type: ignore[union-attr]
        self.assertEqual(rest.findtext("type"), "eighth")
        self.assertEqual(2 * int(rest.findtext("duration", "0")), divisions)

    def test_lone_rest_after_a_repeat_sign_as_long_as_the_other_staff_stays_as_read(
        self,
    ) -> None:
        """
        The Turkish March's sections each start after a repeat sign with a pickup: four
        treble sixteenths over a bass quarter rest, in 2/4.
        """
        repeated_pickup = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . .
note_2 D5 _ _ _ upper&note_2 D3 _ _ _ lower
repeatStart . . . . .
note_16 B4 _ _ _ upper&rest_4 _ _ _ _ lower
note_16 A4 _ _ _ upper
note_16 G4 _ _ _ upper
note_16 A4 _ _ _ upper
barline . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . ."""
        root = self._generate_root(repeated_pickup)
        divisions = int(root.findtext("part/measure/attributes/divisions", "0"))
        self.assertEqual(root.findtext("part/measure/attributes/time/beats"), "2")
        rest = next(n for n in root.iter("note") if n.find("rest") is not None)
        self.assertIsNone(rest.find("rest").get("measure"))  # type: ignore[union-attr]
        self.assertEqual(rest.findtext("type"), "quarter")
        self.assertEqual(int(rest.findtext("duration", "0")), divisions)

    def test_lone_rest_as_long_as_the_other_staff_keeps_its_value_only_in_a_short_bar(
        self,
    ) -> None:
        """
        Only a bar shorter than the meter is a pickup. Where the other staff fills the 2/4
        bar the rest is still the bar's measure rest, and where the staves agree on more
        than the bar it still fills the bar, as before.
        """
        for upper, lower in (("note_2", "rest_2"), ("note_2.", "rest_2.")):
            with self.subTest(read=lower):
                tokens = f"""clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . .
note_2 D5 _ _ _ upper&note_2 D3 _ _ _ lower
barline . . . . .
{upper} E5 _ _ _ upper&{lower} _ _ _ _ lower
barline . . . . ."""
                root = self._generate_root(tokens)
                divisions = int(root.findtext("part/measure/attributes/divisions", "0"))
                self.assertEqual(root.findtext("part/measure/attributes/time/beats"), "2")
                rest = next(n for n in root.iter("note") if n.find("rest") is not None)
                self.assertEqual(rest.find("rest").get("measure"), "yes")  # type: ignore[union-attr]
                self.assertEqual(int(rest.findtext("duration", "0")), 2 * divisions)

    def test_lone_rest_without_time_signature_fills_a_plausible_typical_bar(self) -> None:
        """
        Pages after the first carry no time signature, so the typical bar length stands in,
        but only when it is a whole number of eighths: misread bars can push it elsewhere.
        """
        three_four = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
note_2. C5 _ _ _ upper&note_2. C3 _ _ _ lower
barline . . . . .
note_2. D5 _ _ _ upper&rest_1 _ _ _ _ lower
barline . . . . ."""
        root = self._generate_root(three_four)
        divisions = int(root.findtext("part/measure/attributes/divisions", "0"))
        rest = next(n for n in root.iter("note") if n.find("rest") is not None)
        self.assertEqual(rest.find("rest").get("measure"), "yes")  # type: ignore[union-attr]
        self.assertEqual(int(rest.findtext("duration", "0")), 3 * divisions)

        seventeen_sixteenths = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
note_1 C5 _ _ _ upper&note_1 C3 _ _ _ lower
note_16 E5 _ _ _ upper
barline . . . . .
note_1 D5 _ _ _ upper&rest_1 _ _ _ _ lower
note_16 F5 _ _ _ upper
barline . . . . ."""
        root = self._generate_root(seventeen_sixteenths)
        rest = next(n for n in root.iter("note") if n.find("rest") is not None)
        self.assertIsNone(rest.find("rest").get("measure"))  # type: ignore[union-attr]
        self.assertEqual(rest.findtext("type"), "whole")

    def test_written_time_signature_sets_the_measure_rest_length(self) -> None:
        """
        Once a time signature is written its bar is what a measure rest fills, even when
        the typical bar read from the page (here 15/16, one sixteenth missing per bar) is
        no length a meter has.
        """
        missing_sixteenth = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2. C5 _ _ _ upper&note_2. C3 _ _ _ lower
note_8. D5 _ _ _ upper
barline . . . . .
note_2. E5 _ _ _ upper&rest_1 _ _ _ _ lower
note_8. F5 _ _ _ upper
barline . . . . ."""
        root = self._generate_root(missing_sixteenth)
        divisions = int(root.findtext("part/measure/attributes/divisions", "0"))
        self.assertEqual(root.findtext("part/measure/attributes/time/beats"), "3")
        rest = next(n for n in root.iter("note") if n.find("rest") is not None)
        self.assertEqual(rest.find("rest").get("measure"), "yes")  # type: ignore[union-attr]
        self.assertEqual(int(rest.findtext("duration", "0")), 3 * divisions)

    def test_divisions_represent_the_measure_rest_of_the_written_time_signature(self) -> None:
        """
        Bars of a half, a quarter and a triplet quarter need only 12 divisions per whole
        note, but in eighths they make a 7/8 bar, and a measure rest of 7/8 needs 24.
        """
        triplet_bars = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/8 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower
note_6 E5 _ _ _ upper&note_6 E3 _ _ _ lower
barline . . . . .
note_2 C5 _ _ _ upper&rest_2 _ _ _ _ lower
note_4 D5 _ _ _ upper
note_6 E5 _ _ _ upper
barline . . . . ."""
        root = self._generate_root(triplet_bars)
        divisions = int(root.findtext("part/measure/attributes/divisions", "0"))
        self.assertEqual(root.findtext("part/measure/attributes/time/beats"), "7")
        rest = next(n for n in root.iter("note") if n.find("rest") is not None)
        self.assertEqual(rest.find("rest").get("measure"), "yes")  # type: ignore[union-attr]
        self.assertEqual(2 * int(rest.findtext("duration", "0")), 7 * divisions)

    def test_each_time_signature_counts_the_beats_of_its_own_section(self) -> None:
        """
        Chrono Trigger's Lucca's Theme opens in 6/8 and changes to 2/4. One typical bar for
        the whole page gave both sections the same length; each section has its own.
        """
        six_eight_then_two_four = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_2 . . . . .
timeSignature/8 . . . . .
note_4. D5 _ _ _ upper&note_4. D3 _ _ _ lower
note_4. E5 _ _ _ upper&note_4. E3 _ _ _ lower
barline . . . . .
note_4. F5 _ _ _ upper&note_4. F3 _ _ _ lower
note_4. G5 _ _ _ upper&note_4. G3 _ _ _ lower
barline . . . . .
timeSignature/4 . . . . .
note_2 A5 _ _ _ upper&note_2 A3 _ _ _ lower
barline . . . . ."""
        root = self._generate_root(six_eight_then_two_four)
        times = [(t.findtext("beats"), t.findtext("beat-type")) for t in root.iter("time")]
        self.assertEqual(times, [("6", "8"), ("2", "4")])

    def test_time_signature_without_measured_bars_uses_the_typical_bar(self) -> None:
        """A section whose bars hold only rests says nothing about its own bar length."""
        resting_section = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_1 C5 _ _ _ upper&note_1 C3 _ _ _ lower
barline . . . . .
timeSignature/8 . . . . .
rest_1 _ _ _ _ upper&rest_1 _ _ _ _ lower
barline . . . . ."""
        root = self._generate_root(resting_section)
        times = [(t.findtext("beats"), t.findtext("beat-type")) for t in root.iter("time")]
        self.assertEqual(times, [("4", "4"), ("8", "8")])

    def test_symbols_chorded_with_notes_are_not_written_as_rests(self) -> None:
        """
        The transformer sometimes chords a barline with a note. Only notes and rests are
        written from a chord group; the barline used to come out as a rest.
        """
        barline_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower
note_2 D5 _ _ _ upper&barline . . . . .&note_2 D3 _ _ _ lower
barline . . . . ."""
        root = self._generate_root(barline_in_chord)
        self.assertEqual([n for n in root.iter("note") if n.find("rest") is not None], [])
        pitches = sorted(n.findtext("pitch/step", "") for n in root.iter("note"))
        self.assertEqual(pitches, ["C", "C", "D", "D"])

    def _generate_root(self, token_lines: str) -> ET.Element:
        tokens = read_token_lines(token_lines.splitlines())
        xml = generate_xml(XmlGeneratorArguments(), [tokens], "")
        return ET.fromstring(xml.to_string())  # noqa: S314 - generated by the test itself

    def test_triplets_over_eighths_start_on_their_beats(self) -> None:
        """
        A triplet quarter beside an eighth: the next eighth starts under the held triplet,
        and every duration and backup is a whole number of divisions.
        """
        triplet_over_eighths = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_6 C5 _ _ _ upper&note_8 C3 _ _ _ lower
note_8 E3 _ _ _ lower
note_6 D5 _ _ _ upper
note_4 G3 _ _ _ lower
note_6 E5 _ _ _ upper
note_2 F5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(triplet_over_eighths.splitlines())
        xml = _write_under_read_meter(tokens, (4, 4))
        root = ET.fromstring(xml.to_string())  # noqa: S314 - generated by the test itself
        divisions = int(root.findtext("part/measure/attributes/divisions", "0"))
        self.assertEqual(divisions % 3, 0)
        durations = [int(d.text or "0") for d in root.iter("duration")]
        self.assertTrue(all(d > 0 for d in durations))
        by_pitch = {name: onset for _, name, onset in _onsets(xml)}
        self.assertAlmostEqual(by_pitch["E3"], 0.5)
        self.assertAlmostEqual(by_pitch["D5"], 2 / 3)
        self.assertAlmostEqual(by_pitch["F5"], 2.0)

    def _norm_expected(self, expected: str) -> str:
        norm = expected.replace("\n", "")
        norm = re.sub(r",\s+", ",", norm)
        norm = re.sub(r"\[\s+", "[", norm)
        return norm

    def _xml_to_str(self, xml: Any) -> str:
        def recurse(node_or_list: Any) -> str:
            if isinstance(node_or_list, list):
                return (
                    "["
                    + ",".join(recurse(child) for child in node_or_list if child is not None)
                    + "]"
                )

            node = node_or_list
            name = node.__class__.__name__

            ignore_nodes = (
                "XMLAlter",
                "XMLOctave",
                "XMLType",
                "XMLPartList",
                "XMLDefaults",
                "XMLIdentification",
                "XMLEncoding",
                "XMLSoftware",
                "XMLStaves",
                "XMLPartSymbol",
            )

            if name in ignore_nodes:
                return ""
            value = getattr(node, "value_", None)

            if hasattr(node, "children"):
                children = node.children
            elif hasattr(node, "get_children"):
                children = node.get_children()
            else:
                children = []

            child_strs = [recurse(child) for child in children if child is not None]
            child_strs = [child for child in child_strs if child != ""]

            parts = []
            if value is not None and value != "":
                parts.append(f"value: {value}")
            if child_strs:
                parts.append(f"[{','.join(child_strs)}]")

            if parts:
                return f"{name}({','.join(parts)})"
            else:
                return f"{name}()"

        return recurse(xml)

    def _build_test_note(
        self, duration: int, staff: int, voice: int, is_chord: bool = False
    ) -> mxl.XMLNote:
        note = mxl.XMLNote()
        if is_chord:
            note.add_child(mxl.XMLChord())
        note.add_child(mxl.XMLDuration(value_=duration))
        note.add_child(mxl.XMLStaff(value_=staff))
        note.add_child(mxl.XMLVoice(value_=str(voice)))
        return note

    def _build_test_backup(self, duration: int) -> mxl.XMLBackup:
        backup = mxl.XMLBackup()
        backup.add_child(mxl.XMLDuration(value_=duration))
        return backup

    def _read_note_voice(self, note: mxl.XMLNote) -> str:
        voices = note.get_children_of_type(mxl.XMLVoice)
        self.assertGreater(len(voices), 0)
        return str(voices[0].value_)

    def test_clef_change_joined_to_a_chord_of_notes(self) -> None:
        """
        The transformer sometimes joins a clef change in one staff to a chord of notes in the
        other (a real clef change at that beat). The clef must be written, before the notes
        that follow it, and must not be treated as a rest (that used to fail an assertion
        and lose the whole page).
        """
        clef_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_4 A4 _ _ _ upper&clef_G2 _ _ _ _ lower
note_4 F4 _ _ _ lower
note_2 G4 _ _ _ upper&note_2 E4 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(clef_in_chord.splitlines())
        xml = _written(tokens)
        measure = _first_measure(xml)
        pitches = [name for _, name, _ in _onsets(xml)]
        self.assertEqual(sorted(pitches), ["A4", "C3", "C5", "E4", "F4", "G4"])
        order = [
            ("clef", c.get("number"), c.findtext("sign"))
            for element in measure
            for c in ([element] if element.tag == "note" else element.iter("clef"))
        ]
        lower_clefs = [entry for entry in order if entry[0] == "clef" and entry[1] == "2"]
        self.assertEqual([sign for _, _, sign in lower_clefs], ["F", "G"])
        notes_and_clefs = [
            "clef" if element.tag != "note" else _pitch(element) + _staff(element)
            for element in measure
            if element.tag == "note" or element.find("clef") is not None
        ]
        self.assertLess(notes_and_clefs.index("clef", 1), notes_and_clefs.index("F2"))

    def test_barline_joined_to_a_chord_of_notes(self) -> None:
        """A barline the transformer joined to the last chord of a measure still ends the measure."""
        barline_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower&barline _ _ _ _ .
note_2 E5 _ _ _ upper&note_2 E3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(barline_in_chord.splitlines())
        xml = _written(tokens)
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual(sorted(_pitch(n) for n in _notes(measures[0])), ["C", "C", "D", "D"])
        self.assertEqual(sorted(_pitch(n) for n in _notes(measures[1])), ["E", "E"])

    def test_clef_with_a_pitch_is_not_written_as_a_note(self) -> None:
        """A clef token that came with a pitch, joined to a note, must not become an extra note."""
        clef_with_pitch = """clef_G2 _ _ _ _ upper&clef_G2 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 F4 _ _ _ upper&clef_F4 F5 _ _ _ lower
note_2 G4 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(clef_with_pitch.splitlines())
        xml = _written(tokens)
        pitches = sorted(name for _, name, _ in _onsets(xml))
        self.assertEqual(pitches, ["C3", "F4", "G4"])

    def test_barline_joined_to_a_chord_and_repeated_after_it_counts_once(self) -> None:
        """A barline joined to a chord and then written again on its own ends only one measure."""
        barline_twice = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower&barline F5 _ _ _ upper
barline . . . . .
note_2 E5 _ _ _ upper&note_2 E3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(barline_twice.splitlines())
        xml = _written(tokens)
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual(sorted(_pitch(n) for n in _notes(measures[0])), ["C", "C", "D", "D"])

    def test_mixed_chord_is_split_into_before_notes_after(self) -> None:
        """
        Clefs, keys and time signatures go before the notes of the chord, in the order
        they have at the start of a line, barlines after.
        """
        chord = [
            EncodedSymbol("barline"),
            EncodedSymbol("timeSignature/4"),
            EncodedSymbol("note_4", "C4", position="upper"),
            EncodedSymbol("keySignature_1"),
            EncodedSymbol("clef_G2", position="lower"),
            EncodedSymbol("rest_4", position="lower"),
        ]
        parts = _split_mixed_chord(chord)
        rhythms = [[s.rhythm for s in part] for part in parts]
        self.assertEqual(
            rhythms,
            [
                ["clef_G2"],
                ["keySignature_1"],
                ["timeSignature/4"],
                ["note_4", "rest_4"],
                ["barline"],
            ],
        )

    def test_chords_without_notes_or_only_notes_are_not_split(self) -> None:
        clefs = [
            EncodedSymbol("clef_G2", position="upper"),
            EncodedSymbol("clef_F4", position="lower"),
        ]
        notes = [
            EncodedSymbol("note_4", "C4", position="upper"),
            EncodedSymbol("rest_4", position="lower"),
        ]
        self.assertEqual(_split_mixed_chord(clefs), [clefs])
        self.assertEqual(_split_mixed_chord(notes), [notes])

    def test_key_signature_joined_to_a_chord_of_notes(self) -> None:
        key_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower
barline . . . . .
note_2 D5 _ _ _ upper&note_2 D3 _ _ _ lower&keySignature_2 . . . . .
barline . . . . ."""
        tokens = read_token_lines(key_in_chord.splitlines())
        xml = _written(tokens)
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual(measures[1].findtext("attributes/key/fifths"), "2")
        self.assertEqual(sorted(_pitch(n) for n in _notes(measures[1])), ["D", "D"])

    def test_time_signature_joined_to_a_chord_of_notes(self) -> None:
        time_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower&timeSignature/4 . . . . .
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(time_in_chord.splitlines())
        xml = _written(tokens)
        self.assertIsNotNone(_first_measure(xml).find("attributes/time"))
        self.assertEqual(sorted(name for _, name, _ in _onsets(xml)), ["C3", "C5", "D3", "D5"])

    def test_clef_joined_to_a_chord_of_rests(self) -> None:
        clef_with_rests = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
rest_4 _ _ _ _ upper&rest_4 _ _ _ _ lower&clef_G2 _ _ _ _ lower
note_4 C5 _ _ _ upper&note_4 E4 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(clef_with_rests.splitlines())
        xml = _written(tokens)
        lower_clefs = [
            c.findtext("sign") for c in _first_measure(xml).iter("clef") if c.get("number") == "2"
        ]
        self.assertEqual(lower_clefs, ["F", "G"])
        self.assertEqual(sorted(name for _, name, _ in _onsets(xml, rests=False)), ["C5", "E4"])

    def test_repeat_joined_to_a_chord_of_notes_is_kept(self) -> None:
        """
        A repeat sorts before the notes of its chord, so the chord used to be written as a
        repeat only and its notes were dropped without an error.
        """
        repeat_in_chord = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower&repeatEnd . . . . .
note_2 D5 _ _ _ upper&note_2 D3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(repeat_in_chord.splitlines())
        xml = _written(tokens)
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual(sorted(_pitch(n) for n in _notes(measures[0])), ["C", "C"])
        repeat = measures[0].find("barline/repeat")
        self.assertIsNotNone(repeat)
        self.assertEqual(repeat.get("direction"), "backward")  # type: ignore[union-attr]

    def test_repeat_split_off_a_chord_wins_over_a_following_plain_barline(self) -> None:
        repeat_then_barline = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower&repeatEnd . . . . .
barline . . . . .
note_2 D5 _ _ _ upper&note_2 D3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(repeat_then_barline.splitlines())
        xml = _written(tokens)
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        repeat = measures[0].find("barline/repeat")
        self.assertIsNotNone(repeat)
        self.assertEqual(repeat.get("direction"), "backward")  # type: ignore[union-attr]

    def test_clef_joined_to_the_middle_of_a_triplet(self) -> None:
        """Splitting the chord must not break the triplet around it."""
        clef_in_triplet = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_12 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_12 D5 _ _ _ upper&clef_G2 _ _ _ _ lower
note_12 E5 _ _ _ upper
note_4 F5 _ _ _ upper&note_4 F4 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(clef_in_triplet.splitlines())
        xml = _written(tokens)
        triplet = [
            n for n in _notes(_first_measure(xml)) if n.find("time-modification") is not None
        ]
        self.assertEqual([_pitch(n) for n in triplet], ["C", "D", "E"])
        marks = [[t.get("type") for t in n.iter("tuplet")] for n in triplet]
        self.assertEqual(marks, [["start"], [], ["stop"]])
        by_pitch = {name: onset for _, name, onset in _onsets(xml)}
        self.assertEqual(by_pitch["F5"], 1.0)

    def test_real_page_clef_change_in_a_chord(self) -> None:
        """First measure of a system of Clair de lune (Mutopia edition, page 4) as read by the transformer."""
        clair_de_lune = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_-5 . . . . .
rest_8 _ _ _ _ upper&note_8 D2 b _ _ lower&note_8 A2 b _ _ lower
note_4 F3 _ _ _ upper2&note_4 A3 b _ _ upper&clef_G2 _ _ _ _ lower
note_8 F4 _ _ _ lower&note_8 A4 b _ slurStart lower
note_4. F5 _ _ _ upper&note_4. A5 b _ _ upper&note_2. F4 _ _ _ lower&note_2. C4 b _ _ lower2&note_2. A4 b _ slurStop lower
note_4. F5 _ _ slurStart_slurStop upper&note_4. D5 b _ _ upper
barline . . . . ."""
        tokens = read_token_lines(clair_de_lune.splitlines())
        xml = _written(tokens)
        lower_clefs = [
            c.findtext("sign") for c in _first_measure(xml).iter("clef") if c.get("number") == "2"
        ]
        self.assertEqual(lower_clefs, ["F", "G"])
        # The fork's writer knows no upper2 and lower2 positions, which upstream's
        # vocabulary has, and writes those voices in its own order and staff. So the
        # notes are compared by pitch and onset only.
        onsets = sorted((name, onset) for _, name, onset in _onsets(xml, rests=False))
        self.assertEqual(
            onsets,
            sorted(
                [
                    ("D2", 0.0),
                    ("A2", 0.0),
                    ("F3", 0.5),
                    ("A3", 0.5),
                    ("F4", 1.5),
                    ("A4", 1.5),
                    ("F5", 2.0),
                    ("A5", 2.0),
                    ("F4", 2.0),
                    ("C4", 2.0),
                    ("A4", 2.0),
                    ("F5", 3.5),
                    ("D5", 3.5),
                ]
            ),
        )
        self.assertEqual([_pitch(n) for n in _notes(_first_measure(xml))][0], "rest")

    def test_real_page_barline_in_a_chord(self) -> None:
        """First two measures of polish-scores train_050 as read by the transformer."""
        train_050 = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_-2 . . . . .
note_8 F5 _ _ _ upper&note_8 F4 _ _ _ upper&note_8 F2 _ _ _ lower&note_8 D5 _ _ _ upper&note_8 B4 b _ _ upper
rest_8 _ _ _ _ upper&rest_8 _ _ _ _ lower
clef_G2 _ _ _ _ lower&clef_F4 _ _ _ _ upper
rest_8 _ _ _ _ upper&note_2 F4 _ _ _ lower&note_2 D4 _ _ _ lower&note_2 B4 b accent _ lower
note_12 C3 # _ slurStart upper
note_12 E3 b _ _ upper
note_12 D3 _ _ _ upper
note_12 F3 _ _ _ upper
note_12 B3 b _ _ upper
note_8 F2 _ _ _ lower&barline _ _ _ _ .
rest_8 _ _ _ _ upper&rest_8 _ _ _ _ lower
rest_8 _ _ _ _ upper&note_2 E4 _ _ _ lower&note_2 C4 # _ _ lower&note_2 A4 _ _ _ lower
note_8 G3 # _ slurStart upper
note_8 B3 b _ _ upper
note_8 A3 _ _ _ upper
note_8 E4 _ _ _ upper
note_8 G4 _ _ _ upper
barline . . . . ."""
        tokens = read_token_lines(train_050.splitlines())
        xml = _written(tokens)
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual([_pitch(n) for n in _notes(measures[0])][-1], "F")
        self.assertIn("G", [_pitch(n) for n in _notes(measures[1])])

    def test_barline_split_off_a_chord_and_a_repeat_joined_to_the_next_chord(self) -> None:
        """Only a plain barline after a split-off barline is a duplicate, not a chord with a repeat."""
        both = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignature/4 . . . . .
note_2 C5 _ _ _ upper&note_2 C3 _ _ _ lower&barline _ _ _ _ .
note_2 D5 _ _ _ upper&note_2 D3 _ _ _ lower&repeatEnd . . . . .
note_2 E5 _ _ _ upper&note_2 E3 _ _ _ lower
barline . . . . ."""
        tokens = read_token_lines(both.splitlines())
        xml = _written(tokens)
        measures = xml.findall("part/measure")
        self.assertEqual(len(measures), 3)
        self.assertEqual(
            [sorted(_pitch(n) for n in _notes(m)) for m in measures],
            [["C", "C"], ["D", "D"], ["E", "E"]],
        )

    def test_symbols_split_off_a_chord_are_written_as_if_they_came_separately(self) -> None:
        """A key and time signature joined to a chord give the same MusicXML as on their own lines."""
        joined = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower&keySignature_2 . . . . .&timeSignature/4 . . . . .
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower
barline . . . . ."""
        separate = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_2 . . . . .
timeSignature/4 . . . . .
note_4 C5 _ _ _ upper&note_4 C3 _ _ _ lower
note_4 D5 _ _ _ upper&note_4 D3 _ _ _ lower
barline . . . . ."""
        written = [
            ET.tostring(
                _written(read_token_lines(t.splitlines())),
                encoding="unicode",
            )
            for t in (joined, separate)
        ]
        self.assertEqual(written[0], written[1])
