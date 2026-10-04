# ruff: noqa: E501, S101

import re
import unittest
import xml.etree.ElementTree as ET
from typing import Any

import musicxml.xmlelement.xmlelement as mxl

from homr.music_xml_generator import (
    SymbolChord,
    XmlGeneratorArguments,
    _split_mixed_chord,
    generate_xml,
    rebalance_measure_voices,
)
from homr.transformer.vocabulary import EncodedSymbol
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


def _onsets(xml: ET.Element) -> list[tuple[str, str, float]]:
    """(staff, pitch with octave, onset in quarter notes) for every note of the first measure."""
    part = xml.find("part")
    assert part is not None
    divisions = int(part.findtext("measure/attributes/divisions", "1"))
    time, onset, result = 0, 0, []
    for element in _first_measure(xml):
        if element.tag == "backup":
            time -= int(element.findtext("duration", "0"))
        elif element.tag == "forward":
            time += int(element.findtext("duration", "0"))
        elif element.tag == "note":
            if element.find("chord") is None:
                onset = time
                time += _duration(element)
            pitch = element.find("pitch")
            if pitch is not None:
                name = pitch.findtext("step", "") + pitch.findtext("octave", "")
                result.append((_staff(element), name, onset / divisions))
    return sorted(result)


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
        self.assertEqual(sorted(name for _, name, _ in _onsets(xml)), ["C5", "E4"])

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
        onsets = sorted((name, onset) for _, name, onset in _onsets(xml))
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
