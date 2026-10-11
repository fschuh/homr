"""Tuplets the transformer missed, repaired from the page's beams and bracketed numerals.

The transformer often writes a tuplet's notes with their plain value and no ratio: six
16ths under one beam in a beat come out as six plain 16ths, and the bar runs long. Most
such groups print no numeral. Two kinds of evidence on the page say a group is a tuplet:

- a beam joins its notes, and their plain values add up to three, six or another
  non-power-of-two count of their smallest value. The group is then a triplet (3:2):
  a beam alone implies no other ratio, so quintuplets and septuplets need their numeral;
- a bracketed numeral ("3", "5", "6", "7", read by ``tuplet_numerals``) spans its notes,
  whose values, as printed where the note-value reader is sure of them, add up to that
  many of their smallest value.

A beam group is no candidate when:

- a lone note an eighth or shorter, in no beam group, stands next to it: a flag, or a
  beam the detector lost, which would split the group;
- its stems are unevenly spaced (the widest gap more than EVEN_SPACING times the
  narrowest, or MIXED_SPACING when its values differ): a much wider gap holds a note the
  staff lost, such as one written on the other staff;
- a notehead of one of its chords lies more than HEAD_ON_STEM off the chord's stem: a
  separate note the transformer merged into the chord;
- a printed value the note-value reader is sure of disagrees with its token: a missed dot
  (a 16th and a dotted eighth read as a 16th and an eighth) looks like three 16ths;
- in a compound meter it is a whole number of dotted beats, as three eighths in 6/8 are,
  or it fills the whole bar.

A candidate is applied only where its bar proves it. The bar's length must come from a
trusted meter: one read on the page or carried from the previous page (see
``time_signatures`` and ``timing_repairs``), never one inferred from the bars, which
missed tuplets make too long. A bar's candidates then apply together, and only when every
staff they touch falls short or overflows as recognized, placed either way (see
``homr.bar_placement``), and adds up exactly with them as the writer places the bar, no
staff ends further from the bar length, and at least MIN_SUPPORTING_BARS bars of the page
under the same length add up as the writer places them, with their own candidates applied.
Every candidate and what became of it is recorded in the sidecar.
"""

import copy
import math
import re
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from homr import bar_placement as placement
from homr.transformer.vocabulary import EncodedSymbol

if TYPE_CHECKING:
    from homr.music_xml_generator import SymbolChord
    from homr.visual_sidecar.note_values import NoteValueReading, StemEnd
    from homr.visual_sidecar.tuplet_numerals import Numeral

TUPLET_REPAIRS_VERSION = 1

#: Bars under a length that must add up as written before it confirms a repair.
MIN_SUPPORTING_BARS = 3
#: The widest gap between a beam group's stems may be this many times the narrowest when
#: its notes share one value, and this many when they do not.
EVEN_SPACING = 1.6
MIXED_SPACING = 2.2
#: A chord's noteheads lie within this many staff spaces of its stem.
HEAD_ON_STEM = 1.0
#: A note this short or shorter, in no beam group, next to a group keeps it unrepaired.
LONE_SHORT = Fraction(1, 8)
#: The notes under a numeral's bracket: noteheads between its hooks, widened by this many
#: staff spaces, and within this many staff spaces of the numeral vertically.
BRACKET_MARGIN = 0.6
NUMERAL_REACH = 10.0
TRIPLET = Fraction(3, 2)

# What became of a group.
CANDIDATE = "candidate"
LONE_SHORT_NEIGHBOUR = "lone_short_neighbour"
UNEVEN_SPACING = "uneven_spacing"
HEAD_OFF_STEM = "head_off_stem"
VALUE_DISAGREES = "value_disagrees"
REST_OR_SINGLE = "rest_or_single"
NOT_PLAIN = "not_plain"
UNITS_DIFFER = "units_differ"

# Why a bar's candidates were not applied.
NO_TRUSTED_METER = "no_trusted_meter"
ADDS_UP_AS_RECOGNIZED = "adds_up_as_recognized"
NOT_EXACT = "not_exact_with_repairs"
A_STAFF_GETS_WORSE = "a_staff_gets_worse"
TOO_FEW_SUPPORTING_BARS = "too_few_supporting_bars"

_NOTE = re.compile(r"note_(\d+)(\.*)")
_KERN_BY_VALUE = {"whole": 1, "half": 2, "quarter": 4, "eighth": 8, "16th": 16, "32nd": 32}

Chord = list[EncodedSymbol]
Meter = tuple[int, int]


class TupletSource(Protocol):
    """What the repair reads from the page, through the visual sidecar."""

    def stem_ends(self, chords: list[Chord]) -> list["StemEnd | None"]: ...

    def beamed_together(self, first: "StemEnd", second: "StemEnd") -> bool: ...

    def notehead_prediction_point(self, symbol: EncodedSymbol) -> tuple[float, float] | None: ...

    def notehead_point(self, symbol: EncodedSymbol) -> tuple[float, float] | None: ...

    def read_token_values(self, symbols: list[EncodedSymbol]) -> "dict[int, NoteValueReading]": ...

    def tuplet_numerals(self) -> "list[Numeral]": ...

    def read_time_signature(self, symbol: EncodedSymbol) -> Meter | None: ...

    def time_signature_status(self, symbol: EncodedSymbol) -> str | None: ...

    def carried_meter(self) -> Meter | None: ...

    def record_tuplet_repairs(
        self,
        groups: list[dict[str, Any]],
        notes: list[tuple[EncodedSymbol, dict[str, Any]]],
    ) -> None: ...


def repair_tuplets(groups: list["SymbolChord"], source: TupletSource) -> list["SymbolChord"]:
    """The voice's chord groups with the tuplets the page proves, and their tuplet marks."""
    from homr import (  # noqa: PLC0415 - it imports this module
        music_xml_generator as writer,
    )

    candidates: dict[int, str] = {}
    kind_of: dict[int, str] = {}
    numeral_repairs, found = numeral_candidates(groups, source)
    for key, new in numeral_repairs.items():
        candidates[key] = new
        kind_of[key] = "numeral"
    meters = trusted_meters(groups, source)
    beam_repairs, beam_found = beam_candidates(groups, source, meters)
    found.extend(beam_found)
    for key, new in beam_repairs.items():
        if key not in candidates:
            candidates[key] = new
            kind_of[key] = "beam"

    lengths = [Fraction(*meter) if meter is not None else None for meter in meters]
    lone_rests = writer._lone_rest_ids(groups)
    bars = _bars(groups)
    support = _support(bars, lengths, candidates, lone_rests)

    accepted: dict[int, str] = {}
    notes: list[tuple[EncodedSymbol, dict[str, Any]]] = []
    for number, bar in enumerate(bars, start=1):
        here = {id(s): candidates[id(s)] for g in bar for s in g.symbols if id(s) in candidates}
        if not here:
            continue
        length = lengths[number - 1] if number - 1 < len(lengths) else None
        reason = _refusal(bar, here, length, support, lone_rests)
        if not reason:
            accepted.update(here)
        for g in bar:
            for s in g.symbols:
                if id(s) in here:
                    notes.append(
                        (
                            s,
                            {
                                "bar": number,
                                "source": kind_of[id(s)],
                                "recognized": s.rhythm,
                                "repaired": here[id(s)],
                                "applied": not reason,
                                "reason": reason,
                                "bar_length": None if length is None else str(length),
                            },
                        )
                    )
    source.record_tuplet_repairs(found, notes)
    if not accepted:
        return groups
    repaired = [
        writer.SymbolChord([_with(s, accepted) for s in g.symbols], g.tuplet_mark) for g in groups
    ]
    return writer.add_tuplet_start_stop(repaired)


def trusted_meters(groups: list["SymbolChord"], source: TupletSource) -> list[Meter | None]:
    """Each bar's meter when it is trusted: read on the page or carried from the previous
    page, until a time-signature token without a trusted read starts a section. A token
    with nothing printed near it keeps the trusted meter in force, as the writer does."""
    from homr.visual_sidecar.time_signatures import NO_READ  # noqa: PLC0415 - cycle

    meter = source.carried_meter()
    out: list[Meter | None] = []
    for bar in _bars(groups):
        for group in bar:
            for symbol in group.symbols:
                if not symbol.rhythm.startswith("timeSignature/"):
                    continue
                read = source.read_time_signature(symbol)
                if read is not None:
                    meter = read
                elif meter is None or source.time_signature_status(symbol) != NO_READ:
                    meter = None
        out.append(meter)
    return out


def beam_candidates(
    groups: list["SymbolChord"], source: TupletSource, meters: list[Meter | None]
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """The triplet each beam group of plain notes would be, by id of each token, and a
    record of every group a rule declined or that became a candidate."""
    out: dict[int, str] = {}
    found: list[dict[str, Any]] = []
    for number, bar in enumerate(_bars(groups), start=1):
        meter = meters[number - 1] if number - 1 < len(meters) else None
        for staff in ("upper", "lower"):
            chords = _staff_chords(bar, staff)
            readable = [c for c in chords if c is not None]
            ends = dict(zip([id(c) for c in readable], source.stem_ends(readable), strict=True))
            beamed = beam_groups(chords, ends, source)
            grouped = {id(chord) for members in beamed for chord in members}
            for members in beamed:
                record = {"source": "beam", "bar": number, "staff": staff, "notes": len(members)}
                first = next(i for i, c in enumerate(chords) if c is members[0])
                last = next(i for i, c in enumerate(chords) if c is members[-1])
                if _lone_short(chords, grouped, first - 1) or _lone_short(
                    chords, grouped, last + 1
                ):
                    found.append({**record, "outcome": LONE_SHORT_NEIGHBOUR})
                    continue
                tokens = [s for chord in members for s in chord]
                kerns = [_plain_kern(s) for s in tokens]
                if any(kern is None for kern in kerns):
                    continue  # a tuplet already, or not a plain note
                ruled_out = _ruled_out(members, ends, tokens, source)
                if ruled_out:
                    found.append({**record, "outcome": ruled_out})
                    continue
                lengths = [min(s.get_duration().fraction for s in chord) for chord in members]
                units = _units(lengths)
                if units is None or not _is_tuplet_count(units):
                    continue
                if meter is not None and _meter_explains(sum(lengths, Fraction(0)), meter):
                    continue
                normal = 2 ** (units.bit_length() - 1)
                if Fraction(units, normal) != TRIPLET:
                    continue  # a beam alone implies only a triplet
                for s in tokens:
                    new = _scaled(s.rhythm, units, normal)
                    if new is not None:
                        out[id(s)] = new
                found.append({**record, "outcome": CANDIDATE, "ratio": f"{units}:{normal}"})
    return out, found


def beam_groups(
    chords: list[Chord | None], ends: "dict[int, StemEnd | None]", source: TupletSource
) -> list[list[Chord]]:
    """Runs of two or more consecutive chords whose stems a beam joins. A rest, or a chord
    whose stem is not found, ends a run."""
    out: list[list[Chord]] = []
    current: list[Chord] = []
    previous_end = None
    for chord in chords:
        end = ends.get(id(chord)) if chord is not None else None
        joined = (
            chord is not None
            and end is not None
            and bool(current)
            and previous_end is not None
            and source.beamed_together(previous_end, end)
        )
        if joined:
            current.append(chord)  # type: ignore[arg-type]
        else:
            if len(current) >= 2:
                out.append(current)
            current = [chord] if chord is not None and end is not None else []
        previous_end = end
    if len(current) >= 2:
        out.append(current)
    return out


def _lone_short(chords: list[Chord | None], grouped: set[int], index: int) -> bool:
    """Whether the chord at index is a note an eighth or shorter in no beam group."""
    if not 0 <= index < len(chords):
        return False
    chord = chords[index]
    if chord is None or id(chord) in grouped:
        return False
    return min(s.get_duration().fraction for s in chord) <= LONE_SHORT


def _ruled_out(
    members: list[Chord],
    ends: "dict[int, StemEnd | None]",
    tokens: list[EncodedSymbol],
    source: TupletSource,
) -> str:
    """Why the page says a beam group is not a missed tuplet, or "" when nothing does."""
    stems = [end.x for chord in members if (end := ends.get(id(chord))) is not None]
    gaps = [b - a for a, b in zip(stems, stems[1:], strict=False)]
    values = {min(s.get_duration().fraction for s in chord) for chord in members}
    limit = EVEN_SPACING if len(values) == 1 else MIXED_SPACING
    if gaps and min(gaps) > 0 and max(gaps) > limit * min(gaps):
        return UNEVEN_SPACING
    for chord in members:
        end = ends.get(id(chord))
        if end is None:
            continue
        for symbol in chord:
            point = source.notehead_prediction_point(symbol)
            if point is not None and abs(point[0] - end.x) > HEAD_ON_STEM * end.space:
                return HEAD_OFF_STEM
    readings = source.read_token_values(tokens)
    if any(reading.status == "disagrees" for reading in readings.values()):
        return VALUE_DISAGREES
    return ""


def _meter_explains(length: Fraction, meter: Meter) -> bool:
    """Whether a beam group this long is plain under the meter: a whole number of dotted
    beats in a compound meter, or the whole bar."""
    beats, beat_type = meter
    compound = beats % 3 == 0 and beats > 3
    return (compound and length % Fraction(3, beat_type) == 0) or length == Fraction(
        beats, beat_type
    )


def numeral_candidates(
    groups: list["SymbolChord"], source: TupletSource
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """The notes under each bracketed numeral take its ratio, by id of each token, with a
    record of every numeral placed in a bar.

    The notes are one staff's consecutive chords in one bar whose noteheads lie within the
    bracket; of the two staves, the one nearer the numeral. They must be plain notes with
    no rest between them, and their length in their smallest unit must be the numeral, as
    a quarter and an eighth under "3".
    """
    numerals = source.tuplet_numerals()
    out: dict[int, str] = {}
    found: list[dict[str, Any]] = []
    if not numerals:
        return out, found
    for number, bar in enumerate(_bars(groups), start=1):
        sequences = {
            staff: [(chord, _chord_point(chord, source)) for chord in _staff_chords(bar, staff)]
            for staff in ("upper", "lower")
        }
        for numeral in numerals:
            under = _under_bracket(numeral, sequences)
            if under is None:
                continue
            staff, span = under
            members = [sequences[staff][i][0] for i in span]
            record = {
                "source": "numeral",
                "bar": number,
                "staff": staff,
                "number": numeral.number,
                "notes": len(members),
            }
            chords = [chord for chord in members if chord is not None]
            if len(chords) < len(members) or len(chords) < 2:
                found.append({**record, "outcome": REST_OR_SINGLE})
                continue
            tokens = [s for chord in chords for s in chord]
            if any(_plain_kern(s) is None for s in tokens):
                found.append({**record, "outcome": NOT_PLAIN})
                continue
            values = _printed_values(tokens, source)
            lengths = [min(_length(*values[id(s)]) for s in chord) for chord in chords]
            if _units(lengths) != numeral.number:
                found.append({**record, "outcome": UNITS_DIFFER})
                continue
            actual = numeral.number
            normal = 2 ** (actual.bit_length() - 1)
            for s in tokens:
                kern, dotted = values[id(s)]
                new = _scaled(f"note_{kern}{'.' if dotted else ''}", actual, normal)
                if new is not None:
                    out[id(s)] = new
            found.append({**record, "outcome": CANDIDATE, "ratio": f"{actual}:{normal}"})
    return out, found


def _under_bracket(
    numeral: "Numeral",
    sequences: dict[str, list[tuple[Chord | None, tuple[float, float] | None]]],
) -> tuple[str, list[int]] | None:
    """The staff nearer the numeral that has noteheads under its bracket, and the span of
    its chords from the first such to the last."""
    sp = numeral.space
    x0, x1 = numeral.bracket[0] - BRACKET_MARGIN * sp, numeral.bracket[1] + BRACKET_MARGIN * sp
    best = None
    for staff, chords in sequences.items():
        inside = [
            i
            for i, (_, p) in enumerate(chords)
            if p is not None and x0 <= p[0] <= x1 and abs(p[1] - numeral.y) <= NUMERAL_REACH * sp
        ]
        if not inside:
            continue
        distance = float(np.mean([abs(chords[i][1][1] - numeral.y) for i in inside]))  # type: ignore[index]
        if best is None or distance < best[0]:
            best = (distance, staff, inside)
    if best is None:
        return None
    _, staff, inside = best
    return staff, list(range(inside[0], inside[-1] + 1))


def _chord_point(chord: Chord | None, source: TupletSource) -> tuple[float, float] | None:
    """The mean source position of a chord's linked noteheads."""
    if chord is None:
        return None
    points = [p for p in (source.notehead_point(s) for s in chord) if p is not None]
    if not points:
        return None
    return float(np.mean([p[0] for p in points])), float(np.mean([p[1] for p in points]))


def _printed_values(
    tokens: list[EncodedSymbol], source: TupletSource
) -> dict[int, tuple[int, bool]]:
    """Each token's plain kern and dot: as printed where the reader is sure, else as read."""
    readings = source.read_token_values(tokens)
    values: dict[int, tuple[int, bool]] = {}
    for s in tokens:
        match = _NOTE.fullmatch(s.rhythm)
        assert match is not None  # noqa: S101 - only plain notes get here
        kern, dotted = int(match[1]), bool(match[2])
        reading = readings.get(id(s))
        if reading is not None and reading.printed is not None:
            kern = _KERN_BY_VALUE[reading.printed]
            if reading.dotted is not None:
                dotted = reading.dotted
        values[id(s)] = (kern, dotted)
    return values


def _length(kern: int, dotted: bool) -> Fraction:
    return Fraction(1, kern) * (Fraction(3, 2) if dotted else 1)


def _support(
    bars: list[list["SymbolChord"]],
    lengths: list[Fraction | None],
    candidates: dict[int, str],
    lone_rests: dict[int, Fraction],
) -> dict[Fraction, int]:
    """How many bars under each trusted length add up as the writer places them, each
    with its own candidates applied."""
    from homr import (  # noqa: PLC0415 - it imports this module
        music_xml_generator as writer,
    )

    support: dict[Fraction, int] = {}
    for number, bar in enumerate(bars):
        length = lengths[number] if number < len(lengths) else None
        timed = any(s.rhythm.startswith(("note", "rest")) for g in bar for s in g.symbols)
        if length is None or not timed:
            continue
        here = {id(s): candidates[id(s)] for g in bar for s in g.symbols if id(s) in candidates}
        written = (
            [writer.SymbolChord([_with(s, here) for s in g.symbols], g.tuplet_mark) for g in bar]
            if here
            else bar
        )
        ends = _written_ends(written, length, lone_rests)
        if ends and all(end == length for end in ends.values()):
            support[length] = support.get(length, 0) + 1
    return support


def _refusal(
    bar: list["SymbolChord"],
    here: dict[int, str],
    length: Fraction | None,
    support: dict[Fraction, int],
    lone_rests: dict[int, Fraction],
) -> str:
    """Why a bar's candidates are not applied, or "" when the bar proves them."""
    from homr import (  # noqa: PLC0415 - it imports this module
        music_xml_generator as writer,
    )

    if length is None:
        return NO_TRUSTED_METER
    touched = {placement._staff(s.position) for g in bar for s in g.symbols if id(s) in here}
    repaired = [writer.SymbolChord([_with(s, here) for s in g.symbols], g.tuplet_mark) for g in bar]
    before = _written_ends(bar, length, lone_rests)
    after = _written_ends(repaired, length, lone_rests)
    # A staff that either placement of the notes as read ends on the bar length shows no
    # missed tuplet, even where the writer places the bar the other way: there the staff
    # is off only for how the other staff's groups start.
    as_recognized = [
        placement.staff_ends(bar, rule, length, lone_rests)
        for rule in (placement.EARLIEST_END, placement.AFTER_SHORTEST)
    ]
    if any(ends.get(staff) == length for ends in as_recognized for staff in touched):
        return ADDS_UP_AS_RECOGNIZED
    if not all(after.get(staff) == length for staff in touched):
        return NOT_EXACT
    if not all(
        staff in after and abs(after[staff] - length) <= abs(end - length)
        for staff, end in before.items()
    ):
        return A_STAFF_GETS_WORSE
    if support.get(length, 0) < MIN_SUPPORTING_BARS:
        return TOO_FEW_SUPPORTING_BARS
    return ""


def _written_ends(
    chords: list["SymbolChord"], length: Fraction, lone_rests: dict[int, Fraction]
) -> dict[str, Fraction]:
    """Each staff's end as the writer places the bar under the trusted bar length: after
    the shortest note, or by the earliest end where that ends every staff on the length."""
    rule = placement.placement_rule(chords, length, lone_rests)
    return placement.staff_ends(chords, rule, length, lone_rests)


def _bars(groups: list["SymbolChord"]) -> list[list["SymbolChord"]]:
    """The voice's groups bar by bar, barlines left out; the last bar may have none."""
    bars: list[list[SymbolChord]] = [[]]
    for group in groups:
        if group.is_barline():
            bars.append([])
        else:
            bars[-1].append(group)
    return bars


def _staff_chords(bar: list["SymbolChord"], staff: str) -> list[Chord | None]:
    """One staff's chords of plain timed notes in a bar, grace notes left out, with None
    for each group that holds a rest on the staff, which ends any beam group."""
    chords: list[Chord | None] = []
    for group in bar:
        mine = [s for s in group.symbols if placement._staff(s.position) == staff]
        notes = [s for s in mine if s.rhythm.startswith("note") and not s.rhythm.endswith("G")]
        if any(s.rhythm.startswith("rest") for s in mine):
            chords.append(None)
        elif notes:
            chords.append(notes)
    return chords


def _plain_kern(symbol: EncodedSymbol) -> int | None:
    """The kern number of a plain note's token (8 for an eighth), or None for a tuplet's
    or anything that is not a note."""
    match = _NOTE.fullmatch(symbol.rhythm)
    if match is None:
        return None
    kern = int(match[1])
    return kern if kern > 0 and kern & (kern - 1) == 0 else None


def _units(lengths: list[Fraction]) -> int | None:
    """The total length in the lengths' largest common unit, when it is a whole number."""
    common = math.lcm(*(x.denominator for x in lengths))
    unit = Fraction(math.gcd(*(int(x * common) for x in lengths)), common)
    units = sum(lengths) / unit
    return units.numerator if units.denominator == 1 else None


def _is_tuplet_count(units: int) -> bool:
    """Three or more units, not a power of two: no plain values add up so."""
    return units >= 3 and units & (units - 1) != 0


def _scaled(rhythm: str, actual: int, normal: int) -> str | None:
    """A plain note token as the tuplet's: kern times actual over normal, the dot kept;
    None where that is no whole kern number, as for a whole note in a triplet."""
    match = _NOTE.fullmatch(rhythm)
    if match is None:
        return None
    kern = Fraction(int(match[1]) * actual, normal)
    return f"note_{kern.numerator}{match[2]}" if kern.denominator == 1 else None


def _with(symbol: EncodedSymbol, repairs: dict[int, str]) -> EncodedSymbol:
    if id(symbol) not in repairs:
        return symbol
    repaired = copy.copy(symbol)
    repaired.rhythm = repairs[id(symbol)]
    repaired._duration = None  # get_duration caches the old value
    return repaired
