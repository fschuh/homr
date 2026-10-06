"""Choose, bar by bar, how the MusicXML writer decides when each chord group starts.

Tokens say which notes start together, not when each group starts. The writer starts the
next group when the earliest sounding note ends, which places hands that move
independently on their beats. That rule trusts every length, though. When one note is
read too short, such as a 16th read as a 32nd under a held chord, the next group starts
off the beat while the other hand's notes still end on it. From there each group starts
at whichever of the two ends comes first, so the rest of the bar rushes and the staff
falls silent before the barline. Starting each group after the shortest note of the
group before, as the writer used to, turns the same misread into a small shift.

The writer places every bar both ways and keeps the earliest-end placement unless the
other leaves every staff at least as close to the bar length and some staff closer, or
ends every staff within an eighth note of it where the earliest-end placement does not.
The second test takes the bars the first refuses only because a staff that ended on the
barline ends a little past it instead, as when the notes after the misread are written
on the staff that holds the chord. No note changes its value: each bar is written as one
of the two rules writes it. A bar whose shared-notehead repairs apply keeps the
earliest-end placement, since the repairs are defined by it, and so does a bar without a
bar length to measure against.
"""

from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homr.music_xml_generator import SymbolChord
    from homr.visual_sidecar.timing_repairs import TimingRepairs

#: The next group starts when the earliest sounding note ends.
EARLIEST_END = "earliest_end"
#: The next group starts when the shortest note of the group just written ends.
AFTER_SHORTEST = "after_shortest"
#: How far from the bar length a staff may end for the bar to add up: less than an eighth.
NEAR_THE_BARLINE = Fraction(1, 8)


def after_shortest_advance(group: "SymbolChord") -> Fraction:
    """How far the next group starts after this one, by the shortest note in it."""
    durations = [
        symbol.get_duration().fraction
        for symbol in group.symbols
        if symbol.rhythm.startswith(("note", "rest"))
    ]
    timed = [duration for duration in durations if duration > 0]  # grace notes take no time
    return min(timed) if timed else Fraction(0)


def staff_ends(
    chords: list["SymbolChord"],
    rule: str,
    bar_length: Fraction | None,
    lone_rests: set[int],
) -> dict[str, Fraction]:
    """When each staff's last note or rest ends if the writer places this bar by ``rule``.

    ``chords`` are a bar's groups as the writer meets them. Only groups the writer times are
    placed: those led by a note or rest, other than a multi-measure rest. A staff's lone rest
    met at the bar's start is measured as the measure rest the writer makes of it.
    """
    from homr import (  # noqa: PLC0415 - it imports this module
        music_xml_generator as writer,
    )

    clock = Fraction(0)
    sounding: list[Fraction] = []
    ends: dict[str, Fraction] = {}
    for group in chords:
        if not group.symbols:
            continue
        lead = group.symbols[0].rhythm
        if not lead.startswith(("note", "rest")):
            continue
        if len(group.symbols) == 1 and lead.endswith("m"):
            continue
        timed = writer.SymbolChord(
            [s for s in group.symbols if s.rhythm.startswith(("note", "rest"))],
            group.tuplet_mark,
        )
        written = (
            writer._with_measure_rests(timed, lone_rests, bar_length)
            if clock == Fraction(0) and bar_length is not None
            else timed
        )
        for symbol in written.symbols:
            duration = symbol.get_duration().fraction
            if duration > 0:
                staff = _staff(symbol.position)
                ends[staff] = max(ends.get(staff, Fraction(0)), clock + duration)
        if rule == EARLIEST_END:
            advance = writer._advance_to_next_group(written, clock, sounding)
        else:
            advance = after_shortest_advance(written)
        clock += advance
        sounding = [end for end in sounding if end > clock]
    return ends


def after_shortest_is_closer(
    earliest_end: dict[str, Fraction],
    after_shortest: dict[str, Fraction],
    bar_length: Fraction,
) -> bool:
    """Whether placing groups after the shortest note leaves every staff at least as close
    to the bar length as the earliest-end placement does, and some staff closer.

    Both must measure the same staves: they place the same notes.
    """
    if not earliest_end or set(earliest_end) != set(after_shortest):
        return False
    distances = [
        (abs(after_shortest[staff] - bar_length), abs(earliest_end[staff] - bar_length))
        for staff in earliest_end
    ]
    return all(shortest <= earliest for shortest, earliest in distances) and any(
        shortest < earliest for shortest, earliest in distances
    )


def after_shortest_adds_up(
    earliest_end: dict[str, Fraction],
    after_shortest: dict[str, Fraction],
    bar_length: Fraction,
) -> bool:
    """Whether placing groups after the shortest note ends every staff within an eighth
    note of the bar length while the earliest-end placement leaves some staff farther.

    Both must measure the same staves: they place the same notes.
    """
    if not earliest_end or set(earliest_end) != set(after_shortest):
        return False

    def near(ends: dict[str, Fraction]) -> bool:
        return all(abs(end - bar_length) < NEAR_THE_BARLINE for end in ends.values())

    return near(after_shortest) and not near(earliest_end)


def _staff(position: str) -> str:
    # The writer puts every symbol that is not on the upper staff on the lower one.
    return "upper" if position == "upper" else "lower"


class BarPlacement:
    """The placement rule each bar of one voice is written with.

    The writer asks for each group's advance; the bar's rule is decided when its first
    timed group is written, with the bar length then in effect.
    """

    def __init__(
        self,
        groups: list["SymbolChord"],
        lone_rests: set[int],
        timing_repairs: "TimingRepairs",
    ) -> None:
        self._lone_rests = lone_rests
        self._bar_of_group: list[int] = []
        self._bars: list[list[SymbolChord]] = [[]]
        for group in groups:
            self._bar_of_group.append(len(self._bars) - 1)
            if group.is_barline():
                self._bars.append([])
            else:
                self._bars[-1].append(group)
        self._repaired = [
            any(timing_repairs.lost_note_lengths(group) for group in bar) for bar in self._bars
        ]
        self._rules: dict[int, str] = {}

    def rule(self, group_no: int, bar_length: Fraction | None) -> str:
        """The rule the bar holding group ``group_no`` is placed by, deciding it if needed."""
        bar = self._bar_of_group[group_no]
        if bar not in self._rules:
            self._rules[bar] = self._decide(bar, bar_length)
        return self._rules[bar]

    def advance(
        self,
        group_no: int,
        written: "SymbolChord",
        clock: Fraction,
        sounding: list[Fraction],
        bar_length: Fraction | None,
    ) -> Fraction:
        """How far the next group starts after ``written``, updating ``sounding`` in place."""
        from homr import (  # noqa: PLC0415 - it imports this module
            music_xml_generator as writer,
        )

        if self.rule(group_no, bar_length) == EARLIEST_END:
            return writer._advance_to_next_group(written, clock, sounding)
        return after_shortest_advance(written)

    def _decide(self, bar: int, bar_length: Fraction | None) -> str:
        if bar_length is None or self._repaired[bar]:
            return EARLIEST_END
        chords = self._bars[bar]
        earliest_end = staff_ends(chords, EARLIEST_END, bar_length, self._lone_rests)
        after_shortest = staff_ends(chords, AFTER_SHORTEST, bar_length, self._lone_rests)
        if after_shortest_is_closer(
            earliest_end, after_shortest, bar_length
        ) or after_shortest_adds_up(earliest_end, after_shortest, bar_length):
            return AFTER_SHORTEST
        return EARLIEST_END
