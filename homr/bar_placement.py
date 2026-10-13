"""Choose, bar by bar, how the MusicXML writer decides when each chord group starts.

Tokens say which notes start together, not when each group starts. The writer starts the
next group when the shortest note of the group just written ends. Where one hand moves
under a note the other hand still holds, that starts the later groups late and the bar
overflows: one late hiccup, after which the notes keep their spacing. Starting the next
group when the earliest sounding note ends instead, as upstream homr does, places such
hands on their beats, but it trusts every length. When one note is read too short, such
as a 16th read as a 32nd under a held chord, the next group starts off the beat while
the other hand's notes still end on it. From there each group starts at whichever of the
two ends comes first, so the rest of the bar rushes and the staff falls silent before the
barline. A page read right rushes too where a staff holds three voices and an inner one
ends before the melody does: the next beat starts when the inner voice ends.

So a bar is placed by the earliest end only where the bar proves it: its meter is
trusted, read on the page or carried from the previous page, and the earliest-end
placement ends every staff exactly on the bar length where placing after the shortest
note does not. Every other bar is placed after the shortest note. A meter the writer
infers from the page's bars proves nothing, since misread bars shift it. No note changes
its value: each bar is written as one of the two rules writes it. Under either rule the
lost shorter note of an applied shared-notehead repair counts among its group's notes.
"""

from collections.abc import Callable
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homr.music_xml_generator import SymbolChord
    from homr.visual_sidecar.timing_repairs import TimingRepairs

#: The next group starts when the earliest sounding note ends.
EARLIEST_END = "earliest_end"
#: The next group starts when the shortest note of the group just written ends.
AFTER_SHORTEST = "after_shortest"

#: How long the lost shorter notes of a group's applied shared-notehead repairs last.
LostNotes = Callable[["SymbolChord"], list[Fraction]]


def no_lost_notes(group: "SymbolChord") -> list[Fraction]:
    return []


def after_shortest_advance(group: "SymbolChord", lost: list[Fraction] | None = None) -> Fraction:
    """How far the next group starts after this one, by the shortest note in it or among
    the ``lost`` notes of its shared-notehead repairs."""
    durations = [
        symbol.get_duration().fraction
        for symbol in group.symbols
        if symbol.rhythm.startswith(("note", "rest"))
    ]
    timed = [duration for duration in durations if duration > 0]  # grace notes take no time
    timed.extend(lost or [])
    return min(timed) if timed else Fraction(0)


def staff_ends(
    chords: list["SymbolChord"],
    rule: str,
    bar_length: Fraction | None,
    lone_rests: dict[int, Fraction],
    lost_notes: LostNotes = no_lost_notes,
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
        lost = lost_notes(written)
        if rule == EARLIEST_END:
            sounding.extend(clock + length for length in lost)
            advance = writer._advance_to_next_group(written, clock, sounding)
        else:
            advance = after_shortest_advance(written, lost)
        clock += advance
        sounding = [end for end in sounding if end > clock]
    return ends


def earliest_end_preferred(
    earliest_end: dict[str, Fraction],
    after_shortest: dict[str, Fraction],
    bar_length: Fraction,
) -> bool:
    """Whether a bar whose staves end so is placed by the earliest end: that placement
    ends every staff exactly on the bar length, and placing after the shortest note does
    not."""

    def exact(ends: dict[str, Fraction]) -> bool:
        return bool(ends) and all(end == bar_length for end in ends.values())

    return exact(earliest_end) and not exact(after_shortest)


def placement_rule(
    chords: list["SymbolChord"],
    trusted_bar_length: Fraction | None,
    lone_rests: dict[int, Fraction],
    lost_notes: LostNotes = no_lost_notes,
) -> str:
    """The rule a bar is placed by, given its bar length if its meter is trusted."""
    if trusted_bar_length is None:
        return AFTER_SHORTEST
    earliest_end = staff_ends(chords, EARLIEST_END, trusted_bar_length, lone_rests, lost_notes)
    after_shortest = staff_ends(chords, AFTER_SHORTEST, trusted_bar_length, lone_rests, lost_notes)
    if earliest_end_preferred(earliest_end, after_shortest, trusted_bar_length):
        return EARLIEST_END
    return AFTER_SHORTEST


def _staff(position: str) -> str:
    # The writer puts every symbol that is not on the upper staff on the lower one.
    return "upper" if position == "upper" else "lower"


class BarPlacement:
    """The placement rule each bar of one voice is written with.

    The writer asks for each group's advance; the bar's rule is decided when its first
    timed group is written, with the trusted bar length then in effect.
    """

    def __init__(
        self,
        groups: list["SymbolChord"],
        lone_rests: dict[int, Fraction],
        timing_repairs: "TimingRepairs",
    ) -> None:
        self._lone_rests = lone_rests
        self._lost_notes: LostNotes = timing_repairs.lost_note_lengths
        self._bar_of_group: list[int] = []
        self._bars: list[list[SymbolChord]] = [[]]
        for group in groups:
            self._bar_of_group.append(len(self._bars) - 1)
            if group.is_barline():
                self._bars.append([])
            else:
                self._bars[-1].append(group)
        self._rules: dict[int, str] = {}

    def rule(self, group_no: int, trusted_bar_length: Fraction | None) -> str:
        """The rule the bar holding group ``group_no`` is placed by, deciding it if needed."""
        bar = self._bar_of_group[group_no]
        if bar not in self._rules:
            self._rules[bar] = placement_rule(
                self._bars[bar], trusted_bar_length, self._lone_rests, self._lost_notes
            )
        return self._rules[bar]

    def advance(
        self,
        group_no: int,
        written: "SymbolChord",
        clock: Fraction,
        sounding: list[Fraction],
        trusted_bar_length: Fraction | None,
    ) -> Fraction:
        """How far the next group starts after ``written``, updating ``sounding`` in place.

        ``sounding`` already holds the ends of the group's lost shorter notes."""
        from homr import (  # noqa: PLC0415 - it imports this module
            music_xml_generator as writer,
        )

        if self.rule(group_no, trusted_bar_length) == EARLIEST_END:
            return writer._advance_to_next_group(written, clock, sounding)
        return after_shortest_advance(written, self._lost_notes(written))
