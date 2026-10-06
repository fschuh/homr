"""Start the next notes when the shorter of two voices sharing a notehead ends.

Two voices that sound one pitch at the same moment often share a printed notehead: one
voice's stem rises from it and the other's falls from it, each with its own flags or
beams, as when an eighth note sits on the first 16th of a run. The transformer gives
such a head one token, usually with the longer value, and the shorter voice's note is
lost. The MusicXML writer then starts the next notes when the longer value ends, so
every later note in the bar starts late and the bar overflows.

Where the page shows a second stem with a shorter value than the token, the writer
starts the next notes when that shorter value ends instead, as if the lost note were
still sounding. Nothing else changes: no note is added, and the token keeps its value
and its one notehead.

The page decides which note and which value; the bar only confirms. A bar's repairs
are applied together, and only when the bar does not add up as recognized and adds up
exactly to its time signature's bar length with them. That bar length is itself the
typical bar length, so the repairs may change it only to a length that some bar
needing no repair already has.

The repair needs the visual sidecar, which links tokens to noteheads. Without it, or
when it is switched off, the writer behaves exactly as it would without this module.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Protocol

from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar.note_values import SharedNoteheadReading

if TYPE_CHECKING:
    from homr.music_xml_generator import SymbolChord

TIMING_REPAIRS_VERSION = 1

APPLIED = "applied"
DECLINED = "declined"
NOT_NEEDED = "not_needed"

#: Durations of the printed values, relative to a whole note as token durations are.
DURATION_BY_VALUE = {
    "whole": Fraction(1),
    "half": Fraction(1, 2),
    "quarter": Fraction(1, 4),
    "eighth": Fraction(1, 8),
    "16th": Fraction(1, 16),
    "32nd": Fraction(1, 32),
}
VALUE_BY_DURATION = {duration: value for value, duration in DURATION_BY_VALUE.items()}

#: An undotted note outside a tuplet: the only tokens whose value the stems can confirm.
_PLAIN_NOTE = re.compile(r"note_(1|2|4|8|16|32)")


class SharedNoteheadSource(Protocol):
    """What the repair needs from the visual sidecar."""

    def read_shared_notehead(self, symbol: EncodedSymbol) -> SharedNoteheadReading | None: ...

    def record_shared_notehead_timing(self, record: "SharedNoteheadRecord") -> None: ...

    def set_timing_repairs_enabled(self, enabled: bool) -> None: ...


@dataclass(frozen=True)
class SharedNoteheadRecord:
    """What became of one token whose notehead carries two stems."""

    symbol: EncodedSymbol
    part: int
    measure: int
    reading: SharedNoteheadReading
    #: The printed value the next notes start after, if a repair was proposed.
    next_notes_after: str | None
    status: str
    reason: str


def shared_notehead_candidate(
    symbol: EncodedSymbol, reading: SharedNoteheadReading
) -> tuple[Fraction | None, str]:
    """When the next notes should start after this token's moment, or why not at all.

    The shorter value is the shortest one read on either stem. When both stems are read,
    the token must carry one of their values: a token matching neither is a misread
    this repair does not judge.
    """
    if not _PLAIN_NOTE.fullmatch(symbol.rhythm):
        return None, "not_a_plain_note"
    if reading.dotted is True:
        return None, "dotted"
    printed = [DURATION_BY_VALUE[value] for value in (reading.up, reading.down) if value]
    if not printed:
        return None, "values_unclear"
    token = symbol.get_duration().fraction
    if len(printed) == 2 and token not in printed:
        return None, "token_matches_neither"
    shorter = min(printed)
    if shorter >= token:
        return None, "token_is_shortest"
    return shorter, ""


@dataclass
class _Bar:
    chords: list[list[EncodedSymbol]] = field(default_factory=list)
    #: The time signature tokens met in this bar, in order.
    time_signatures: list[EncodedSymbol] = field(default_factory=list)
    #: The section the bar is measured in: the identity of the time signature in effect.
    section: int | None = None
    candidates: list[int] = field(default_factory=list)


class TimingRepairs:
    """The shared-notehead repairs one voice gets, decided before it is written.

    The writer takes the division and typical bar lengths from here, adds the ends
    from ``shared_notehead_ends`` to the sounding notes before placing each group, and
    otherwise writes as usual.
    """

    def __init__(
        self,
        groups: list["SymbolChord"],
        source: SharedNoteheadSource | None,
        enabled: bool = True,
    ) -> None:
        from homr import (  # noqa: PLC0415 - it imports this module
            music_xml_generator as writer,
        )

        self._writer = writer
        self._groups = groups
        self._readings: dict[int, SharedNoteheadReading] = {}
        self._candidates: dict[int, Fraction] = {}
        self._outcomes: dict[int, tuple[str, str]] = {}
        self._accepted: dict[int, Fraction] = {}
        self._division, self._nominator = writer.find_division_and_time_signature_nominator(groups)
        self._sections = writer.time_signature_section_nominators(groups)
        if source is not None:
            source.set_timing_repairs_enabled(enabled)
        self._source = source if enabled else None
        if self._source is not None:
            self._collect(self._source)
        if self._candidates:
            self._plan()

    def division_and_nominator(self) -> tuple[int, Fraction]:
        return self._division, self._nominator

    def section_nominators(self) -> dict[int, Fraction]:
        return dict(self._sections)

    def lost_note_lengths(self, group: "SymbolChord") -> list[Fraction]:
        """How long the lost shorter notes of this group's applied repairs last.

        Unlike ``shared_notehead_ends`` it records nothing, so it can be asked ahead.
        """
        return [self._accepted[id(s)] for s in group.symbols if id(s) in self._accepted]

    def shared_notehead_ends(
        self, group: "SymbolChord", clock: Fraction, *, part: int, measure: int
    ) -> list[Fraction]:
        """When the lost shorter notes of this group end, to count among sounding notes.

        Called once as each group is written; it also records what became of every
        shared notehead in the group.
        """
        ends = []
        for symbol in group.symbols:
            key = id(symbol)
            reading = self._readings.get(key)
            if reading is None:
                continue
            if key in self._accepted:
                ends.append(clock + self._accepted[key])
            status, reason = self._outcomes[key]
            shorter = self._candidates.get(key)
            if self._source is not None:
                self._source.record_shared_notehead_timing(
                    SharedNoteheadRecord(
                        symbol=symbol,
                        part=part,
                        measure=measure,
                        reading=reading,
                        next_notes_after=VALUE_BY_DURATION[shorter] if shorter else None,
                        status=status,
                        reason=reason,
                    )
                )
        return ends

    def _collect(self, source: SharedNoteheadSource) -> None:
        for group in self._groups:
            notes = [s for s in group.symbols if s.rhythm.startswith("note")]
            for symbol in notes:
                # Only a head standing alone on its staff at its moment is read: the
                # stems of a chord pass its other heads.
                if sum(other.position == symbol.position for other in notes) != 1:
                    continue
                reading = source.read_shared_notehead(symbol)
                if reading is None or reading.reason != "two_stems":
                    continue
                key = id(symbol)
                self._readings[key] = reading
                shorter, reason = shared_notehead_candidate(symbol, reading)
                if shorter is None:
                    status = NOT_NEEDED if reason == "token_is_shortest" else DECLINED
                    self._outcomes[key] = (status, reason)
                else:
                    self._candidates[key] = shorter

    def _plan(self) -> None:
        writer = self._writer
        bars = self._bars()
        with_candidates = self._measured(self._candidates)
        _, nominator = writer.find_division_and_time_signature_nominator(with_candidates)
        sections_with = writer.time_signature_section_nominators(with_candidates)

        # Lengths of the bars no repair touches: a typical length the repairs produce
        # must be one of these, or the repairs would be confirming a meter they made.
        unrepaired: set[Fraction] = set()
        unrepaired_by_section: dict[int | None, set[Fraction]] = defaultdict(set)
        for bar in bars:
            if bar.candidates:
                continue
            length = writer._bar_length_evidence(bar.chords)
            if length > 0:
                unrepaired.add(length)
                unrepaired_by_section[bar.section].add(length)
        if nominator != self._nominator and nominator not in unrepaired:
            nominator = self._nominator
        sections = dict(self._sections)
        for key, length in sections_with.items():
            if length == self._sections.get(key) or length in unrepaired_by_section[key]:
                sections[key] = length

        # The bar length each bar is written to, as the writer's conversion state has it.
        bar_length = nominator if nominator > 0 and (nominator * 8).denominator == 1 else None
        for bar in bars:
            for time_signature in bar.time_signatures:
                beat_type = int(time_signature.rhythm.split("/")[1])
                typical = sections.get(id(time_signature), nominator)
                bar_length = Fraction(max(int(typical * beat_type), 1), beat_type)
            if not bar.candidates:
                continue
            caps = {key: self._candidates[key] for key in bar.candidates}
            before = writer._bar_length_evidence(bar.chords)
            after = writer._bar_length_evidence(
                [chord + self._lost_notes(chord, caps) for chord in bar.chords]
            )
            if bar_length is None:
                outcome = (DECLINED, "no_bar_length")
            elif before == bar_length:
                outcome = (DECLINED, "bar_already_adds_up")
            elif after != bar_length:
                outcome = (DECLINED, "bar_still_does_not_add_up")
            else:
                outcome = (APPLIED, "bar_adds_up")
                self._accepted.update(caps)
            for key in bar.candidates:
                self._outcomes[key] = outcome

        if self._accepted:
            # The division must express the lost notes' ends too.
            self._division, _ = writer.find_division_and_time_signature_nominator(
                self._measured(self._accepted)
            )
            self._nominator, self._sections = nominator, sections

    def _bars(self) -> list[_Bar]:
        bars = []
        bar = _Bar()
        section: int | None = None
        for group in [*self._groups, self._writer.SymbolChord([EncodedSymbol("barline")])]:
            if group.is_barline():
                bar.section = section
                bars.append(bar)
                bar = _Bar()
                continue
            for symbol in group.symbols:
                if symbol.rhythm.startswith("timeSignature/"):
                    section = id(symbol)
                    bar.time_signatures.append(symbol)
                elif id(symbol) in self._candidates:
                    bar.candidates.append(id(symbol))
            bar.chords.append(group.symbols)
        return bars

    def _measured(self, caps: dict[int, Fraction]) -> list["SymbolChord"]:
        """The groups with each lost note as a rest, only to measure bars with."""
        return [
            (
                self._writer.SymbolChord([*group.symbols, *lost], group.tuplet_mark)
                if (lost := self._lost_notes(group.symbols, caps))
                else group
            )
            for group in self._groups
        ]

    @staticmethod
    def _lost_notes(chord: list[EncodedSymbol], caps: dict[int, Fraction]) -> list[EncodedSymbol]:
        return [
            EncodedSymbol(f"rest_{caps[id(symbol)].denominator}", position=symbol.position)
            for symbol in chord
            if id(symbol) in caps
        ]
