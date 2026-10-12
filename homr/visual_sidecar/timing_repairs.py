"""Start the next notes when the shorter of two voices sharing a notehead ends.

Two voices that sound one pitch at the same moment often share a printed notehead: one
voice's stem rises from it and the other's falls from it, each with its own flags or
beams, as when an eighth note sits on the first 16th of a run. The transformer gives
such a head one token, usually with the longer value, and the shorter voice's note is
lost. The MusicXML writer then starts the next notes when the longer value ends, so
every later note in the bar starts late and the bar overflows.

Where the page shows a second stem with a shorter value than the token, the writer
starts the next notes when that shorter value ends instead, as if the lost note were
still in its group. Nothing else changes: no note is added, and the token keeps its
value and its one notehead.

The page decides which note and which value; the bar only confirms. A bar's repairs
are applied together, and only when the bar, placed as the writer places it (see
``homr.bar_placement``), does not add up as recognized and adds up exactly to its time
signature's bar length with them. That bar length is itself the typical bar length, so
the repairs may change it only to a length that some bar needing no repair already has.

The repair needs the visual sidecar, which links tokens to noteheads. Without it, or
when it is switched off, the writer behaves exactly as it would without this module.

Printed time signatures
-----------------------

With the page's time signatures read (see ``time_signatures``), a time-signature token a
trusted read lands on governs its section with the printed meter instead of the typical
bar length of the section's bars. The meter in force at the end of the previous page of
the score, when the caller passes it, governs the bars before the page's first time
signature: it is written into the page's first measure without being printed, unless
the page starts with a time signature of its own.

The transformer also reads time signatures where none is printed, as where a key changes
mid-piece. Under a trusted meter (read on the page or carried to it), a token with
nothing shaped like a time signature printed near it is not written, and its bars keep
the trusted meter. A token at a printed glyph that no read could make out ends the
trusted meter instead: its section's meter is inferred, as without the reader. The meter
in force at the page's end, for the next page, is the trusted one, or none once an
inferred section starts.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Protocol

import musicxml.xmlelement.xmlelement as mxl

from homr.transformer.vocabulary import EncodedSymbol
from homr.visual_sidecar.note_values import SharedNoteheadReading
from homr.visual_sidecar.time_signatures import NO_READ, Meter

if TYPE_CHECKING:
    from homr.music_xml_generator import ConversionState, SymbolChord

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

    def read_time_signature(self, symbol: EncodedSymbol) -> Meter | None: ...

    def printed_time_symbol(self, symbol: EncodedSymbol) -> str | None: ...

    def time_signature_status(self, symbol: EncodedSymbol) -> str | None: ...

    def carried_meter(self) -> Meter | None: ...

    def record_final_meter(self, meter: Meter | None) -> None: ...


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
    from ``shared_notehead_ends`` to the sounding notes before placing each group, counts
    the lengths from ``lost_note_lengths`` among each group's notes, and otherwise writes
    as usual.
    """

    def __init__(
        self,
        groups: list["SymbolChord"],
        source: SharedNoteheadSource | None,
        enabled: bool = True,
        guess_meter_per_section: bool = False,
    ) -> None:
        from homr import (  # noqa: PLC0415 - it imports this module
            music_xml_generator as writer,
        )

        self._writer = writer
        self._groups = groups
        self._per_section = guess_meter_per_section
        self._readings: dict[int, SharedNoteheadReading] = {}
        self._candidates: dict[int, Fraction] = {}
        self._outcomes: dict[int, tuple[str, str]] = {}
        self._accepted: dict[int, Fraction] = {}
        self._division, self._nominator, self._sections = self._guess_meter(groups)
        self._printed: dict[int, Fraction] = {}
        self._printed_symbols: dict[int, str] = {}
        self._ignored: set[int] = set()
        self._carried: Meter | None = None
        if source is not None:
            self._follow_meters(source)
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

    def ignores(self, symbol: EncodedSymbol) -> bool:
        """Whether a time-signature token is left unwritten: nothing is printed there."""
        return id(symbol) in self._ignored

    def name_time_symbol(self, symbol: EncodedSymbol, attributes: mxl.XMLAttributes) -> None:
        """Writes the time signature just written for this token as common or cut time where
        the page prints that symbol."""
        printed_symbol = self._printed_symbols.get(id(symbol))
        if printed_symbol is None:
            return
        times = [child for child in attributes.get_children() if isinstance(child, mxl.XMLTime)]
        times[-1].symbol = printed_symbol

    def restate_carried_meter(
        self, attributes: mxl.XMLAttributes, state: "ConversionState"
    ) -> None:
        """Measures the page's first bars against the meter carried from the previous page,
        and writes it, unprinted, into the first measure's attributes unless the page
        starts with a time signature of its own."""
        if self._carried is None:
            return
        beats, beat_type = self._carried
        state.bar_length = state.trusted_bar_length = Fraction(beats, beat_type)
        if self._starts_with_time_signature():
            return
        time = mxl.XMLTime(print_object="no")
        time.add_child(mxl.XMLBeats(value_=str(beats)))
        time.add_child(mxl.XMLBeatType(value_=str(beat_type)))
        attributes.add_child(time)
        state.beats = beats

    def trust_printed_meter(self, symbol: EncodedSymbol, state: "ConversionState") -> None:
        """Trusts the bar length just written for a time-signature token where a trusted
        read lands on it, and stops trusting any bar length where none does."""
        state.trusted_bar_length = state.bar_length if id(symbol) in self._printed else None

    def _follow_meters(self, source: SharedNoteheadSource) -> None:
        """Applies the printed and carried meters, and records the page's final one."""
        trusted = self._carried = source.carried_meter()
        for group in self._groups:
            for symbol in group.symbols:
                if not symbol.rhythm.startswith("timeSignature/"):
                    continue
                read = source.read_time_signature(symbol)
                if read is not None:
                    self._printed[id(symbol)] = Fraction(*read)
                    printed_symbol = source.printed_time_symbol(symbol)
                    if printed_symbol is not None:
                        self._printed_symbols[id(symbol)] = printed_symbol
                    trusted = read
                elif trusted is not None and source.time_signature_status(symbol) == NO_READ:
                    self._ignored.add(id(symbol))
                else:
                    trusted = None
        self._sections.update(self._printed)
        source.record_final_meter(trusted)

    def _guess_meter(
        self, groups: list["SymbolChord"]
    ) -> tuple[int, Fraction, dict[int, Fraction]]:
        """The division, the typical bar length, and with ``guess_meter_per_section`` the
        typical bar length of each time signature's section, as the writer guesses them."""
        division, nominator = self._writer.find_division_and_time_signature_nominator(groups)
        if not self._per_section:
            return division, nominator, {}
        return (
            division,
            self._writer.sounding_time_signature_nominator(groups),
            self._writer.time_signature_section_nominators(groups),
        )

    def _typical_length_of(self, chords: list[list[EncodedSymbol]]) -> Fraction:
        """A bar's length as ``_guess_meter`` measures it for the typical bar length."""
        if self._per_section:
            return self._writer._bar_length_evidence(chords)
        return sum(
            (self._writer.SymbolChord(chord).get_duration() for chord in chords), Fraction(0)
        )

    def _starts_with_time_signature(self) -> bool:
        """Whether a written time signature comes before the voice's first note or rest."""
        for group in self._groups:
            for symbol in group.symbols:
                if symbol.rhythm.startswith("timeSignature/") and not self.ignores(symbol):
                    return True
                if symbol.rhythm.startswith(("note", "rest")):
                    return False
        return False

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
        _, nominator, sections_with = self._guess_meter(with_candidates)

        # Lengths of the bars no repair touches: a typical length the repairs produce
        # must be one of these, or the repairs would be confirming a meter they made.
        unrepaired: set[Fraction] = set()
        unrepaired_by_section: dict[int | None, set[Fraction]] = defaultdict(set)
        for bar in bars:
            if bar.candidates:
                continue
            length = self._typical_length_of(bar.chords)
            if length > 0:
                unrepaired.add(length)
                unrepaired_by_section[bar.section].add(length)
        if nominator != self._nominator and nominator not in unrepaired:
            nominator = self._nominator
        sections = dict(self._sections)
        for key, length in sections_with.items():
            if key in self._printed:
                continue
            if length == self._sections.get(key) or length in unrepaired_by_section[key]:
                sections[key] = length

        # The bar length each bar is written to, and whether it is trusted, as the writer's
        # conversion state has them; the bar is measured as the writer places it.
        bar_length = nominator if nominator > 0 and (nominator * 8).denominator == 1 else None
        trusted: Fraction | None = None
        if self._carried is not None:
            bar_length = trusted = Fraction(*self._carried)
        for bar in bars:
            for time_signature in bar.time_signatures:
                if id(time_signature) in self._ignored:
                    continue
                beat_type = int(time_signature.rhythm.split("/")[1])
                typical = sections.get(id(time_signature), nominator)
                bar_length = Fraction(max(int(typical * beat_type), 1), beat_type)
                trusted = bar_length if id(time_signature) in self._printed else None
            if not bar.candidates:
                continue
            caps = {key: self._candidates[key] for key in bar.candidates}
            before = writer._written_bar_length(bar.chords, trusted)
            after = writer._written_bar_length(
                [chord + self._lost_notes(chord, caps) for chord in bar.chords], trusted
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
