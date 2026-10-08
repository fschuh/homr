"""Read each linked note's value from the printed symbol and compare it with the token.

A note's value is printed in a few independent marks: a hollow or filled notehead, a stem
or none, the number of flags or beams at the stem's free end, and an augmentation dot.
This reader measures those marks on the page and reports the value only when every check
it relies on agrees; otherwise it reports nothing. Tuplets are not read: a triplet eighth
is compared as an eighth.

The notes of a chord share one stem, its flags or beams and its dots, so they are read
together: the stem's direction comes from the outermost noteheads, and a dot printed
beside any of them dots the chord.

This is diagnostic only. A reading never changes the MusicXML or any note link; it is
exported beside them so a consumer can show where the printed value and the recognized
one disagree.

A single notehead that two voices share carries two stems, one rising from its right
edge and one falling from its left, each with its own flags or beams. Those are read
separately by ``read_shared_notehead``; the shared-notehead timing repair in
``timing_repairs`` relies on that reading.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from homr.model import Staff
from homr.type_definitions import NDArray
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform
from homr.visual_sidecar.models import VisualGroup
from homr.visual_sidecar.rests import SegmentationMasks

NOTE_VALUE_VERIFICATION_VERSION = 1

AGREES = "agrees"
DISAGREES = "disagrees"
UNKNOWN = "unknown"

NO_STEM = "no_stem"
HOLLOW = "hollow"
FILLED = "filled"

#: Printed values by the number of flags or beams at a filled note's stem.
VALUES_BY_BANDS = ("quarter", "eighth", "16th", "32nd")
#: Plain kern numbers and the value each names.
VALUES_BY_KERN = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "16th", 32: "32nd"}

# All sizes are in staff spaces.
#: The notehead's core: an ellipse this fraction of the notehead's size. A filled head's
#: core is all ink; a hollow head's is mostly paper. Between the two the head is not read.
CORE_SCALE = 0.4
FILLED_CORE_INK = 0.95
HOLLOW_CORE_INK = 0.6
#: Gray levels a filled head's core may be lighter than the darkest tenth of the head.
CORE_CONTRAST = 60
#: Bands are counted along columns this far beside the stem, on each side.
BAND_COLUMN_OFFSETS = (0.3, 0.55)
#: How far from the stem's free end towards the notehead bands are counted. Three beams
#: span about two staff spaces; the window stops short of the notehead itself.
BAND_WINDOW = 2.5
BAND_WINDOW_NOTEHEAD_CLEARANCE = 1.2
#: The window also looks this far past the stem's end, where only paper may follow:
#: ink there is a beam the stem trace fell short of, or a fingering numeral.
BAND_BEYOND = 0.75
#: A flag or beam crossing a column is at least this thick, and at most this thick; a
#: thinner run is a slur or tie, a thicker one another symbol in the way.
BAND_THICKNESS = (0.12, 0.8)
#: A run lying on segmentation's notehead mask for at least this share of its length is
#: not counted: it may be a notehead on the stem that the chord's group left out, which
#: crosses the columns no thicker than a beam.
BAND_NOTEHEAD_SHARE = 0.5
#: The stem contours recorded on a chord's notes lie within this of each other, and the
#: stem within this of a notehead's edge.
STEM_SPREAD = 0.5
STEM_ATTACHMENT = 0.3
#: A stem stands at least this far right of the centre of a head it rises from, or
#: this far left of the centre of a head it falls from.
STEM_SIDE = 0.2
#: The share of the chord's height, from its top head to its bottom head, the stem covers.
STEM_SPAN = 0.9
#: A stem shorter than this cannot carry the window, so its bands are not counted.
MIN_STEM_LENGTH = 2.2
#: Ink running on this far beyond the chord's other end means a second stem leaves it
#: there: two voices share the noteheads, and the value is not read.
OPPOSITE_STEM_LENGTH = 1.0
#: A whole note has no stem: no vertical run this long starts within the slack of either
#: side of its head, whose width segmentation only estimates.
STEMLESS_RUN = 2.0
STEMLESS_EDGE_SLACK = 0.3
#: A thin run centred this close to a staff or ledger line position is a leftover of the
#: line, which the staff mask does not always cover, not a flag or beam.
LINE_TOLERANCE = 0.2
LINE_THICKNESS = 0.3
#: An augmentation dot: round, isolated ink this size, right of the chord's noteheads.
DOT_SIZE = (0.2, 0.65)
DOT_REACH = (0.1, 1.4)
#: A dot this close above or below another notehead is that note's staccato.
STACCATO_REACH = 2.0
STACCATO_HALF_WIDTH = 0.25
#: A vertical run at least this long, this close right of the dot, is a barline.
REPEAT_BARLINE_REACH = 1.2
REPEAT_BARLINE_LENGTH = 3.0
#: A shared notehead's stems are looked for from this far inside the head's estimated
#: edge to this far outside it: the up-stem at the right edge, the down-stem at the left.
SHARED_STEM_SEARCH = (0.35, 0.2)
#: Columns tracing within this of the longest one belong to the same printed stem.
SHARED_STEM_WIDTH_TOLERANCE = 0.25
#: A shared notehead's stem passes no other notehead within this of it, from this far
#: past the head's centre to this far short of the stem's free end, where segmentation
#: may label a beam's end as notehead. A stem that does belongs to a chord.
SHARED_STEM_HALF_WIDTH = 0.3
SHARED_STEM_HEAD_CLEARANCE = 0.8
SHARED_STEM_END_CLEARANCE = 1.0

StaffLinesAtX = Callable[[Staff, float, int], Sequence[float]]


@dataclass(frozen=True)
class NoteValueReading:
    musicxml_id: str
    #: The value the page prints, or None when the reader is not sure.
    printed: str | None
    #: Whether the notehead carries an augmentation dot, or None when not sure.
    dotted: bool | None
    status: str
    reason: str


@dataclass(frozen=True)
class ChordMember:
    musicxml_id: str
    group: VisualGroup
    #: The transformer's rhythm token for this note, e.g. ``note_8.``.
    rhythm: str


@dataclass(frozen=True)
class PrintedValue:
    value: str | None
    dotted: bool | None
    reason: str


@dataclass(frozen=True)
class SharedNoteheadReading:
    """The two values printed on a notehead that two voices share."""

    #: The value on the stem rising from the head's right edge and on the stem falling
    #: from its left edge. Each is None when its flags or beams are not read: a flag
    #: crossing the columns at a slant, or a slur ending beside it, leaves it unread.
    up: str | None
    down: str | None
    #: Whether the notehead carries an augmentation dot, or None when not sure.
    dotted: bool | None
    #: ``two_stems`` when the head carries both stems, otherwise why it does not.
    reason: str


def recognized_value(rhythm: str) -> tuple[str | None, bool]:
    """The plain value and dot of a transformer note token; tuplets read as their plain value."""
    match = re.fullmatch(r"note_(\d+)(\.*)(G?)", rhythm)
    if match is None or match[3]:
        return None, False
    kern = int(match[1])
    plain = 2 ** int(np.floor(np.log2(kern))) if kern > 0 else 0
    return VALUES_BY_KERN.get(plain), bool(match[2])


def compare(member: ChordMember, printed: PrintedValue) -> NoteValueReading:
    """The member's token against what the page prints."""
    expected, expected_dot = recognized_value(member.rhythm)
    if expected is None:
        return NoteValueReading(member.musicxml_id, None, None, UNKNOWN, "not_a_plain_note")
    if printed.value is None:
        return NoteValueReading(member.musicxml_id, None, printed.dotted, UNKNOWN, printed.reason)
    disagrees = printed.value != expected or (
        printed.dotted is not None and printed.dotted != expected_dot
    )
    status = DISAGREES if disagrees else AGREES
    return NoteValueReading(
        member.musicxml_id, printed.value, printed.dotted, status, printed.reason
    )


def _runs(column: NDArray) -> list[tuple[int, int]]:
    """(start, length) of each run of consecutive ink in a 0/1 column."""
    padded = np.concatenate([[0], column.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return [
        (int(start), int(end - start)) for start, end in zip(edges[::2], edges[1::2], strict=True)
    ]


def _line_positions(lines: list[float], unit: float) -> list[float]:
    """The staff lines and the ledger-line positions six spaces beyond them."""
    above = [lines[0] - step * unit for step in range(1, 7)]
    below = [lines[-1] + step * unit for step in range(1, 7)]
    return above + list(lines) + below


def _near(x: float, unit: float) -> list[float]:
    """Columns at and just beside an edge whose position segmentation only estimates."""
    return [x - 0.1 * unit, x, x + 0.1 * unit]


def _on_a_line(y: float, line_positions: list[float], unit: float) -> bool:
    return min(abs(y - line) for line in line_positions) <= LINE_TOLERANCE * unit


class NoteValueReader:
    def __init__(
        self,
        image: NDArray | None,
        masks: SegmentationMasks | None,
        coordinate_transform: PredictionCoordinateTransform,
        staff_lines_at_x: StaffLinesAtX,
    ) -> None:
        self.gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image is not None and image.ndim == 3
            else image
        )
        self.masks = masks
        self.transform = coordinate_transform
        # The recovery stage's robust fit of the five line ys at an x.
        self.staff_lines_at_x = staff_lines_at_x
        self._lineless: NDArray | None = None
        self._clean: NDArray | None = None
        self._band_ink: NDArray | None = None
        self._dot_ink: NDArray | None = None

    def read_chord(self, members: list[ChordMember], staff: Staff | None) -> list[NoteValueReading]:
        """Read one chord's printed value and compare it with each member's token."""
        groups = [member.group for member in members]
        lines = self._staff_lines(groups, staff)
        if isinstance(lines, str):
            return [compare(member, PrintedValue(None, None, lines)) for member in members]
        unit = float(np.median(np.diff(lines)))
        fills = {self._fill(group, unit) for group in groups} - {None}
        if len(fills) > 1:
            # Hollow and filled heads at one moment are two voices. A filled head is
            # read with only the stem recorded on it; a hollow one is not read, as the
            # other voice's stem passes right beside it.
            return [
                compare(
                    member,
                    (
                        self._read_heads([member.group], lines, unit)
                        if self._fill(member.group, unit) == FILLED
                        else PrintedValue(None, None, "two_voices")
                    ),
                )
                for member in members
            ]
        printed = self._read_heads(groups, lines, unit)
        return [compare(member, printed) for member in members]

    def read_shared_notehead(
        self, group: VisualGroup, staff: Staff | None
    ) -> SharedNoteheadReading:
        """Read both values printed on a filled notehead that carries two stems.

        The caller passes a head that stands alone at its moment on its staff. Two
        voices sounding its pitch together may share it: one voice's stem rises from
        its right edge and the other's falls from its left, each with its own flags or
        beams. A stem that passes another notehead belongs to a chord, not to this head.
        """
        lines = self._staff_lines([group], staff)
        if isinstance(lines, str):
            return SharedNoteheadReading(None, None, None, lines)
        unit = float(np.median(np.diff(lines)))
        if self._fill(group, unit) != FILLED:
            return SharedNoteheadReading(None, None, None, "notehead_unclear")
        stems = [self._edge_stem(group, up, unit) for up in (True, False)]
        notehead_y = group.prediction_center[1]
        values: list[str | None] = []
        for stem in stems:
            if stem is None:
                return SharedNoteheadReading(None, None, None, "one_stem")
            stem_x, free_end = stem
            if self._stem_passes_a_head(stem_x, notehead_y, free_end, unit):
                return SharedNoteheadReading(None, None, None, "stem_unclear")
            bands = self._bands(stem_x, (free_end, notehead_y), lines, unit)
            readable = bands is not None and bands < len(VALUES_BY_BANDS)
            values.append(VALUES_BY_BANDS[bands] if readable and bands is not None else None)
        dotted = self._dotted([group], lines, unit)
        return SharedNoteheadReading(values[0], values[1], dotted, "two_stems")

    def _edge_stem(self, group: VisualGroup, up: bool, unit: float) -> tuple[float, float] | None:
        """The x and free end of the stem rising from the head's right edge, or falling
        from its left edge, if one at least long enough to carry bands leaves it there."""
        cx, cy = group.prediction_center
        half_width = group.prediction_notehead_size[0] / 2
        inside, outside = SHARED_STEM_SEARCH
        if up:
            first, last = cx + half_width - inside * unit, cx + half_width + outside * unit
        else:
            first, last = cx - half_width - outside * unit, cx - half_width + inside * unit
        lengths = {
            x: self._stem_length([float(x)], cy, up, unit)
            for x in range(int(np.floor(first)), int(np.ceil(last)) + 1)
        }
        longest = max(lengths.values(), default=0.0)
        if longest < MIN_STEM_LENGTH * unit:
            return None
        # A printed stem is a few columns wide, and each traces about as far.
        tolerance = SHARED_STEM_WIDTH_TOLERANCE * unit
        columns = [x for x, length in lengths.items() if length >= longest - tolerance]
        stem_x = float(np.median(columns))
        side = 1 if up else -1
        if side * (stem_x - cx) < STEM_SIDE * unit:
            return None
        return stem_x, cy - longest if up else cy + longest

    def _stem_passes_a_head(
        self, stem_x: float, notehead_y: float, free_end: float, unit: float
    ) -> bool:
        if self.masks is None:
            raise ValueError("Reading note values needs the page and its segmentation")
        # Stems shorter than MIN_STEM_LENGTH are not read, so the span is never empty.
        step = 1 if free_end > notehead_y else -1
        start = notehead_y + step * SHARED_STEM_HEAD_CLEARANCE * unit
        stop = free_end - step * SHARED_STEM_END_CLEARANCE * unit
        y0, y1 = sorted((int(start), int(stop)))
        x0 = max(0, int(stem_x - SHARED_STEM_HALF_WIDTH * unit))
        x1 = int(stem_x + SHARED_STEM_HALF_WIDTH * unit) + 1
        return bool((self.masks.notehead[y0 : y1 + 1, x0:x1] > 0).any())

    def _staff_lines(self, groups: list[VisualGroup], staff: Staff | None) -> list[float] | str:
        """The five line ys at the chord, or why the chord cannot be read."""
        if self.gray is None or self.masks is None or staff is None or not groups:
            return "no_segmentation"
        cx = float(np.mean([group.prediction_center[0] for group in groups]))
        try:
            lines = [float(y) for y in self.staff_lines_at_x(staff, cx, groups[0].staff_index)]
        except ValueError:
            return "no_staff_lines"
        if float(np.median(np.diff(lines))) < 4:
            return "staff_too_small"
        return lines

    def _read_heads(
        self, groups: list[VisualGroup], lines: list[float], unit: float
    ) -> PrintedValue:
        dotted = self._dotted(groups, lines, unit)
        fills = {self._fill(group, unit) for group in groups}
        if len(fills) != 1 or None in fills:
            return PrintedValue(None, dotted, "notehead_unclear")
        stem_x, reason = self._stem_x(groups, unit)
        if fills == {HOLLOW}:
            value, reason = self._hollow_value(groups, stem_x, reason, unit)
        else:
            value, reason = self._filled_value(groups, stem_x, reason, lines, unit)
        return PrintedValue(value, dotted, reason)

    def _hollow_value(
        self, groups: list[VisualGroup], stem_x: float | None, stem_reason: str, unit: float
    ) -> tuple[str | None, str]:
        if stem_x is not None:
            if self._free_end(groups, stem_x, unit) is None:
                return None, "stem_unclear"
            return "half", "hollow_head_with_stem"
        # Only where segmentation found no stem at all, and the page shows none at the
        # head either: segnet may have missed it, or another voice's stem may pass by.
        if stem_reason == NO_STEM and all(self._stemless(group, unit) for group in groups):
            return "whole", "hollow_head_without_stem"
        return None, "stem_unclear"

    def _filled_value(
        self,
        groups: list[VisualGroup],
        stem_x: float | None,
        stem_reason: str,
        lines: list[float],
        unit: float,
    ) -> tuple[str | None, str]:
        if stem_x is None:
            return None, stem_reason
        free = self._free_end(groups, stem_x, unit)
        if free is None:
            return None, "stem_unclear"
        bands = self._bands(stem_x, free, lines, unit)
        if bands is None:
            return None, "bands_unclear"
        if bands >= len(VALUES_BY_BANDS):
            return None, "too_many_bands"
        return VALUES_BY_BANDS[bands], f"{bands}_bands"

    def _lineless_ink(self) -> NDArray:
        """Page ink without staff lines (where its vertical run is no longer than a line)."""
        if self._lineless is None:
            if self.gray is None or self.masks is None:
                raise ValueError("Reading note values needs the page and its segmentation")
            threshold, _ = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ink = (self.gray < np.clip(threshold, 80, 200)).astype(np.uint8)
            crossing = (
                cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((4, 1), np.uint8)) > 0
            ).astype(np.uint8)
            lines = (self.masks.staff > 0).astype(np.uint8) & (1 - crossing)
            self._lineless = ink & (1 - lines)
        return self._lineless

    def _clean_ink(self) -> NDArray:
        """Lineless ink without noteheads either, so a neighbour's head is not a band."""
        if self._clean is None:
            if self.masks is None:
                raise ValueError("Reading note values needs the page and its segmentation")
            heads = cv2.dilate(
                (self.masks.notehead > 0).astype(np.uint8), np.ones((3, 3), np.uint8)
            )
            self._clean = self._lineless_ink() & (1 - heads)
        return self._clean

    def _beam_ink(self) -> NDArray:
        """Lineless ink without accidentals, which may stand beside a stem's free end.
        Noteheads stay: segmentation often labels a beam's end at the stem as notehead.
        ``_bands`` gives up where a run lies on a notehead instead of counting it."""
        if self._band_ink is None:
            if self.masks is None:
                raise ValueError("Reading note values needs the page and its segmentation")
            accidentals = (self.masks.clefs_keys > 0).astype(np.uint8)
            self._band_ink = self._lineless_ink() & (
                1 - cv2.dilate(accidentals, np.ones((3, 3), np.uint8))
            )
        return self._band_ink

    def _dot_candidates(self) -> NDArray:
        """Clean ink without stems either: an up-stem rises right beside the notehead."""
        if self._dot_ink is None:
            if self.masks is None:
                raise ValueError("Reading note values needs the page and its segmentation")
            # Stems and accidentals sit right beside noteheads too: an up-stem rises at
            # the head's right edge, and the next note's accidental may precede it.
            beside = ((self.masks.stems_rest > 0) | (self.masks.clefs_keys > 0)).astype(np.uint8)
            self._dot_ink = self._clean_ink() & (1 - cv2.dilate(beside, np.ones((3, 3), np.uint8)))
        return self._dot_ink

    def _fill(self, group: VisualGroup, unit: float) -> str | None:
        """Hollow or filled, when the head's core and the sidecar's own test agree."""
        ink = self._lineless_ink()
        cx, cy = group.prediction_center
        width, height = group.prediction_notehead_size
        axes = (
            max(1, int(round(width * CORE_SCALE / 2))),
            max(1, int(round(height * CORE_SCALE / 2))),
        )
        x0, y0 = int(cx) - axes[0] - 1, int(cy) - axes[1] - 1
        core = np.zeros((2 * axes[1] + 3, 2 * axes[0] + 3), np.uint8)
        if x0 < 0 or y0 < 0 or x0 + core.shape[1] > ink.shape[1]:
            return None
        if y0 + core.shape[0] > ink.shape[0]:
            return None
        cv2.ellipse(core, (int(cx) - x0, int(cy) - y0), axes, -20, 0, 360, 1, -1)
        window = ink[y0 : y0 + core.shape[0], x0 : x0 + core.shape[1]]
        share = float((window & core).sum() / max(1, core.sum()))
        if share >= FILLED_CORE_INK and not group.is_hollow_notehead:
            # A scan may print a hollow head's hole gray, dark enough to pass for ink;
            # a filled head's core is as dark as its rim.
            if self._core_lighter_than_rim(group, core, x0, y0):
                return None
            return FILLED
        if share <= HOLLOW_CORE_INK and group.is_hollow_notehead:
            return HOLLOW
        return None

    def _core_lighter_than_rim(self, group: VisualGroup, core: NDArray, x0: int, y0: int) -> bool:
        if self.gray is None:
            return False
        height, width = self.gray.shape[:2]
        cx, cy = group.prediction_center
        half_w, half_h = (int(round(size / 2)) for size in group.prediction_notehead_size)
        hx0, hy0 = max(0, int(cx) - half_w), max(0, int(cy) - half_h)
        hx1, hy1 = min(width, int(cx) + half_w + 1), min(height, int(cy) + half_h + 1)
        head = np.zeros((hy1 - hy0, hx1 - hx0), np.uint8)
        cv2.ellipse(head, (int(cx) - hx0, int(cy) - hy0), (half_w, half_h), -20, 0, 360, 1, -1)
        rim = float(np.percentile(self.gray[hy0:hy1, hx0:hx1][head > 0], 10))
        inside = self.gray[y0 : y0 + core.shape[0], x0 : x0 + core.shape[1]][core > 0]
        return float(np.median(inside)) - rim > CORE_CONTRAST

    def _stem_x(self, groups: list[VisualGroup], unit: float) -> tuple[float | None, str]:
        """The x of the one stem the noteheads share, or why there is none to read."""
        xs = []
        for group in groups:
            for contour in group.stem_contours:
                points = [
                    self.transform.source_point_to_prediction((float(x), float(y)))
                    for x, y in contour
                ]
                if points:
                    xs.append(float(np.median([x for x, _ in points])))
        if not xs:
            return None, NO_STEM
        if max(xs) - min(xs) > STEM_SPREAD * unit:
            return None, "stem_unclear"  # two stems: two voices share the noteheads
        stem_x = float(np.median(xs))
        # A barline segmented as a stem passes beside the noteheads, not at their edge.
        attached = any(
            abs(stem_x - group.prediction_center[0])
            <= group.prediction_notehead_size[0] / 2 + STEM_ATTACHMENT * unit
            for group in groups
        )
        return (stem_x, "") if attached else (None, "stem_unclear")

    def _free_end(
        self, groups: list[VisualGroup], stem_x: float, unit: float
    ) -> tuple[float, float] | None:
        """The stem's free end and the notehead nearest it, if one stem leaves the chord,
        on one side. Segmentation often stops a stem short of its beam, so the printed
        stem is traced instead."""
        ys = [group.prediction_center[1] for group in groups]
        top, bottom = min(ys), max(ys)
        up_length = self._stem_length([stem_x], top, True, unit)
        down_length = self._stem_length([stem_x], bottom, False, unit)
        if up_length >= MIN_STEM_LENGTH * unit and down_length < OPPOSITE_STEM_LENGTH * unit:
            up = True
        elif down_length >= MIN_STEM_LENGTH * unit and up_length < OPPOSITE_STEM_LENGTH * unit:
            up = False
        else:
            return None
        # An up-stem rises from a notehead's right edge, a down-stem falls from a left edge;
        # a stem passing a head's other edge is another voice's.
        side = 1 if up else -1
        if not any(
            side * (stem_x - group.prediction_center[0]) >= STEM_SIDE * unit for group in groups
        ):
            return None
        # A stem leaving the chord's other edge the other way belongs to a second voice.
        if up:
            other_x = min(
                g.prediction_center[0] - g.prediction_notehead_size[0] / 2 for g in groups
            )
            other = self._stem_length(_near(other_x, unit), bottom, False, unit)
        else:
            other_x = max(
                g.prediction_center[0] + g.prediction_notehead_size[0] / 2 for g in groups
            )
            other = self._stem_length(_near(other_x, unit), top, True, unit)
        if other >= OPPOSITE_STEM_LENGTH * unit:
            return None
        # The stem runs past every notehead it holds. A head it does not reach, like a
        # whole note stacked on a chord of half notes, belongs to another voice.
        if bottom > top and not self._stem_spans(stem_x, top, bottom):
            return None
        return (top - up_length, top) if up else (bottom + down_length, bottom)

    def _stem_spans(self, stem_x: float, top: float, bottom: float) -> bool:
        ink = self._lineless_ink()
        x = int(round(stem_x))
        column = ink[int(top) : int(bottom) + 1, max(0, x - 1) : x + 2].max(axis=1)
        return bool(column.size) and float(column.mean()) >= STEM_SPAN

    def _stem_length(self, xs: list[float], notehead_y: float, up: bool, unit: float) -> float:
        """How far ink runs on from the notehead along the longest of these columns."""
        ink = self._lineless_ink()
        lengths = [0.0]
        for x in xs:
            end = self._traced_end(ink, x, notehead_y, up=up, unit=unit)
            if end is not None:
                lengths.append(notehead_y - end if up else end - notehead_y)
        return max(lengths)

    @staticmethod
    def _traced_end(
        ink: NDArray, stem_x: float, notehead_y: float, up: bool, unit: float
    ) -> float | None:
        """Follow the stem's ink from the notehead outward to where it ends."""
        height, width = ink.shape
        x = int(round(stem_x))
        if not 1 <= x < width - 1:
            return None
        step = -1 if up else 1
        y = int(notehead_y + step * 0.5 * unit)
        last_ink, misses = None, 0
        while 0 <= y < height and misses <= max(2, int(0.15 * unit)):
            if ink[y, x - 1 : x + 2].any():
                last_ink, misses = y, 0
            else:
                misses += 1
            y += step
        return float(last_ink) if last_ink is not None else None

    def _stemless(self, group: VisualGroup, unit: float) -> bool:
        """No long vertical ink run starts near either side of the notehead, whose width
        segmentation only estimates."""
        ink = self._clean_ink()
        cx, cy = group.prediction_center
        half_width = group.prediction_notehead_size[0] / 2
        height, width = ink.shape
        reach = int(STEMLESS_RUN * unit)
        # The run is measured from the head's edge: the notehead mask hides its start.
        half_height = int(group.prediction_notehead_size[1] / 2)
        slack = max(1, int(round(STEMLESS_EDGE_SLACK * unit)))
        for x in (int(cx - half_width), int(cx + half_width)):
            for x_offset in range(-slack, slack + 1):
                column_x = x + x_offset
                if not 0 <= column_x < width:
                    continue
                for top, bottom in (
                    (int(cy) - half_height - reach, int(cy)),
                    (int(cy), int(cy) + half_height + reach),
                ):
                    column = ink[max(0, top) : min(height, bottom), column_x]
                    longest = max((length for _, length in _runs(column)), default=0)
                    if column.size and longest >= 0.8 * reach:
                        return False
        return True

    def _on_notehead(self, start: int, length: int, x: int) -> bool:
        """Whether a run in the column at x lies on segmentation's notehead mask. Such a run
        may be a head on the stem that the chord's group left out, or a beam's end that
        segmentation labelled as notehead; which one cannot be told."""
        if self.masks is None:
            return False
        on_head = self.masks.notehead[start : start + length, x - 1 : x + 2] > 0
        return float(on_head.max(axis=1).mean()) >= BAND_NOTEHEAD_SHARE

    def _bands(
        self, stem_x: float, free: tuple[float, float], lines: list[float], unit: float
    ) -> int | None:
        """Flags or beams crossing short columns beside the stem's free end, if each
        side's columns agree; the side that shows more of them gives the count."""
        free_end, notehead_y = free
        up = free_end < notehead_y
        ink = self._beam_ink()
        height, width = ink.shape
        reach = min(BAND_WINDOW, abs(free_end - notehead_y) / unit - BAND_WINDOW_NOTEHEAD_CLEARANCE)
        if reach <= 0.5:
            return None
        start = free_end - BAND_BEYOND * unit if up else free_end - reach * unit
        end = free_end + reach * unit if up else free_end + BAND_BEYOND * unit
        top, bottom = max(0, int(start)), min(height, int(end) + 1)
        line_positions = _line_positions(lines, unit)
        counts = []
        for side in (-1, 1):
            side_counts = set()
            for offset in BAND_COLUMN_OFFSETS:
                x = int(round(stem_x + side * offset * unit))
                if not 1 <= x < width - 1:
                    return None
                runs = [
                    (top + run_start, length)
                    for run_start, length in _runs(ink[top:bottom, x - 1 : x + 2].max(axis=1))
                    if length >= BAND_THICKNESS[0] * unit
                    and not (
                        length <= LINE_THICKNESS * unit
                        and _on_a_line(top + run_start + length / 2, line_positions, unit)
                    )
                ]
                # A run this thick is another symbol in the way, which may hide a band; a
                # run lying on a notehead may be no band at all. Either way the stem is
                # not read.
                if any(
                    length > BAND_THICKNESS[1] * unit or self._on_notehead(start, length, x)
                    for start, length in runs
                ):
                    return None
                # Bands hang from the stem. Ink wholly past its end may be a beam the
                # trace fell short of, or a fingering numeral; only an articulation's dot,
                # standing alone, may sit there.
                past_end = [
                    (start, length)
                    for start, length in runs
                    if (start > free_end + 1 if not up else start + length - 1 < free_end - 1)
                ]
                if any(
                    not self._isolated(x - 1, start, 3, length, unit) for start, length in past_end
                ):
                    return None
                side_counts.add(len(runs) - len(past_end))
            if len(side_counts) != 1:
                return None
            counts.append(side_counts.pop())
        return max(counts)

    def _dotted(self, groups: list[VisualGroup], lines: list[float], unit: float) -> bool | None:
        """A dot right of the chord beside any of its heads dots it; clean paper beside
        every head means no dot; anything else leaves the dot unread."""
        right = max(
            group.prediction_center[0] + group.prediction_notehead_size[0] / 2 for group in groups
        )
        verdicts = [
            self._dot_beside(right, group.prediction_center[1], lines, unit) for group in groups
        ]
        if True in verdicts:
            return True
        if all(verdict is False for verdict in verdicts):
            return False
        return None

    def _dot_beside(self, right: float, cy: float, lines: list[float], unit: float) -> bool | None:
        ink = self._dot_candidates()
        height, width = ink.shape
        x0, x1 = int(right + DOT_REACH[0] * unit), int(right + DOT_REACH[1] * unit)
        # A note on a line is dotted in the space above it, or below it in a lower voice.
        y0, y1 = int(cy - 0.8 * unit), int(cy + 0.8 * unit)
        if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
            return None
        window = ink[y0:y1, x0:x1]
        if not window.any():
            return False
        count, _, stats, centroids = cv2.connectedComponentsWithStats(window, connectivity=8)
        line_positions = _line_positions(lines, unit)
        dots = 0
        for label in range(1, count):
            left, top, w, h, area = stats[label]
            if h < DOT_SIZE[0] * unit and w < 2 * DOT_SIZE[1] * unit:
                continue  # a thin leftover of a staff line, or specks
            round_dot = (
                DOT_SIZE[0] * unit <= w <= DOT_SIZE[1] * unit
                and DOT_SIZE[0] * unit <= h <= DOT_SIZE[1] * unit
                and 0.6 <= w / h <= 1.6
                and area >= 0.6 * w * h
            )
            dot_x, dot_y = x0 + centroids[label][0], y0 + centroids[label][1]
            if (
                not round_dot
                or _on_a_line(dot_y, line_positions, unit)
                or not self._isolated(x0 + left, y0 + top, w, h, unit)
                or self._staccato(dot_x, dot_y, right, unit)
                or self._repeat_sign(dot_x, dot_y, unit)
            ):
                return None
            dots += 1
        return dots == 1 if dots <= 1 else None

    def _isolated(self, left: int, top: int, w: int, h: int, unit: float) -> bool:
        """The blob is a whole symbol, not the end of a flag, rest or accidental: in the
        page's ink it touches nothing else."""
        ink = self._lineless_ink()
        margin = int(np.ceil(0.5 * unit))
        x0, y0 = max(0, left - margin), max(0, top - margin)
        x1, y1 = min(ink.shape[1], left + w + margin), min(ink.shape[0], top + h + margin)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink[y0:y1, x0:x1], connectivity=8)
        inside = labels[top - y0 : top - y0 + h, left - x0 : left - x0 + w]
        for label in set(inside[inside > 0].tolist()):
            _, _, label_w, label_h, _ = stats[label]
            if label_w > DOT_SIZE[1] * unit or label_h > DOT_SIZE[1] * unit:
                return False
        return True

    def _staccato(self, dot_x: float, dot_y: float, right: float, unit: float) -> bool:
        """Another notehead right above or below the dot, which then belongs to it."""
        if self.masks is None:
            return False
        x0 = max(int(right) + 1, int(dot_x - STACCATO_HALF_WIDTH * unit))
        x1 = int(dot_x + STACCATO_HALF_WIDTH * unit) + 1
        y0 = max(0, int(dot_y - STACCATO_REACH * unit))
        y1 = int(dot_y + STACCATO_REACH * unit) + 1
        return x1 > x0 and bool((self.masks.notehead[y0:y1, x0:x1] > 0).any())

    def _repeat_sign(self, dot_x: float, dot_y: float, unit: float) -> bool:
        """A barline right after the dot: an end-repeat sign's dots stand there."""
        ink = self._lineless_ink()
        x0, x1 = int(dot_x) + 1, int(dot_x + REPEAT_BARLINE_REACH * unit) + 1
        y0 = max(0, int(dot_y - REPEAT_BARLINE_LENGTH * unit))
        y1 = int(dot_y + REPEAT_BARLINE_LENGTH * unit) + 1
        for x in range(x0, min(x1, ink.shape[1])):
            longest = max((length for _, length in _runs(ink[y0:y1, x])), default=0)
            if longest >= REPEAT_BARLINE_LENGTH * unit:
                return True
        return False
