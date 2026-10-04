"""Printed metronome marks, such as "♩. = 140", read from the page and written to MusicXML.

The transformer reads what is on a staff, never the text above it, so homr's MusicXML
carries no tempo and a player falls back to its own default, usually quarter = 120. That
default distorts more than the speed. A 6/8 bar holds three quarters and a 2/4 bar two,
so a score that prints dotted quarter = 140 and later quarter = 140 to keep its beat
steady plays its 6/8 bars with a beat a third slower than its 2/4 bars.

The reader looks above each system for an "=" in the ink: two short, flat bars, one above
the other, with digits right after them. The note to its left is read from its shape (a
hollow or filled head, a stem, flags and a dot) and only the digits go to OCR. OCR is not
trusted with the note, which it reads as "J", drops, or reduces to its dot, and its text
detector misses marks crowded by an 8va bracket or an accelerando line; read alone, the
digits come back reliably. A mark is written only when its note is read; otherwise the
page keeps no tempo there, as before.

A mark applies from the measure its tempo text starts over. Text that starts between two
measures, over a barline, a clef or a key change, belongs to the measure that follows it.
Where measures lie is known from the notes and printed rests the sidecar linked to
MusicXML.
"""

import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import cv2
import musicxml.xmlelement.xmlelement as mxl
import numpy as np

from homr.model import MultiStaff, Staff
from homr.type_definitions import NDArray
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform

TEMPO_MARKS_VERSION = 1

# What became of a mark. Reading leaves it UNREAD (no note) or UNPLACED; placement makes
# it PLACED in a measure; writing makes a placed mark WRITTEN or NOT_WRITTEN.
UNREAD = "unread"
UNPLACED = "unplaced"
PLACED = "placed"
WRITTEN = "written"
NOT_WRITTEN = "not_written"

#: Bands are scaled so that a staff space is this many pixels. Glyph sizes below are in
#: these pixels, and OCR reads digits of this size (a 300 dpi page) well.
UNIT_PIXELS = 20.0
#: How far above a system's top line marks are searched, in staff spaces, unless the
#: system above comes first.
BAND_HEIGHT = 14.0
#: Marks may begin left of the staff, above the clef or the instrument name.
BAND_LEFT_MARGIN = 4.0
#: A sliver of the top staff is kept, so that a mark touching the top line stays whole.
BAND_STAFF_OVERLAP = 0.5

#: The printed number at the start of what OCR read: "140", "64-68". Punctuation or a
#: word after it may be read too. A range plays at its first number, as notation programs
#: play it.
NUMBER_PATTERN = re.compile(r"^\W?\s*(\d{2,3})(?:\s*[-–~]\s*(\d{2,3}))?(?!\d)")
MIN_BPM = 20
MAX_BPM = 400

#: Length of each beat unit in quarter notes.
BEAT_UNIT_QUARTERS = {
    "whole": Fraction(4),
    "half": Fraction(2),
    "quarter": Fraction(1),
    "eighth": Fraction(1, 2),
    "16th": Fraction(1, 4),
}

# "=" bars in band pixels: a little wider than a staff space at most sizes of text.
EQUALS_BAR_WIDTH = (6, 70)
EQUALS_BAR_MAX_HEIGHT = 9
EQUALS_BAR_SOLIDITY = 0.6
#: Digits are taller than the "=" from its top bar to its bottom bar by about this factor.
DIGIT_TO_EQUALS_HEIGHT = (1.2, 4.0)

# The rest are relative to the height of the first digit after the "=".
#: The first digit starts within this distance of the "="; the next ones follow more
#: closely than the words of the text do.
DIGIT_REACH = 0.9
DIGIT_GAP = 0.4
#: The note sits within this many digit heights left of the "=", dot included.
NOTE_REACH = 2.5
#: A stem makes the note taller than the digits by at least this factor; a whole note
#: is shorter than them.
STEM_NOTE_HEIGHT = 1.15
#: A notehead is about as wide as a digit is tall, never tiny and never a word.
HEAD_WIDTH = (0.45, 1.7)
HEAD_HEIGHT = (0.3, 1.1)
#: A filled head is solid; a hollow one has a hole or much less ink than its box.
HOLLOW_INK = 0.6
HOLE_AREA = 0.04
#: Flags are ink beside the stem's free end; a flag adds at least this share of a head.
FLAG_AREA = 0.2
#: An augmentation dot is small, round-ish and near the head's height.
DOT_SIZE = (0.12, 0.5)
DOT_REACH = 1.2
#: Words of the tempo text before the note ("Allegro", "Presto (") are this far apart at
#: most, and no taller than this; the mark begins where they begin.
WORD_GAP = 0.9
WORD_HEIGHT = 2.2

#: A mark starting this many staff spaces right of a measure's last symbol still
#: applies to that measure.
PLACEMENT_TOLERANCE = 0.5
#: The system a mark was read above is the one whose top lies within this many staff
#: spaces of where reading found it.
SYSTEM_MATCH_TOLERANCE = 2.0


@dataclass(frozen=True)
class SystemBand:
    """A system in source pixels: where marks above it are searched."""

    top: float
    bottom: float
    left: float
    right: float
    unit_size: float


@dataclass(frozen=True)
class NoteReading:
    beat_unit: str
    dotted: bool
    box: tuple[float, float, float, float]


@dataclass
class TempoMark:
    """One "=" with a number after it found above a system, and what became of it."""

    #: What OCR read after the "=".
    text: str
    #: From the start of the tempo text (or the "=", if no note was read) to the last
    #: digit, in source pixels.
    box: tuple[float, float, float, float]
    #: The number as printed, a range kept as such ("64-68").
    per_minute: str
    #: Beats per minute of the printed beat unit; a printed range's first number.
    beats_per_minute: float
    #: The system the mark was found above.
    system: SystemBand
    note: NoteReading | None
    status: str
    reason: str
    measure: int | None = None

    @property
    def quarter_bpm(self) -> float | None:
        """Quarter notes per minute, as MusicXML's sound tempo counts them."""
        if self.note is None:
            return None
        quarters = BEAT_UNIT_QUARTERS[self.note.beat_unit]
        if self.note.dotted:
            quarters = quarters * Fraction(3, 2)
        return round(self.beats_per_minute * float(quarters), 2)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "box": [round(value, 3) for value in self.box],
            "per_minute": self.per_minute,
            "beat_unit": self.note.beat_unit if self.note is not None else None,
            "dotted": self.note.dotted if self.note is not None else None,
            "quarter_bpm": self.quarter_bpm,
            "measure": self.measure,
            "status": self.status,
            "reason": self.reason,
        }


#: Reads the single line of text an image holds.
LineReader = Callable[[NDArray], str]


@dataclass(frozen=True)
class _Component:
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    label: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


class _Ink:
    def __init__(self, gray: NDArray) -> None:
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        threshold = min(max(threshold, 100.0), 200.0)
        mask = (gray < threshold).astype(np.uint8)
        count, self.labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        self.components = [
            _Component(
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
                int(stats[label, cv2.CC_STAT_AREA]),
                label,
            )
            for label in range(1, count)
        ]

    def mask_of(self, component: _Component) -> NDArray:
        region = self.labels[component.y0 : component.y1, component.x0 : component.x1]
        return (region == component.label).astype(np.uint8)


def _equals_signs(components: Sequence[_Component]) -> list[tuple[_Component, _Component]]:
    """Pairs of short flat bars of equal width, one just above the other."""
    bars = [
        c
        for c in components
        if EQUALS_BAR_WIDTH[0] <= c.width <= EQUALS_BAR_WIDTH[1]
        and c.height <= EQUALS_BAR_MAX_HEIGHT
        and c.width >= 1.8 * c.height
        and c.area >= EQUALS_BAR_SOLIDITY * c.width * c.height
    ]
    pairs = []
    for upper in bars:
        for lower in bars:
            width = max(upper.width, lower.width)
            gap = lower.y0 - upper.y1
            if (
                gap >= 0.3 * max(upper.height, lower.height)
                and gap <= 0.6 * width
                and abs(upper.x0 - lower.x0) <= 0.3 * width
                and abs(upper.width - lower.width) <= 0.35 * width
            ):
                pairs.append((upper, lower))
    return pairs


def _digits_after(
    components: Sequence[_Component], upper: _Component, lower: _Component
) -> tuple[list[_Component], float] | None:
    """The glyphs of the number right after an "=", and the height of its first digit.

    The "=" of an equivalence such as "(♩. = ♩)" has a note after it instead, which is
    taller than a digit and does not start level with the "=".
    """
    equals_right = max(upper.x1, lower.x1)
    equals_height = lower.y1 - upper.y0
    equals_middle = (upper.y0 + lower.y1) / 2
    following = sorted(
        (
            c
            for c in components
            if c.x0 >= equals_right
            and DIGIT_TO_EQUALS_HEIGHT[0] * equals_height
            <= c.height
            <= DIGIT_TO_EQUALS_HEIGHT[1] * equals_height
            and c.y0 <= equals_middle <= c.y1
        ),
        key=lambda c: c.x0,
    )
    if not following:
        return None
    first = following[0]
    if first.x0 - equals_right > DIGIT_REACH * first.height or first.width >= 1.2 * first.height:
        return None
    # Further digits, a dash, a full stop, or a digit the scan broke apart: everything up
    # to a word gap that stays within the first digit's height. A parenthesis does not.
    slack = 0.2 * first.height
    beside = sorted(
        (
            c
            for c in components
            if c.x0 > first.x0 and c.y0 >= first.y0 - slack and c.y1 <= first.y1 + slack
        ),
        key=lambda c: c.x0,
    )
    digits = [first]
    right = first.x1
    for c in beside:
        if c.x0 - right > DIGIT_GAP * first.height:
            break
        digits.append(c)
        right = max(right, c.x1)
    return digits, float(first.height)


def classify_note(  # noqa: PLR0911 - each shape test rules the glyph out
    mask: NDArray, digit_height: float
) -> tuple[str, tuple[int, int, int, int]] | None:
    """The value of a lone note glyph, and its head's box within the mask.

    Metronome marks print stems up; a glyph read otherwise is not taken for a note.
    """
    height, width = mask.shape
    runs = np.array([_longest_run(mask[:, x]) for x in range(width)])
    has_stem = height >= STEM_NOTE_HEIGHT * digit_height and runs.max() >= 0.7 * height
    if not has_stem:
        if not HEAD_HEIGHT[0] * digit_height <= height <= HEAD_HEIGHT[1] * digit_height:
            return None
        if not HEAD_WIDTH[0] * digit_height <= width <= HEAD_WIDTH[1] * digit_height:
            return None
        if width < height or not _is_hollow(mask):
            return None
        return "whole", (0, 0, width, height)

    stem_x = int(np.argmax(runs))
    stem_left = stem_right = stem_x
    while stem_left > 0 and runs[stem_left - 1] >= 0.6 * runs[stem_x]:
        stem_left -= 1
    while stem_right < width - 1 and runs[stem_right + 1] >= 0.6 * runs[stem_x]:
        stem_right += 1
    without_stem = mask.copy()
    without_stem[:, max(0, stem_left - 1) : stem_right + 2] = 0
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(without_stem, connectivity=8)
    # (area, left, top, width, height) of each piece beside the stem.
    pieces = [
        (
            int(stats[label, cv2.CC_STAT_AREA]),
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        for label in range(1, count)
    ]
    heads = [piece for piece in pieces if piece[2] + piece[4] / 2 >= 0.55 * height]
    if not heads:
        return None
    head_area, head_left, head_top, head_width, head_height = max(heads)
    if head_left >= stem_left:
        return None
    # The stem hides the part of the head it joins at the right.
    head_box = (
        head_left,
        head_top,
        max(head_left + head_width, stem_right + 1),
        head_top + head_height,
    )
    full_width = head_box[2] - head_box[0]
    if not HEAD_WIDTH[0] * digit_height <= full_width <= HEAD_WIDTH[1] * digit_height:
        return None
    if not HEAD_HEIGHT[0] * digit_height <= head_height <= HEAD_HEIGHT[1] * digit_height:
        return None
    hollow = _is_hollow(mask[head_box[1] : head_box[3], head_box[0] : head_box[2]])
    flags = [
        piece
        for piece in pieces
        if piece[1] > stem_right
        and piece[2] + piece[4] / 2 < 0.55 * height
        and piece[0] >= FLAG_AREA * head_area
    ]
    if hollow:
        return ("half", head_box) if not flags else None
    if not flags:
        return "quarter", head_box
    # Flags stand apart a little right of the stem.
    probe = min(width - 1, stem_right + 1 + max(1, int(0.25 * full_width)))
    column = mask[: int(0.6 * height), probe]
    crossings = int(np.count_nonzero(np.diff(np.concatenate(([0], column, [0]))) == 1))
    if crossings == 1:
        return "eighth", head_box
    if crossings == 2:
        return "16th", head_box
    return None


def _longest_run(column: NDArray) -> int:
    padded = np.concatenate(([0], column.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    if len(edges) == 0:
        return 0
    return int((edges[1::2] - edges[::2]).max())


def _is_hollow(head: NDArray) -> bool:
    if head.size == 0:
        return False
    contours, hierarchy = cv2.findContours(
        head.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is not None:
        holes = [
            cv2.contourArea(contour)
            for contour, info in zip(contours, hierarchy[0], strict=True)
            if info[3] >= 0
        ]
        if holes and max(holes) >= HOLE_AREA * head.size:
            return True
    return float(head.mean()) < HOLLOW_INK


def _read_note(
    ink: _Ink, upper: _Component, lower: _Component, digit_height: float
) -> NoteReading | str:
    """The note left of an "=", or why there is none."""
    equals_left = min(upper.x0, lower.x0)
    equals_middle = (upper.y0 + lower.y1) / 2
    reach = NOTE_REACH * digit_height
    nearby = [
        c
        for c in ink.components
        if c.x1 <= equals_left + 1
        and c.x1 >= equals_left - reach
        and c.y1 >= equals_middle - 2.5 * digit_height
        and c.y0 <= equals_middle + 1.5 * digit_height
    ]
    for note in sorted(nearby, key=lambda c: -c.x1):
        if note.height < HEAD_HEIGHT[0] * digit_height:
            continue  # the dot, or a speck between the note and the "="
        if not note.y0 <= equals_middle <= note.y1:
            continue  # text above or below the mark
        classified = classify_note(ink.mask_of(note), digit_height)
        if classified is None:
            return "glyph_is_not_a_note"
        beat_unit, (_head_left, head_top, _head_right, head_bottom) = classified
        head_height = head_bottom - head_top
        head_middle = note.y0 + (head_top + head_bottom) / 2
        dotted = any(
            DOT_SIZE[0] * digit_height <= max(c.width, c.height) <= DOT_SIZE[1] * digit_height
            and c.width <= 2 * c.height
            and c.height <= 2 * c.width
            and c.x0 >= note.x1 - 1
            and c.x1 <= equals_left
            and abs(c.center_y - head_middle) <= DOT_REACH * head_height
            for c in nearby
        )
        return NoteReading(beat_unit, dotted, (note.x0, note.y0, note.x1, note.y1))
    return "no_note_left_of_equals"


def _line_start(
    components: Sequence[_Component], start: float, digits: Sequence[_Component], height: float
) -> float:
    """Where the line of tempo text that ends in a mark begins, walking left from start.

    Flat dashes are not words: "accel. _ _ _ ♩ = 140" takes effect where the dashes end.
    """
    top = min(c.y0 for c in digits) - 0.6 * height
    bottom = max(c.y1 for c in digits) + 0.4 * height
    words = [
        c
        for c in components
        if c.y1 >= top
        and c.y0 <= bottom
        and c.height <= WORD_HEIGHT * height
        and c.width <= 4 * height
        and not (c.width >= 2.5 * c.height and c.height <= 0.25 * height)
    ]
    left = start
    while True:
        reached = [
            c.x0
            for c in words
            if c.x0 < left and c.x1 <= left + 1 and left - c.x1 <= WORD_GAP * height
        ]
        if not reached:
            return left
        left = min(reached)


def parse_number(text: str) -> tuple[str, float] | None:
    """The printed number of a mark and its beats per minute, from what OCR read."""
    match = NUMBER_PATTERN.match(text)
    if match is None:
        return None
    low = int(match.group(1))
    high = int(match.group(2)) if match.group(2) is not None else low
    if high < low or not MIN_BPM <= low <= MAX_BPM or not MIN_BPM <= high <= MAX_BPM:
        return None
    return (match.group(1) if high == low else f"{low}-{high}"), float(low)


@dataclass(frozen=True)
class _BandMark:
    text: str
    box: tuple[float, float, float, float]
    per_minute: str
    beats_per_minute: float
    note: NoteReading | str


def _band_marks(band: NDArray, read_line: LineReader) -> list[_BandMark]:
    """Each mark in a band, in band pixels."""
    ink = _Ink(band)
    found = []
    for upper, lower in _equals_signs(ink.components):
        digits_found = _digits_after(ink.components, upper, lower)
        if digits_found is None:
            continue
        digits, digit_height = digits_found
        pad = int(0.3 * digit_height)
        top = max(0, min(c.y0 for c in digits) - pad)
        bottom = min(band.shape[0], max(c.y1 for c in digits) + pad)
        left = max(0, digits[0].x0 - pad)
        right = min(band.shape[1], max(c.x1 for c in digits) + pad)
        text = read_line(band[top:bottom, left:right])
        parsed = parse_number(text)
        if parsed is None:
            continue
        note = _read_note(ink, upper, lower, digit_height)
        if isinstance(note, NoteReading):
            start = _line_start(ink.components, note.box[0], digits, digit_height)
            box = (start, min(note.box[1], top), right, bottom)
        else:
            box = (min(upper.x0, lower.x0), top, right, bottom)
        found.append(_BandMark(text, box, parsed[0], parsed[1], note))
    return found


def read_tempo_marks(
    image: NDArray, systems: Sequence[SystemBand], read_line: LineReader
) -> list[TempoMark]:
    """Every metronome mark above the given systems; systems are in image pixels."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marks: list[TempoMark] = []
    ordered = sorted(systems, key=lambda band: band.top)
    for index, system in enumerate(ordered):
        ceiling = ordered[index - 1].bottom if index > 0 else 0.0
        top = int(max(ceiling, system.top - BAND_HEIGHT * system.unit_size, 0.0))
        bottom = int(min(system.top + BAND_STAFF_OVERLAP * system.unit_size, gray.shape[0]))
        left = int(max(system.left - BAND_LEFT_MARGIN * system.unit_size, 0.0))
        right = int(min(system.right + system.unit_size, gray.shape[1]))
        if bottom - top < system.unit_size or right - left < system.unit_size:
            continue
        scale = UNIT_PIXELS / system.unit_size
        band = cv2.resize(
            gray[top:bottom, left:right],
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        for found in _band_marks(band, read_line):
            note = (
                NoteReading(
                    found.note.beat_unit,
                    found.note.dotted,
                    _band_box_to_image(found.note.box, scale, left, top),
                )
                if isinstance(found.note, NoteReading)
                else None
            )
            marks.append(
                TempoMark(
                    text=found.text,
                    box=_band_box_to_image(found.box, scale, left, top),
                    per_minute=found.per_minute,
                    beats_per_minute=found.beats_per_minute,
                    system=system,
                    note=note,
                    status=UNPLACED if note is not None else UNREAD,
                    reason="" if isinstance(found.note, NoteReading) else found.note,
                )
            )
    return marks


def _band_box_to_image(
    box: tuple[float, float, float, float], scale: float, left: int, top: int
) -> tuple[float, float, float, float]:
    return (
        float(box[0] / scale + left),
        float(box[1] / scale + top),
        float(box[2] / scale + left),
        float(box[3] / scale + top),
    )


def system_bands(
    staffs: Sequence[MultiStaff], transform: PredictionCoordinateTransform
) -> list[SystemBand]:
    """Each system's top staff, outer edges and staff space, in source pixels."""
    bands = []
    for system in staffs:
        if not system.staffs:
            continue
        left = min(staff.min_x for staff in system.staffs)
        right = max(staff.max_x for staff in system.staffs)
        top_left = transform.prediction_point_to_source((left, system.staffs[0].min_y))
        bottom_right = transform.prediction_point_to_source((right, system.staffs[-1].max_y))
        unit = float(np.median([_staff_space(staff) for staff in system.staffs]))
        unit_top = transform.prediction_point_to_source((left, system.staffs[0].min_y + unit))
        bands.append(
            SystemBand(
                top=float(top_left[1]),
                bottom=float(bottom_right[1]),
                left=float(top_left[0]),
                right=float(bottom_right[0]),
                unit_size=float(unit_top[1] - top_left[1]),
            )
        )
    return bands


def _staff_space(staff: Staff) -> float:
    """The distance between neighbouring lines of a staff, in prediction pixels.

    Not homr's average_unit_size: that averages every gap between a grand staff's ten
    lines, the wide one between its two staves included.
    """
    return float(np.median([np.median(np.diff(point.y)) for point in staff.grid]))


_reader: Any = None
_reader_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def _read_line(image: NDArray) -> str:
    """OCR of a crop that holds one short line; recognition only."""
    global _reader  # noqa: PLW0603 - one OCR model per process, made on first use
    with _reader_lock:
        if _reader is None:
            from rapidocr import RapidOCR  # noqa: PLC0415 - loads its models on import

            # The crops are a few digits wide: one thread each keeps OCR from competing
            # with the transformer, which runs at the same time.
            _reader = RapidOCR(
                params={
                    "Global.use_det": False,
                    "Global.use_cls": False,
                    "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                }
            )
        result = _reader(image, use_det=False, use_cls=False)
    return "".join(result.txts or ())


def _read_page(image_path: str, bands: list[SystemBand], read_line: LineReader) -> list[TempoMark]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to read " + image_path)
    return read_tempo_marks(image, bands, read_line)


def read_tempo_marks_in_background(
    image_path: str,
    staffs: Sequence[MultiStaff],
    transform: PredictionCoordinateTransform,
    read_line: LineReader = _read_line,
) -> "Future[list[TempoMark]]":
    """Reads the page's marks while the transformer runs."""
    return _executor.submit(_read_page, image_path, system_bands(staffs, transform), read_line)


def measure_at(x: float, positions: Sequence[tuple[int, float]], tolerance: float) -> int | None:
    """The measure tempo text starting at x applies to, given (measure, x) of the linked
    symbols of its system.

    That is the first measure whose last symbol lies at or right of x: the measure under
    the text, or the one after the gap the text starts in. Text right of the system's
    last symbol has no measure.
    """
    last_x: dict[int, float] = {}
    for measure, symbol_x in positions:
        last_x[measure] = max(last_x.get(measure, symbol_x), symbol_x)
    for measure in sorted(last_x):
        if last_x[measure] >= x - tolerance:
            return measure
    return None


def build_tempo_direction(mark: TempoMark) -> mxl.XMLDirection:
    if mark.note is None or mark.quarter_bpm is None:
        raise ValueError("A tempo direction needs the mark's note")
    direction = mxl.XMLDirection(placement="above")
    direction_type = mxl.XMLDirectionType()
    direction.add_child(direction_type)
    metronome = mxl.XMLMetronome()
    direction_type.add_child(metronome)
    metronome.add_child(mxl.XMLBeatUnit(value_=mark.note.beat_unit))
    if mark.note.dotted:
        metronome.add_child(mxl.XMLBeatUnitDot())
    metronome.add_child(mxl.XMLPerMinute(value_=mark.per_minute))
    tempo = mark.quarter_bpm
    direction.add_child(mxl.XMLSound(tempo=int(tempo) if tempo == int(tempo) else tempo))
    return direction


def write_tempo_directions(root: mxl.XMLElement, marks: Sequence[TempoMark]) -> None:
    """Writes each placed mark at the start of its measure of the first part.

    A measure takes one tempo: where two marks share it, the one printed first is written.
    """
    parts = [child for child in root.get_children() if isinstance(child, mxl.XMLPart)]
    measures = (
        {
            child.number: child
            for child in parts[0].get_children()
            if isinstance(child, mxl.XMLMeasure)
        }
        if parts
        else {}
    )
    written: set[int] = set()
    for mark in sorted(marks, key=lambda mark: (mark.measure or 0, mark.box[0])):
        if mark.status != PLACED or mark.measure is None:
            continue
        measure = measures.get(str(mark.measure))
        if measure is None:
            mark.status, mark.reason = NOT_WRITTEN, "measure_not_in_first_part"
            continue
        if mark.measure in written:
            mark.status, mark.reason = NOT_WRITTEN, "measure_has_an_earlier_mark"
            continue
        _insert_at_start(measure, build_tempo_direction(mark))
        written.add(mark.measure)
        mark.status = WRITTEN


def _insert_at_start(measure: mxl.XMLMeasure, element: mxl.XMLElement) -> None:
    """Puts element before the measure's first note, after its leading layout and attributes."""
    children = measure.get_children()
    lead = 0
    while lead < len(children) and isinstance(
        children[lead], (mxl.XMLPrint, mxl.XMLAttributes, mxl.XMLBarline)
    ):
        lead += 1
    for child in children:
        measure.remove(child)
    for child in [*children[:lead], element, *children[lead:]]:
        measure.add_child(child)
