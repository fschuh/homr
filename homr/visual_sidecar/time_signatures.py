"""Printed time signatures, read from the page so the writer need not guess the top number.

The transformer's time-signature token carries only the bottom number
(``timeSignature/4``), and the writer infers the top one from the typical bar length of
the bars the signature governs. Misread note values, above all missed tuplets, make bars
too long, and the inferred meter goes wrong with them: 4/4 written as 3/2.

A printed time signature sits inside the staff: the top number fills the space from the
top line to the middle line, the bottom number the space from the middle line to the
bottom line, stacked on one x. Common time (C) and cut time (a C with a vertical stroke)
are one glyph centred on the middle line. Segnet labels none of these, so they are found
in the page's ink with the staff lines taken out and read with OCR one glyph at a time.

A read is used only when it can be trusted:

- its numbers are a meter, a top of 1 to 32 over a power of two;
- every read of it agrees: each staff of its system that reads it, read with the staff
  lines removed at their exact thickness and again with a pixel of slack for scans;
- the transformer put a time-signature token with that bottom number at that place in
  the system (common time is 4, cut time 2). Reads the transformer did not see at all are
  not used.

``TimeSignatureLinks`` ties each token to the read that lands on it and records what
became of it. The writer uses the read meter for the token's section; how it treats the
tokens no read lands on, and the meter carried from the previous page, is described in
``timing_repairs``.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from homr.model import MultiStaff
from homr.type_definitions import NDArray
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform

TIME_SIGNATURES_VERSION = 1

# All sizes are in staff spaces (the median gap between neighbouring lines).
EDGE = 0.4  # how far a glyph's top or bottom may sit from its staff line
GLYPH_HEIGHT = (1.5, 2.5)
GLYPH_WIDTH = (0.6, 3.2)
CLUSTER_GAP = 0.25  # pieces of one glyph, or digits of one number, lie closer than this
PAIR_HEIGHT = (3.4, 4.8)  # a top and bottom number that touch
PAIR_OFFSET = 0.6  # centres of the top and bottom numbers differ by less than this
SYMBOL_HEIGHT = (1.6, 2.6)  # common time
CUT_HEIGHT = (2.6, 4.2)  # cut time: the stroke reaches past the C
SYMBOL_CENTER = 0.35  # a symbol's centre lies this close to the middle line
SYMBOL_WIDTH = (1.0, 2.4)
LINE_RUN = 0.35  # a vertical run this short on a staff line is the line itself
LINE_SNAP = 0.2  # how far the drawn line may lie from homr's line
MIN_COMPONENT_AREA = 12  # pixels; smaller specks are noise
OCR_HEIGHT = 64  # crops are scaled to this height before OCR
OCR_MARGIN = 0.4  # paper added around a crop before OCR
STROKE_SPAN = 0.85  # cut time's stroke spans this share of the glyph's height
STROKE_WIDTH = 0.3  # and is no wider than this
#: Scanned staff lines vary in thickness: the page is read again with lines up to this
#: many pixels thicker taken out, and a signature counts only when every read agrees.
SLACKS = (0, 1)
#: Reads on a system's staves, or in two passes, this close are one signature.
SAME_SIGNATURE = 2.0
#: A token is tied to the read nearest to it within this many staff spaces, and within
#: this many staff spaces of its system's top and bottom lines.
TOKEN_REACH = 6.0
SYSTEM_MARGIN = 2.0
MAX_BEATS = 32
BEAT_TYPES = (1, 2, 4, 8, 16, 32)

# What became of a time-signature token.
READ = "read"
#: Nothing shaped like a time signature is printed near the token.
NO_READ = "no_read"
#: Something shaped like a time signature is printed there, but no trusted read of it.
UNREAD_GLYPH = "unread_glyph"
#: A trusted read lands on the token, with another bottom number.
BEAT_TYPE_DIFFERS = "beat_type_differs"

TextReader = Callable[[NDArray], str]
Meter = tuple[int, int]


@dataclass
class StaffLines:
    system_index: int
    xs: NDArray
    lines: list[NDArray]  # y of each of the five lines at xs, top first
    space: float
    #: The top and bottom of the system the staff belongs to.
    system_span: tuple[float, float] = (0.0, 0.0)

    def line_y(self, k: int, x: float) -> float:
        return float(np.interp(x, self.xs, self.lines[k]))


@dataclass
class _Glyph:
    x0: int
    y0: int
    x1: int
    y1: int
    parts: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass(frozen=True)
class TimeSignatureRead:
    """One printed signature on a system, in source pixels."""

    system_index: int
    x: float
    space: float
    #: The top line of the system's first staff and the bottom line of its last.
    span: tuple[float, float]
    beats: int
    beat_type: int
    symbol: str | None  # "common", "cut", or None for numbers

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "system_index": self.system_index,
            "x": round(self.x, 1),
            "span": [round(v, 1) for v in self.span],
            "beats": self.beats,
            "beat_type": self.beat_type,
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class Glyph:
    """Where something shaped like a time signature is printed, read or not."""

    x: float
    space: float
    span: tuple[float, float]


@dataclass
class TokenRecord:
    """What became of one time-signature token."""

    point: tuple[float, float] | None
    beat_type: int
    status: str
    read: TimeSignatureRead | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "point": None if self.point is None else [round(v, 1) for v in self.point],
            "beat_type": self.beat_type,
            "status": self.status,
            "read": None if self.read is None else self.read.to_json_dict(),
        }


def is_meter(value: object) -> bool:
    """Whether a value is a meter a time signature can print: (beats, beat type)."""
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    beats, beat_type = value
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        return False
    return 1 <= beats <= MAX_BEATS and beat_type in BEAT_TYPES


def validate_meter_options(
    read_time_signatures: bool,
    repair_tuplets: bool,
    time_signature_from_previous_page: object,
) -> None:
    """Rejects writer options that cannot work together: the tuplet repair measures bars
    against the meters read on the page, and a carried meter is only trusted beside them."""
    if repair_tuplets and not read_time_signatures:
        raise ValueError("Repairing tuplets needs the page's time signatures read")
    if time_signature_from_previous_page is None:
        return
    if not read_time_signatures:
        raise ValueError("A time signature from the previous page needs the page's read")
    if not is_meter(time_signature_from_previous_page):
        raise ValueError(
            f"Not a time signature: {time_signature_from_previous_page!r}; expected "
            f"(beats, beat type) with 1 to {MAX_BEATS} beats of 1, 2, 4, 8, 16 or 32"
        )


def staff_lines(
    staffs: Sequence[MultiStaff], transform: PredictionCoordinateTransform
) -> list[StaffLines]:
    """Every staff's five lines in source pixels, with the system it belongs to."""
    out: list[StaffLines] = []
    for system_index, system in enumerate(staffs):
        system_start = len(out)
        for staff in system.staffs:
            points = []
            for p in staff.grid:
                ys = [
                    float(transform.prediction_point_to_source((float(p.x), float(y)))[1])
                    for y in p.y
                ]
                x = float(transform.prediction_point_to_source((float(p.x), float(p.y[0])))[0])
                points.append((x, ys))
            if not points:
                continue
            points.sort(key=lambda point: point[0])
            xs = np.asarray([x for x, _ in points], dtype=float)
            ys_by_x = np.asarray([y for _, y in points], dtype=float)
            for first in range(0, ys_by_x.shape[1] - 4, 5):
                lines = [ys_by_x[:, first + k] for k in range(5)]
                space = float(np.median(np.diff(np.stack(lines), axis=0)))
                if space > 0:
                    out.append(StaffLines(system_index, xs, lines, space))
        if len(out) > system_start:
            span = (
                min(float(np.min(s.lines[0])) for s in out[system_start:]),
                max(float(np.max(s.lines[4])) for s in out[system_start:]),
            )
            for s in out[system_start:]:
                s.system_span = span
    return out


def ink_without_lines(gray: NDArray, staffs: list[StaffLines], slack: int = 0) -> NDArray:
    """Binary ink with each staff line removed where nothing crosses it.

    At each column a line's vertical run of ink goes when it is no thicker than the line
    itself (measured per line, plus slack pixels); a thicker run is a glyph touching or
    lying along the line and stays, which keeps the hairline arcs of C. Where a stroke
    crosses the line, with ink right above and right below the run, the run stays too.
    """
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ink.astype(np.uint8)
    out = ink.copy()
    height, width = ink.shape
    for staff in staffs:
        reach = max(1, int(round(LINE_SNAP * staff.space)))
        x0, x1 = max(0, int(staff.xs[0])), min(width, int(staff.xs[-1]) + 1)
        if x1 <= x0:
            continue
        cols = np.arange(x0, x1)
        # Runs longer than this are never removed and never measured: stop counting there.
        cap = int(LINE_RUN * staff.space) + slack + 2
        runs = [_line_runs(ink, staff, k, cols, reach, cap) for k in range(5)]
        for found, tops, bottoms in runs:
            length = bottoms - tops + 1
            short = length <= LINE_RUN * staff.space
            limit = (int(np.median(length[short])) if short.any() else 2) + slack
            # The capped count makes a long run look cap+1 long at most: never removed.
            remove = length <= limit
            _fill(out, tops[remove], bottoms[remove], found[remove], 0)
        c = np.concatenate([found for found, _, _ in runs])
        t = np.concatenate([tops for _, tops, _ in runs])
        b = np.concatenate([bottoms for _, _, bottoms in runs])
        removed = out[t, c] == 0
        above = np.zeros(c.shape, bool)
        below = np.zeros(c.shape, bool)
        for d in (1, 2):
            ok = t - d >= 0
            above[ok] |= ink[(t - d)[ok], c[ok]] > 0
            ok = b + d < height
            below[ok] |= ink[(b + d)[ok], c[ok]] > 0
        keep = removed & above & below
        _fill(out, t[keep], b[keep], c[keep], 1)
    return out


def _line_runs(
    ink: NDArray, staff: StaffLines, k: int, cols: NDArray, reach: int, cap: int
) -> tuple[NDArray, NDArray, NDArray]:
    """The columns where line k has ink near homr's line, with each run's top and bottom,
    counted at most cap pixels each way from the line."""
    height = ink.shape[0]
    predicted = np.rint(np.interp(cols, staff.xs, staff.lines[k])).astype(int)
    y = np.full(cols.shape, -1)
    for d in sorted(range(-reach, reach + 1), key=abs):
        candidate = predicted + d
        ok = (y < 0) & (candidate >= 0) & (candidate < height)
        hit = np.zeros(cols.shape, bool)
        hit[ok] = ink[candidate[ok], cols[ok]] > 0
        y[hit] = candidate[hit]
    has = y >= 0
    found, line_y = cols[has], y[has]
    extents = []
    for direction in (-1, 1):
        extent = np.zeros(found.shape, int)
        going = np.ones(found.shape, bool)
        for step in range(1, cap + 1):
            row = line_y + direction * step
            inside = going & (row >= 0) & (row < height)
            more = np.zeros(found.shape, bool)
            more[inside] = ink[row[inside], found[inside]] > 0
            extent[more] = step
            going = more
        extents.append(extent)
    return found, line_y - extents[0], line_y + extents[1]


def _fill(out: NDArray, tops: NDArray, bottoms: NDArray, cols: NDArray, value: int) -> None:
    """out[top : bottom + 1, col] = value for each run."""
    if len(cols) == 0:
        return
    for offset in range(int((bottoms - tops).max()) + 1):
        rows = tops + offset
        inside = rows <= bottoms
        out[rows[inside], cols[inside]] = value


def _components(ink: NDArray, box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = box
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink[y0:y1, x0:x1], connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= MIN_COMPONENT_AREA:
            out.append((int(x + x0), int(y + y0), int(x + x0 + w), int(y + y0 + h)))
    return out


def _clusters(parts: list[tuple[int, int, int, int]], gap: float) -> list[_Glyph]:
    """Join components within gap of each other: the digits of "12", the pieces of a
    glyph that lost a thin arc with a staff line, and numbers that touch."""
    parent = list(range(len(parts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(parts):
        for j in range(i + 1, len(parts)):
            b = parts[j]
            dx = max(0, a[0] - b[2], b[0] - a[2])
            dy = max(0, a[1] - b[3], b[1] - a[3])
            if dx <= gap and dy <= gap:
                parent[find(i)] = find(j)
    groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for i, p in enumerate(parts):
        groups.setdefault(find(i), []).append(p)
    return [_bounding(ps) for ps in groups.values()]


def _bounding(parts: list[tuple[int, int, int, int]]) -> _Glyph:
    return _Glyph(
        min(p[0] for p in parts),
        min(p[1] for p in parts),
        max(p[2] for p in parts),
        max(p[3] for p in parts),
        parts=parts,
    )


def _clip(ink: NDArray, g: _Glyph, ya: int, yb: int) -> _Glyph | None:
    """The part of a glyph between two rows, with its box tightened to its ink."""
    parts = []
    for p in g.parts:
        y0, y1 = max(p[1], ya), min(p[3], yb)
        if y1 <= y0:
            continue
        rows, cols = np.nonzero(ink[y0:y1, p[0] : p[2]])
        if len(rows):
            parts.append(
                (
                    p[0] + int(cols.min()),
                    y0 + int(rows.min()),
                    p[0] + int(cols.max()) + 1,
                    y0 + int(rows.max()) + 1,
                )
            )
    return _bounding(parts) if parts else None


def _within(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _digit_sized(g: _Glyph | None, sp: float) -> bool:
    return (
        g is not None
        and _within((g.y1 - g.y0) / sp, GLYPH_HEIGHT)
        and _within((g.x1 - g.x0) / sp, GLYPH_WIDTH)
    )


def _candidates(ink: NDArray, staff: StaffLines) -> list[tuple[str, _Glyph, _Glyph | None]]:
    """("numbers", top, bottom) and ("symbol", glyph, None) candidates on one staff."""
    sp = staff.space
    x0, x1 = max(0, int(staff.xs[0])), int(staff.xs[-1])
    top_y = max(0, int(min(staff.lines[0]) - 1.5 * sp))
    bottom_y = min(ink.shape[0], int(max(staff.lines[4]) + 1.5 * sp))
    clusters = _clusters(_components(ink, (x0, top_y, x1, bottom_y)), CLUSTER_GAP * sp)

    def near(y: float, k: int, x: float) -> bool:
        return abs(y - staff.line_y(k, x)) <= EDGE * sp

    out: list[tuple[str, _Glyph, _Glyph | None]] = []
    upper, lower = [], []
    for g in clusters:
        h, w = (g.y1 - g.y0) / sp, (g.x1 - g.x0) / sp
        spans_staff = near(g.y0, 0, g.cx) and near(g.y1, 4, g.cx)
        if spans_staff and _within(h, PAIR_HEIGHT) and _within(w, GLYPH_WIDTH):
            # Numbers that touch: the top one sits on the middle line, the bottom one
            # hangs from it.
            split = int(round(staff.line_y(2, g.cx)))
            top, bottom = _clip(ink, g, g.y0, split), _clip(ink, g, split, g.y1)
            if top is not None and bottom is not None:
                if _digit_sized(top, sp) and _digit_sized(bottom, sp):
                    out.append(("numbers", top, bottom))
        if near(g.y0, 0, g.cx) and near(g.y1, 2, g.cx) and _digit_sized(g, sp):
            upper.append(g)
        if near(g.y0, 2, g.cx) and near(g.y1, 4, g.cx) and _digit_sized(g, sp):
            lower.append(g)
        centred = abs((g.y0 + g.y1) / 2 - staff.line_y(2, g.cx)) <= SYMBOL_CENTER * sp
        symbol_sized = _within(h, SYMBOL_HEIGHT) or _within(h, CUT_HEIGHT)
        if centred and _within(w, SYMBOL_WIDTH) and symbol_sized:
            out.append(("symbol", g, None))
    used: set[int] = set()
    for t in sorted(upper, key=lambda g: g.cx):
        for i, b in enumerate(lower):
            if i not in used and abs(t.cx - b.cx) <= PAIR_OFFSET * sp:
                out.append(("numbers", t, b))
                used.add(i)
                break
    return sorted(out, key=lambda c: c[1].cx)


def _crop(ink: NDArray, g: _Glyph, sp: float) -> NDArray:
    """The glyph's own ink, black on white with a margin, scaled for OCR."""
    mask = np.zeros_like(ink)
    for p in g.parts:
        mask[p[1] : p[3], p[0] : p[2]] = ink[p[1] : p[3], p[0] : p[2]]
    crop = ((1 - mask[g.y0 : g.y1, g.x0 : g.x1]) * 255).astype(np.uint8)
    pad = int(round(OCR_MARGIN * sp))
    padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    scale = OCR_HEIGHT / padded.shape[0]
    scaled = cv2.resize(
        padded, (max(1, int(padded.shape[1] * scale)), OCR_HEIGHT), interpolation=cv2.INTER_AREA
    )
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


def crop_box_for_ocr(ink: NDArray, box: tuple[int, int, int, int], sp: float) -> NDArray:
    """The ink inside a box (x0, y0, x1, y1), black on white with a margin, scaled for OCR."""
    return _crop(ink, _Glyph(*box, parts=[box]), sp)


def _without_stroke(ink: NDArray, g: _Glyph, sp: float) -> tuple[_Glyph, NDArray] | None:
    """Cut time's C with its vertical stroke taken out, so OCR sees a C; None when the
    glyph has no thin stroke spanning nearly its height."""
    region = np.zeros_like(ink[g.y0 : g.y1, g.x0 : g.x1])
    for p in g.parts:
        region[p[1] - g.y0 : p[3] - g.y0, p[0] - g.x0 : p[2] - g.x0] = ink[p[1] : p[3], p[0] : p[2]]
    height = g.y1 - g.y0
    stroke = []
    for c in range(region.shape[1]):
        rows = np.flatnonzero(region[:, c])
        if len(rows) and rows[-1] - rows[0] + 1 >= STROKE_SPAN * height:
            stroke.append(c)
    if not stroke or max(stroke) - min(stroke) + 1 > STROKE_WIDTH * sp:
        return None
    cleaned = ink.copy()
    cleaned[g.y0 : g.y1, g.x0 + min(stroke) : g.x0 + max(stroke) + 1] = 0
    clipped = _clip(cleaned, g, g.y0, g.y1)
    return (clipped, cleaned) if clipped is not None else None


_LOOK_ALIKES = str.maketrans(
    {"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "s": "5", "Z": "2", "z": "2"}
)
_C_READS = {"C", "c", "(", "€", "¢", "C|", "c|", "Ç", "G", "e"}


def _number(text: str) -> int | None:
    # OCR occasionally reads a music-font digit as a look-alike letter.
    text = text.strip().translate(_LOOK_ALIKES)
    return int(text) if text.isdigit() else None


def _read_staff(
    ink: NDArray, staff: StaffLines, read_text: TextReader, glyphs: list[float]
) -> list[tuple[float, Meter, str | None]]:
    """(x, meter, symbol) of every candidate on the staff that OCR reads as a meter.

    The x of every candidate, read or not, is added to glyphs.
    """
    sp = staff.space
    out: list[tuple[float, Meter, str | None]] = []
    for kind, a, b in _candidates(ink, staff):
        glyphs.append(a.cx)
        if kind == "numbers" and b is not None:
            top = _number(read_text(_crop(ink, a, sp)))
            bottom = _number(read_text(_crop(ink, b, sp)))
            if top is not None and bottom is not None:
                out.append((a.cx, (top, bottom), None))
            continue
        glyph, source = a, ink
        cut = _within((a.y1 - a.y0) / sp, CUT_HEIGHT)
        if cut:
            stripped = _without_stroke(ink, a, sp)
            if stripped is None:
                continue
            glyph, source = stripped
        if read_text(_crop(source, glyph, sp)).strip() in _C_READS:
            out.append((a.cx, (2, 2) if cut else (4, 4), "cut" if cut else "common"))
    return out


def read_time_signatures(
    gray: NDArray, staffs: list[StaffLines], read_text: TextReader, glyphs: list[Glyph]
) -> list[TimeSignatureRead]:
    """The page's signatures that every read agrees on and that are meters.

    Every candidate glyph, read or not, is added to glyphs.
    """
    signatures: list[tuple[int, float, StaffLines, list[tuple[Meter, str | None]]]] = []
    for slack in SLACKS:
        ink = ink_without_lines(gray, staffs, slack)
        for staff in staffs:
            xs: list[float] = []
            found = _read_staff(ink, staff, read_text, xs)
            glyphs.extend(Glyph(x, staff.space, staff.system_span) for x in xs)
            for x, meter, symbol in found:
                same = next(
                    (
                        s
                        for s in signatures
                        if s[0] == staff.system_index
                        and abs(s[1] - x) <= SAME_SIGNATURE * staff.space
                    ),
                    None,
                )
                if same is None:
                    signatures.append((staff.system_index, x, staff, [(meter, symbol)]))
                else:
                    same[3].append((meter, symbol))
    out = []
    for system_index, x, staff, reads in signatures:
        meters = {meter for meter, _ in reads}
        if len(meters) != 1:
            continue
        beats, beat_type = meters.pop()
        if 1 <= beats <= MAX_BEATS and beat_type in BEAT_TYPES:
            out.append(
                TimeSignatureRead(
                    system_index, x, staff.space, staff.system_span, beats, beat_type, reads[0][1]
                )
            )
    return sorted(out, key=lambda r: (r.system_index, r.x))


#: Page reads run one at a time, beside the transformer.
_executor = ThreadPoolExecutor(max_workers=1)


def _read_page(
    image_path: str, staffs: list[StaffLines], read_text: TextReader
) -> tuple[list[TimeSignatureRead], list[Glyph]]:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Failed to read " + image_path)
    glyphs: list[Glyph] = []
    return read_time_signatures(gray, staffs, read_text, glyphs), glyphs


def read_time_signatures_in_background(
    image_path: str,
    staffs: Sequence[MultiStaff],
    transform: PredictionCoordinateTransform,
    read_text: TextReader | None = None,
) -> "Future[tuple[list[TimeSignatureRead], list[Glyph]]]":
    """Reads the page's time signatures while the transformer runs."""
    if read_text is None:
        from homr.visual_sidecar.tempo_marks import (  # noqa: PLC0415 - loads OCR
            _read_line,
        )

        read_text = _read_line
    return _executor.submit(_read_page, image_path, staff_lines(staffs, transform), read_text)


class TimeSignatureLinks:
    """Ties the transformer's time-signature tokens to the signatures read on the page,
    and records the meters carried into and out of the page."""

    def __init__(self) -> None:
        #: The meter the page starts in, carried from the previous page, if known.
        self.carried_in: Meter | None = None
        #: The trusted meter in force at the page's end: the carried one or the last read
        #: one; None once a token without a trusted read starts a section.
        self.carried_out: Meter | None = None
        self._reads: Future[tuple[list[TimeSignatureRead], list[Glyph]]] | None = None
        self._tokens: dict[int, tuple[tuple[float, float] | None, int]] = {}
        self._decided: dict[int, TokenRecord] = {}
        self._read_once = False
        self.records: list[TokenRecord] = []
        self.unmatched_reads: list[TimeSignatureRead] = []

    def set_reads(self, reads: "Future[tuple[list[TimeSignatureRead], list[Glyph]]]") -> None:
        self._reads = reads

    @property
    def enabled(self) -> bool:
        """Whether the page's time signatures were read at all."""
        return self._reads is not None

    def add_token(self, symbol_id: int, point: tuple[float, float] | None, beat_type: int) -> None:
        """A token and where the transformer saw it, in source pixels."""
        self._tokens[symbol_id] = (point, beat_type)

    def status_of(self, symbol_id: int) -> str | None:
        """What became of a token, or None for a token this page never added."""
        if symbol_id not in self._decided:
            self._decide()
        record = self._decided.get(symbol_id)
        return record.status if record is not None else None

    def meter_of(self, symbol_id: int) -> Meter | None:
        """The printed meter of a token, when a trusted read lands on it."""
        if symbol_id not in self._decided:
            self._decide()
        record = self._decided.get(symbol_id)
        if record is None or record.status != READ or record.read is None:
            return None
        return record.read.beats, record.read.beat_type

    def _decide(self) -> None:
        """Ties every token not yet decided to the nearest read no token holds yet."""
        if self._reads is None:
            return
        reads, glyphs = self._reads.result(120)
        self._read_once = True
        held = [record.read for record in self._decided.values() if record.read is not None]
        taken = {i for i, read in enumerate(reads) if any(read is h for h in held)}
        # Nearest pairs first, so two tokens never claim one read.
        pairs = []
        for symbol_id, (point, _) in self._tokens.items():
            if point is None or symbol_id in self._decided:
                continue
            for i, read in enumerate(reads):
                margin = SYSTEM_MARGIN * read.space
                in_system = read.span[0] - margin <= point[1] <= read.span[1] + margin
                distance = abs(read.x - point[0]) / read.space
                if in_system and distance <= TOKEN_REACH:
                    pairs.append((distance, symbol_id, i))
        matched: dict[int, int] = {}
        for _, symbol_id, i in sorted(pairs):
            if symbol_id in matched or i in taken:
                continue
            matched[symbol_id] = i
            taken.add(i)
        for symbol_id, (point, beat_type) in self._tokens.items():
            if symbol_id in self._decided:
                continue
            landed = reads[matched[symbol_id]] if symbol_id in matched else None
            if landed is None:
                status = UNREAD_GLYPH if _glyph_near(point, glyphs) else NO_READ
            elif landed.beat_type != beat_type:
                status = BEAT_TYPE_DIFFERS
            else:
                status = READ
            record = TokenRecord(point, beat_type, status, landed)
            self._decided[symbol_id] = record
            self.records.append(record)
        self.unmatched_reads = [r for i, r in enumerate(reads) if i not in taken]

    def to_json_dict(self) -> dict[str, Any]:
        if not self._read_once or any(key not in self._decided for key in self._tokens):
            self._decide()
        return {
            "version": TIME_SIGNATURES_VERSION,
            "carried_in": None if self.carried_in is None else list(self.carried_in),
            "carried_out": None if self.carried_out is None else list(self.carried_out),
            "tokens": [record.to_json_dict() for record in self.records],
            "unmatched_reads": [r.to_json_dict() for r in self.unmatched_reads],
        }


def _glyph_near(point: tuple[float, float] | None, glyphs: list[Glyph]) -> bool:
    """Whether something shaped like a time signature is printed where the token is."""
    if point is None:
        return False
    return any(
        g.span[0] - SYSTEM_MARGIN * g.space <= point[1] <= g.span[1] + SYSTEM_MARGIN * g.space
        and abs(g.x - point[0]) <= TOKEN_REACH * g.space
        for g in glyphs
    )
