"""Printed tuplet numerals in brackets, read from the page.

A bracketed tuplet numeral ("3", "5", "6", "7") is a small digit outside the staff in the
gap of a thin bracket whose ends hook toward the notes, as over a quarter-note triplet.
It is found among the ink components of digit size above and below each staff (staff
lines removed), read with OCR, and kept only when the bracket is there. Unbracketed
digits are too often rests, flags, fingerings or text; a beamed group without a bracket
is found from its beam instead (see ``tuplet_repairs``).
"""

from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from homr.model import MultiStaff
from homr.type_definitions import NDArray
from homr.visual_sidecar import time_signatures
from homr.visual_sidecar.coordinate_transform import PredictionCoordinateTransform
from homr.visual_sidecar.time_signatures import StaffLines, TextReader

TUPLET_NUMERALS_VERSION = 1

# Sizes in staff spaces.
BAND = 5.0  # how far from the staff numerals are looked for
BAND_CLEARANCE = 0.2  # and how far from its outer lines they start
NUMERAL_HEIGHT = (0.8, 1.9)
NUMERAL_WIDTH = (0.4, 1.4)
MIN_INK_SHARE = 0.15  # of the digit's box; a thin line or a hairpin is less
BRACKET_REACH = 1.6  # the bracket starts this close to the digit's side
BRACKET_ROWS = 0.6  # within this of the digit's middle
BRACKET_MIN = 0.8  # and runs at least this far
BRACKET_LONGEST = 12.0  # following it stops this far out
BRACKET_DRIFT = 0.12  # its y may change this much per pixel step (a slanted bracket)
BRACKET_GAP = 2  # pixels a bracket line may skip, as on a poor scan
HOOK_MIN = 0.35  # its ends turn toward the notes by at least this much
#: Measure numbers sit within this of the system's left end.
SYSTEM_START = 3.0
TUPLET_DIGITS = {"3": 3, "5": 5, "6": 6, "7": 7}
#: OCR reads music-font digits as these look-alikes, Cyrillic ones among them.
LOOK_ALIKES = str.maketrans(
    {"з": "3", "З": "3", "Э": "3", "б": "6", "S": "5", "s": "5", "G": "6"}  # noqa: RUF001
)


@dataclass(frozen=True)
class Numeral:
    """A bracketed tuplet numeral, in source pixels."""

    system_index: int
    staff_index: int
    x: float
    y: float
    number: int
    #: Above the staff (stems up), or below it.
    above: bool
    #: The x of the bracket's two hooks.
    bracket: tuple[float, float]
    #: The staff space of the staff it was found at.
    space: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "system_index": self.system_index,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "number": self.number,
            "above": self.above,
            "bracket": [round(v, 1) for v in self.bracket],
        }


def _follow(
    ink: NDArray, x: int, y: int, step: int, sp: float, limit: int
) -> tuple[int, int] | None:
    """Follow a thin horizontal line from (x, y) in direction step; its last inked point."""
    height, width = ink.shape
    drift = max(1, int(round(BRACKET_DRIFT * sp)))
    cx, cy, gaps, last = x, y, 0, None
    length = 0
    while 0 <= cx < width and length < limit:
        rows = [
            cy + d
            for d in sorted(range(-drift, drift + 1), key=abs)
            if 0 <= cy + d < height and ink[cy + d, cx]
        ]
        if rows:
            cy, last, gaps = rows[0], (cx, rows[0]), 0
        else:
            gaps += 1
            if gaps > BRACKET_GAP:
                break
        cx += step
        length += 1
    return last


def _hook(ink: NDArray, x: int, y: int, toward: int, sp: float) -> bool:
    """Whether ink runs from (x, y) toward the notes (down when toward > 0) for HOOK_MIN."""
    height, width = ink.shape
    need = int(round(HOOK_MIN * sp))
    for dx in range(-2, 3):
        cx = x + dx
        if not 0 <= cx < width:
            continue
        n = 0
        cy = y
        while (
            0 <= cy < height
            and n <= need
            and any(ink[cy, cx + d] for d in (-1, 0, 1) if 0 <= cx + d < width)
        ):
            cy += toward
            n += 1
        if n > need:
            return True
    return False


def bracket_around(
    ink: NDArray, digit: tuple[int, int, int, int], sp: float, toward: int
) -> tuple[float, float] | None:
    """The x of a bracket's two hooks around a digit box (x0, y0, x1, y1), or None.

    On each side a thin line starts within BRACKET_REACH of the digit, runs outward at
    least BRACKET_MIN, and ends in a hook turning toward the notes (or away from them, as
    some engravings draw it)."""
    x0, y0, x1, y1 = digit
    mid = (y0 + y1) // 2
    reach = int(BRACKET_REACH * sp)
    ends = []
    for side, step in ((x0, -1), (x1, 1)):
        start = None
        for d in range(2, reach):
            x = side + step * d
            if not 0 <= x < ink.shape[1]:
                break
            rows = [
                r
                for r in range(int(mid - BRACKET_ROWS * sp), int(mid + BRACKET_ROWS * sp))
                if 0 <= r < ink.shape[0] and ink[r, x]
            ]
            if rows:
                start = (x, min(rows, key=lambda r: abs(r - mid)))
                break
        if start is None:
            return None
        end = _follow(ink, start[0], start[1], step, sp, int(BRACKET_LONGEST * sp))
        if end is None or abs(end[0] - start[0]) < BRACKET_MIN * sp:
            return None
        if not (_hook(ink, end[0], end[1], toward, sp) or _hook(ink, end[0], end[1], -toward, sp)):
            return None
        ends.append(end[0])
    return (float(min(ends)), float(max(ends)))


def find_numerals(gray: NDArray, staffs: list[StaffLines], read_text: TextReader) -> list[Numeral]:
    """The page's bracketed tuplet numerals, above and below every staff."""
    ink = time_signatures.ink_without_lines(gray, staffs, 0)
    out = []
    seen = set()
    for index, staff in enumerate(staffs):
        sp = staff.space
        x0, x1 = max(0, int(staff.xs[0])), int(staff.xs[-1])
        top, bottom = float(np.min(staff.lines[0])), float(np.max(staff.lines[4]))
        bands = (
            (True, (top - BAND * sp, top - BAND_CLEARANCE * sp)),
            (False, (bottom + BAND_CLEARANCE * sp, bottom + BAND * sp)),
        )
        for above, (band_top, band_bottom) in bands:
            ya, yb = max(0, int(band_top)), min(ink.shape[0], int(band_bottom))
            if yb <= ya:
                continue
            n, _, stats, _ = cv2.connectedComponentsWithStats(ink[ya:yb, x0:x1], connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = (int(v) for v in stats[i])
                if not (
                    NUMERAL_HEIGHT[0] <= h / sp <= NUMERAL_HEIGHT[1]
                    and NUMERAL_WIDTH[0] <= w / sp <= NUMERAL_WIDTH[1]
                ):
                    continue
                if area < MIN_INK_SHARE * w * h:
                    continue
                box = (x + x0, y + ya, x + x0 + w, y + ya + h)
                if box[0] < staff.xs[0] + SYSTEM_START * sp:
                    continue
                # A digit between two staves is in both staves' bands: read it once.
                key = (box[0] // 4, box[1] // 4)
                if key in seen:
                    continue
                seen.add(key)
                text = read_text(time_signatures.crop_box_for_ocr(ink, box, sp))
                text = text.strip().translate(LOOK_ALIKES)
                if text not in TUPLET_DIGITS:
                    continue
                bracket = bracket_around(ink, box, sp, 1 if above else -1)
                if bracket is None:
                    continue
                out.append(
                    Numeral(
                        staff.system_index,
                        index,
                        (box[0] + box[2]) / 2,
                        (box[1] + box[3]) / 2,
                        TUPLET_DIGITS[text],
                        above,
                        bracket,
                        sp,
                    )
                )
    return out


def _read_page(image_path: str, staffs: list[StaffLines], read_text: TextReader) -> list[Numeral]:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Failed to read " + image_path)
    return find_numerals(gray, staffs, read_text)


def read_tuplet_numerals_in_background(
    image_path: str,
    staffs: Sequence[MultiStaff],
    transform: PredictionCoordinateTransform,
    read_text: TextReader | None = None,
) -> "Future[list[Numeral]]":
    """Reads the page's bracketed tuplet numerals while the transformer runs."""
    if read_text is None:
        from homr.visual_sidecar.tempo_marks import (  # noqa: PLC0415 - loads OCR
            _read_line,
        )

        read_text = _read_line
    return time_signatures._executor.submit(
        _read_page, image_path, time_signatures.staff_lines(staffs, transform), read_text
    )
