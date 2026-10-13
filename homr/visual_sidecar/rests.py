"""Pixel evidence for the rests the transformer reports.

Segnet has no rest class. Its ``stems_rests`` output marks stems and barlines only, so a
printed rest is left unlabelled by every segmentation class. That is what makes it
findable: a rest is ink that no class explains. The verifier removes everything the
segmentation does explain (staff lines, noteheads, stems, clefs and keys, braces), joins
the leftover ink back together across the staff lines it was cut by, and asks whether a
free-standing blob of the claimed rest's shape sits where the transformer's attention
says the rest is.

This is diagnostic only. A verdict never changes the symbol stream, the MusicXML, or any
note link; it is exported beside them so a consumer can show which rests have no ink.
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from homr import constants
from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Staff
from homr.staff_canvas_transform import StaffCanvasTransform
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

REST_VERIFICATION_VERSION = 1

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
UNVERIFIED = "unverified"

# Shape families. Whole, half and whole-measure rests are short filled blocks on a
# staff line; every shorter rest is a glyph taller than it is wide.
BLOCK = "block"
GLYPH = "glyph"

# All sizes are in staff spaces (the distance between two staff lines).
#: How far the evidence may sit from the attention coordinate. Attention is a soft
#: localisation: on engraved scores it usually lands within a rest width, but in busy
#: bars it drifts by up to three staff spaces.
SEARCH_RADIUS = 3.0
#: Whole-measure rests are centred in their measure while attention stays near where
#: the measure's content begins, so blocks are searched further out. In exchange a
#: block must hang from or sit on a staff line, which beams and text rarely do.
BLOCK_SEARCH_RADIUS = 6.0
BLOCK_LINE_TOLERANCE = 0.2
#: How far outside the five lines a rest may be displaced: multi-voice writing moves a
#: second voice's rests well below or above the staff. Ink nearer the other staff of a
#: grand staff belongs to that staff.
BAND_MARGIN = 5.0
#: A rest of the other shape family (a block read as a dotted quarter, say) still shows
#: that a rest is printed here, but the expected family wins when both are near. Such a
#: fallback is searched only as far as a glyph would be: a short beam stub looks like a
#: block, and the wide block radius would let one support an invented rest.
FAMILY_MISMATCH_PENALTY = 1.0
#: Inside the staff-line mask, ink survives only as part of a vertical run longer than
#: this: a staff line is a short run, a rest crossing it is a long one. homr thickens
#: the staff mask, so subtracting the mask outright would cut thin rests apart.
LINE_THICKNESS = 0.2
#: Leftover ink in horizontal runs at least this long is a beam (or what is left of a
#: staff line), not part of a rest: the widest rest, a whole or half rest block, spans
#: up to about 1.7 staff spaces in some engravings. Rests printed between beamed notes
#: touch the beam with their tail, so the beam is cut away before the ink is split into
#: blobs.
BEAM_MIN_LENGTH = 2.0
#: Leftover ink is closed vertically by this much to rejoin antialiasing breaks.
BRIDGE_HEIGHT = 0.15
#: Segmentation classes are widened by this much before their ink counts as explained,
#: so antialiased edges of stems and noteheads do not survive as fragments.
EXPLAINED_MARGIN = 0.1
MIN_AREA = 0.08
BLOCK_WIDTH = (0.6, 2.4)
BLOCK_HEIGHT = (0.15, 1.0)
BLOCK_MIN_FILL = 0.5
GLYPH_WIDTH = (0.3, 2.2)
GLYPH_HEIGHT = (0.9, 4.8)


@dataclass(frozen=True)
class SegmentationMasks:
    """The segnet classes, in prediction space, whose ink the verifier treats as explained."""

    staff: NDArray
    stems_rest: NDArray
    notehead: NDArray
    clefs_keys: NDArray
    symbols: NDArray


@dataclass(frozen=True)
class InkBlob:
    """One free-standing piece of unexplained ink, in prediction coordinates."""

    left: int
    top: int
    right: int
    bottom: int
    area: int
    touches_stem: bool

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2

    @property
    def fill(self) -> float:
        return self.area / max(1, self.width * self.height)


@dataclass
class RestVerdict:
    symbol_id: int
    staff_group_index: int
    staff_index: int
    duration: str
    status: str
    reason: str
    #: The rest's location in prediction space: the matched ink when supported,
    #: otherwise the attention x on the staff's middle line.
    center: tuple[float, float] | None = None
    unit_size: float | None = None
    staff_lines: list[float] = field(default_factory=list)
    evidence: InkBlob | None = None
    #: The symbol had no attention coordinate; its x was taken midway between the
    #: nearest symbols before and after it in reading order.
    position_estimated: bool = False


def shape_family(rhythm: str) -> str:
    """Whole (``rest_1``), half (``rest_2``) and whole-measure (``rest_0``) rests are blocks."""
    base = rhythm.split("_", 1)[1] if "_" in rhythm else ""
    digits = "".join(character for character in base if character.isdigit())
    return BLOCK if digits in ("0", "1", "2") else GLYPH


def is_rest(symbol: EncodedSymbol) -> bool:
    return symbol.rhythm.startswith("rest")


def search_radius(family: str) -> float:
    return BLOCK_SEARCH_RADIUS if family == BLOCK else SEARCH_RADIUS


def is_pitchless_note(symbol: EncodedSymbol) -> bool:
    """A note the transformer read without a pitch, which MusicXML generation writes as a rest."""
    return symbol.rhythm.startswith("note") and symbol.pitch in (".", "_")


def _canvas_point(symbol: EncodedSymbol) -> tuple[float, float] | None:
    """The symbol's attention coordinate on the transformer's canvas, if it has one."""
    if symbol.coordinates is None:
        return None
    try:
        values = np.asarray(symbol.coordinates, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(values) < 2 or not np.all(np.isfinite(values[:2])):
        return None
    return float(values[0]), float(values[1])


def _neighbour_estimate(
    symbols: list[EncodedSymbol],
    index: int,
    canvas_points: dict[int, tuple[float, float] | None],
) -> tuple[float, float] | None:
    """Midway between the nearest located symbols before and after ``index``.

    Symbols come in reading order, so a symbol without attention lies between them.
    """
    before = next(
        (canvas_points[id(s)] for s in reversed(symbols[:index]) if canvas_points[id(s)]),
        None,
    )
    after = next((canvas_points[id(s)] for s in symbols[index + 1 :] if canvas_points[id(s)]), None)
    if before is None or after is None:
        return before or after
    return (before[0] + after[0]) / 2, (before[1] + after[1]) / 2


def _middle_line_at(staff: Staff, x: float, staff_index: int) -> float | None:
    line = staff_index * constants.number_of_lines_on_a_staff + 2
    points = [point for point in staff.grid if len(point.y) > line]
    if not points:
        return None
    return float(min(points, key=lambda point: abs(point.x - x)).y[line])


def matches_shape(blob: InkBlob, family: str, unit_size: float, staff_lines: list[float]) -> bool:
    width = blob.width / unit_size
    height = blob.height / unit_size
    if family == BLOCK:
        # Lines continue above and below the staff as ledger lines for displaced rests.
        first, last = staff_lines[0], staff_lines[-1]
        lines = [first - step * unit_size for step in (2, 1)] + staff_lines
        lines += [last + step * unit_size for step in (1, 2)]
        tolerance = BLOCK_LINE_TOLERANCE * unit_size
        on_a_line = any(
            min(abs(blob.top - line), abs(blob.bottom - line)) <= tolerance for line in lines
        )
        return (
            BLOCK_WIDTH[0] <= width <= BLOCK_WIDTH[1]
            and BLOCK_HEIGHT[0] <= height <= BLOCK_HEIGHT[1]
            and blob.fill >= BLOCK_MIN_FILL
            and on_a_line
        )
    return (
        GLYPH_WIDTH[0] <= width <= GLYPH_WIDTH[1]
        and GLYPH_HEIGHT[0] <= height <= GLYPH_HEIGHT[1]
        and not blob.touches_stem
    )


def _ink_threshold(gray: NDArray) -> float:
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.clip(threshold, 80, 200))


def _dilate(mask: NDArray, radius: int) -> NDArray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)


@dataclass(frozen=True)
class ResidualInk:
    """The stages of finding unexplained ink in one region, kept for inspection."""

    ink: NDArray
    explained: NDArray
    staff_lines: NDArray
    beams: NDArray
    residual: NDArray
    bridged: NDArray
    margin: int


def residual_ink(
    gray: NDArray,
    masks: SegmentationMasks,
    note_stems: NDArray,
    region: tuple[int, int, int, int],
    unit_size: float,
) -> ResidualInk:
    """Ink in ``region`` that no segmentation class explains, rejoined across staff lines.

    The ``stems_rests`` class is not taken as explanation: on scanned pages segnet marks
    rests with it too. Only the stems of detected noteheads are, and the other members of
    the class (barlines, stray stems) are too thin or too tall to pass for a rest.
    """
    left, top, right, bottom = region
    crop = gray[top:bottom, left:right]
    ink = (crop < _ink_threshold(crop)).astype(np.uint8)

    def window(mask: NDArray) -> NDArray:
        return (mask[top:bottom, left:right] > 0).astype(np.uint8)

    margin = max(1, int(round(EXPLAINED_MARGIN * unit_size)))
    explained = _dilate(
        window(note_stems)
        | window(masks.notehead)
        | window(masks.clefs_keys)
        | window(masks.symbols),
        margin,
    )
    run_length = max(3, int(round(LINE_THICKNESS * unit_size))) + 1
    crossing = (
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((run_length, 1), np.uint8)) > 0
    ).astype(np.uint8)
    staff_lines = window(masks.staff) & (1 - crossing)
    residual = ink & (1 - explained) & (1 - staff_lines)
    beam_length = max(3, int(round(BEAM_MIN_LENGTH * unit_size)))
    beams = cv2.dilate(
        cv2.morphologyEx(residual, cv2.MORPH_OPEN, np.ones((1, beam_length), np.uint8)),
        np.ones((3, 3), np.uint8),
    )
    residual = residual & (1 - (beams > 0).astype(np.uint8))
    bridge_height = max(3, int(round(BRIDGE_HEIGHT * unit_size)) | 1)
    bridged = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, np.ones((bridge_height, 1), np.uint8))
    return ResidualInk(ink, explained, staff_lines, beams, residual, bridged, margin)


def extract_ink_blobs(
    gray: NDArray,
    masks: SegmentationMasks,
    note_stems: NDArray,
    region: tuple[int, int, int, int],
    unit_size: float,
) -> list[InkBlob]:
    """Find free-standing unexplained ink inside ``region`` (left, top, right, bottom).

    ``note_stems`` marks the stems that belong to detected noteheads. A blob touching
    one is a flag or a beam rather than a rest. Barlines share the stem class but are
    not note stems, so a rest printed right after a barline still counts as free.
    """
    left, top, right, bottom = region
    if right - left < 2 or bottom - top < 2:
        return []
    stages = residual_ink(gray, masks, note_stems, region, unit_size)
    residual, bridged, margin = stages.residual, stages.bridged, stages.margin
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    stem_ring = _dilate(
        (note_stems[top:bottom, left:right] > 0).astype(np.uint8), margin + 2
    ).astype(bool)
    touching = set(np.unique(labels[stem_ring & (bridged > 0)]).tolist())
    min_area = MIN_AREA * unit_size * unit_size
    blobs: list[InkBlob] = []
    for label in range(1, count):
        x, y, width, height, _ = stats[label]
        component = labels[y : y + height, x : x + width] == label
        area = int(np.count_nonzero(component & (residual[y : y + height, x : x + width] > 0)))
        if area < min_area:
            continue
        blobs.append(
            InkBlob(
                left=left + int(x),
                top=top + int(y),
                right=left + int(x + width),
                bottom=top + int(y + height),
                area=area,
                touches_stem=label in touching,
            )
        )
    return blobs


class RestVerifier:
    """Checks each rest token of a staff against the unexplained ink of that staff."""

    def __init__(
        self,
        image: NDArray | None,
        masks: SegmentationMasks | None,
        staff_lines_at_x: Any,
        note_stems: list[RotatedBoundingBox] | None = None,
    ) -> None:
        self.gray = self._grayscale(image)
        self.masks = masks
        self.note_stems = self._rasterize(note_stems or [], self.gray)
        # ``(staff, x, staff_index) -> five line ys``; the recovery stage's robust fit.
        self.staff_lines_at_x = staff_lines_at_x

    @staticmethod
    def _rasterize(stems: list[RotatedBoundingBox], gray: NDArray | None) -> NDArray | None:
        if gray is None:
            return None
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        for stem in stems:
            contour = np.asarray(stem.contours, dtype=np.int32).reshape(-1, 1, 2)
            if len(contour) > 0:
                cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        return mask

    @staticmethod
    def _grayscale(image: NDArray | None) -> NDArray | None:
        if image is None:
            return None
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def verify_staff(
        self,
        symbols: list[EncodedSymbol],
        staff_group_index: int,
        staff: Staff | None,
        canvas_transform: StaffCanvasTransform | None,
    ) -> list[RestVerdict]:
        rests = [symbol for symbol in symbols if is_rest(symbol) or is_pitchless_note(symbol)]
        if not rests:
            return []
        if self.gray is None or self.masks is None:
            return [
                self._unverified(rest, staff_group_index, 0, "no_segmentation") for rest in rests
            ]
        if staff is None or canvas_transform is None:
            return [
                self._unverified(rest, staff_group_index, 0, "no_staff_geometry") for rest in rests
            ]

        physical_staffs = max(1, len(staff.grid[0].y) // constants.number_of_lines_on_a_staff)
        canvas_points = {id(symbol): _canvas_point(symbol) for symbol in symbols}
        # Symbols compare by value, so positions are looked up by identity.
        reading_order = {id(symbol): index for index, symbol in enumerate(symbols)}
        verdicts: list[RestVerdict] = []
        located: list[tuple[RestVerdict, float]] = []
        for rest in rests:
            staff_index = 1 if rest.position == "lower" and physical_staffs > 1 else 0
            if rest.rhythm.endswith("m"):
                verdicts.append(
                    self._unverified(rest, staff_group_index, staff_index, "multi_measure_rest")
                )
                continue
            canvas_point = canvas_points[id(rest)]
            estimated = canvas_point is None
            if canvas_point is None:
                canvas_point = _neighbour_estimate(symbols, reading_order[id(rest)], canvas_points)
            point = (
                canvas_transform.to_prediction(canvas_point) if canvas_point is not None else None
            )
            if point is None:
                verdicts.append(
                    self._unverified(
                        rest, staff_group_index, staff_index, "no_attention_coordinates"
                    )
                )
                continue
            try:
                lines = [float(y) for y in self.staff_lines_at_x(staff, point[0], staff_index)]
            except ValueError:
                verdicts.append(
                    self._unverified(rest, staff_group_index, staff_index, "no_staff_lines")
                )
                continue
            unit_size = float(np.median(np.diff(lines)))
            verdict = RestVerdict(
                symbol_id=rest.visual_match_id,
                staff_group_index=staff_group_index,
                staff_index=staff_index,
                duration=rest.rhythm,
                status=UNSUPPORTED,
                # A pitchless note is a note the transformer saw: whatever ink is there,
                # the rest it becomes in MusicXML is not printed.
                reason="note_without_pitch" if is_pitchless_note(rest) else "no_rest_shaped_ink",
                center=(point[0], lines[2]),
                unit_size=unit_size,
                staff_lines=lines,
                position_estimated=estimated,
            )
            verdicts.append(verdict)
            if is_rest(rest):
                located.append((verdict, point[0]))

        for staff_index in sorted({verdict.staff_index for verdict, _ in located}):
            in_staff = [
                (verdict, x) for verdict, x in located if verdict.staff_index == staff_index
            ]
            blobs = self._blobs_for_staff(staff, staff_index, in_staff)
            owned = [
                blob
                for blob in blobs
                if self._nearest_staff(staff, blob, physical_staffs) == staff_index
            ]
            self._assign(in_staff, owned)
        return verdicts

    @staticmethod
    def _nearest_staff(staff: Staff, blob: InkBlob, physical_staffs: int) -> int:
        """The physical staff whose middle line is nearest the blob's centre."""
        center_x, center_y = blob.center
        distances = []
        for index in range(physical_staffs):
            middle = _middle_line_at(staff, center_x, index)
            distances.append(abs(center_y - middle) if middle is not None else float("inf"))
        return int(np.argmin(distances))

    def _blobs_for_staff(
        self,
        staff: Staff,
        staff_index: int,
        located: list[tuple[RestVerdict, float]],
    ) -> list[InkBlob]:
        if self.gray is None or self.masks is None or self.note_stems is None:
            return []
        unit_size = float(np.median([verdict.unit_size for verdict, _ in located]))
        tops = [verdict.staff_lines[0] for verdict, _ in located]
        bottoms = [verdict.staff_lines[-1] for verdict, _ in located]
        xs = [x for _, x in located]
        height, width = self.gray.shape[:2]
        reach = (BLOCK_SEARCH_RADIUS + BLOCK_WIDTH[1]) * unit_size
        region = (
            max(0, int(min(xs) - reach)),
            max(0, int(min(tops) - BAND_MARGIN * unit_size - GLYPH_HEIGHT[1] * unit_size)),
            min(width, int(max(xs) + reach) + 1),
            min(
                height,
                int(max(bottoms) + BAND_MARGIN * unit_size + GLYPH_HEIGHT[1] * unit_size) + 1,
            ),
        )
        return extract_ink_blobs(self.gray, self.masks, self.note_stems, region, unit_size)

    @staticmethod
    def _assign(located: list[tuple[RestVerdict, float]], blobs: list[InkBlob]) -> None:
        pairs: list[tuple[float, int, int]] = []
        for rest_index, (verdict, x) in enumerate(located):
            unit_size = verdict.unit_size
            if unit_size is None:
                continue
            band_top = verdict.staff_lines[0] - BAND_MARGIN * unit_size
            band_bottom = verdict.staff_lines[-1] + BAND_MARGIN * unit_size
            family = shape_family(verdict.duration)
            other_family = GLYPH if family == BLOCK else BLOCK
            for blob_index, blob in enumerate(blobs):
                center_x, center_y = blob.center
                if not band_top <= center_y <= band_bottom:
                    continue
                distance = abs(center_x - x) / unit_size
                if distance <= search_radius(family) and matches_shape(
                    blob, family, unit_size, verdict.staff_lines
                ):
                    pairs.append((distance, rest_index, blob_index))
                elif distance <= SEARCH_RADIUS and matches_shape(
                    blob, other_family, unit_size, verdict.staff_lines
                ):
                    pairs.append((distance + FAMILY_MISMATCH_PENALTY, rest_index, blob_index))
        claimed_rests: set[int] = set()
        claimed_blobs: set[int] = set()
        for _, rest_index, blob_index in sorted(pairs):
            if rest_index in claimed_rests or blob_index in claimed_blobs:
                continue
            claimed_rests.add(rest_index)
            claimed_blobs.add(blob_index)
            verdict, _ = located[rest_index]
            blob = blobs[blob_index]
            verdict.status = SUPPORTED
            verdict.reason = "rest_shaped_ink"
            verdict.evidence = blob
            verdict.center = blob.center
        for rest_index, (verdict, _) in enumerate(located):
            if rest_index not in claimed_rests and any(pair[1] == rest_index for pair in pairs):
                # Its ink was the closer match for another rest of the same staff.
                verdict.reason = "ink_claimed_by_another_rest"

    @staticmethod
    def _unverified(
        rest: EncodedSymbol, staff_group_index: int, staff_index: int, reason: str
    ) -> RestVerdict:
        return RestVerdict(
            symbol_id=rest.visual_match_id,
            staff_group_index=staff_group_index,
            staff_index=staff_index,
            duration=rest.rhythm,
            status=UNVERIFIED,
            reason=reason,
        )
