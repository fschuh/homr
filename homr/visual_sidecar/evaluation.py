from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any
from xml.etree import ElementTree

from homr.visual_sidecar.models import (
    CROSS_STAFF_ALIGNMENT_METHOD,
    CROSS_STAFF_REPAIR_ACTION,
    VISUAL_SIDECAR_VERSION,
)

SIDECAR_VERSION = VISUAL_SIDECAR_VERSION
REPORT_VERSION = 1
LINKED_VISUAL_STATUSES = {"canonical", "fallback"}
SUPPORTED_VISUAL_STATUSES = {*LINKED_VISUAL_STATUSES, "diagnostic"}
DIATONIC_STEPS = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
CLEF_REFERENCE_PITCHES = {"G": ("G", 4), "F": ("F", 3), "C": ("C", 4)}
ALTER_NAMES = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}


class EvaluationInputError(ValueError):
    """Raised when an artifact cannot be parsed enough to evaluate it."""


class UnsupportedSidecarVersionError(EvaluationInputError):
    """Raised when the evaluator is given anything other than sidecar v3."""


@dataclass(frozen=True)
class Divergence:
    kind: str
    message: str
    musicxml_id: str | None = None
    visual_group_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "message": self.message,
        }
        if self.musicxml_id is not None:
            result["musicxml_id"] = self.musicxml_id
        if self.visual_group_id is not None:
            result["visual_group_id"] = self.visual_group_id
        if self.details:
            result["details"] = self.details
        return result


@dataclass(frozen=True)
class VisualEvalReport:
    counts: dict[str, int]
    divergences: tuple[Divergence, ...]
    diagnostic_visual_group_ids: tuple[str, ...]
    capabilities: dict[str, str] = field(
        default_factory=lambda: {"visual_accidental_check": "not_evaluated"}
    )

    @property
    def passed(self) -> bool:
        return not self.divergences

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": REPORT_VERSION,
            "passed": self.passed,
            "sidecar_version": SIDECAR_VERSION,
            "counts": self.counts,
            "divergences": [divergence.to_dict() for divergence in self.divergences],
            "diagnostics": {
                "visual_group_ids": list(self.diagnostic_visual_group_ids),
            },
            "capabilities": self.capabilities,
        }


@dataclass(frozen=True)
class ClefDefinition:
    sign: str
    line: int
    octave_change: int = 0


@dataclass(frozen=True)
class MusicXmlNote:
    musicxml_id: str | None
    pitch: str
    step: str
    octave: int
    part: int
    measure: int
    musicxml_staff_number: int
    voice: int
    clef: ClefDefinition | None
    active_clefs: dict[int, ClefDefinition | None]
    event_index: int
    is_chord_tone: bool


@dataclass(frozen=True)
class ParsedMusicXml:
    notes: tuple[MusicXmlNote, ...]
    chord_events: tuple[tuple[str, ...], ...]
    non_chord_streams: dict[tuple[int, int, int, int], tuple[str, ...]]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _first_child(element: ElementTree.Element, name: str) -> ElementTree.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _required_text(element: ElementTree.Element, name: str) -> str:
    child = _first_child(element, name)
    if child is None or child.text is None or not child.text.strip():
        raise EvaluationInputError(f"MusicXML element is missing {name}")
    return child.text.strip()


def _optional_int(element: ElementTree.Element, name: str, default: int) -> int:
    child = _first_child(element, name)
    if child is None or child.text is None or not child.text.strip():
        return default
    try:
        return int(child.text.strip())
    except ValueError as error:
        raise EvaluationInputError(f"MusicXML {name} must be an integer") from error


def _parse_pitch(note: ElementTree.Element) -> tuple[str, str, int]:
    pitch = _first_child(note, "pitch")
    if pitch is None:
        raise EvaluationInputError("Pitched MusicXML note has no pitch element")
    step = _required_text(pitch, "step").upper()
    if step not in DIATONIC_STEPS:
        raise EvaluationInputError(f"Unsupported MusicXML pitch step {step!r}")
    try:
        octave = int(_required_text(pitch, "octave"))
    except ValueError as error:
        raise EvaluationInputError("MusicXML octave must be an integer") from error
    alter_text = _first_child(pitch, "alter")
    alter = 0
    if alter_text is not None and alter_text.text is not None and alter_text.text.strip():
        try:
            numeric_alter = float(alter_text.text.strip())
        except ValueError as error:
            raise EvaluationInputError("MusicXML alter must be numeric") from error
        if not numeric_alter.is_integer():
            raise EvaluationInputError("Microtonal MusicXML pitches are not supported")
        alter = int(numeric_alter)
    if alter not in ALTER_NAMES:
        raise EvaluationInputError(f"Unsupported MusicXML alter {alter}")
    return f"{step}{ALTER_NAMES[alter]}{octave}", step, octave


def _parse_clef(clef: ElementTree.Element) -> tuple[int, ClefDefinition | None]:
    try:
        staff_number = int(clef.attrib.get("number", "1"))
    except ValueError as error:
        raise EvaluationInputError("MusicXML clef number must be an integer") from error
    sign_element = _first_child(clef, "sign")
    line_element = _first_child(clef, "line")
    if (
        sign_element is None
        or sign_element.text is None
        or line_element is None
        or line_element.text is None
    ):
        return staff_number, None
    sign = sign_element.text.strip().upper()
    try:
        line = int(line_element.text.strip())
    except ValueError:
        return staff_number, None
    octave_change = _optional_int(clef, "clef-octave-change", 0)
    return staff_number, ClefDefinition(sign, line, octave_change)


def parse_musicxml(musicxml: str) -> ParsedMusicXml:
    try:
        root = ElementTree.fromstring(musicxml)  # noqa: S314 - local HOMR artifact
    except ElementTree.ParseError as error:
        raise EvaluationInputError(f"Invalid MusicXML: {error}") from error

    notes: list[MusicXmlNote] = []
    chord_events: list[tuple[str, ...]] = []
    non_chord_streams: dict[tuple[int, int, int, int], list[str]] = defaultdict(list)
    parts = _children(root, "part")
    for part_index, part in enumerate(parts, start=1):
        active_clefs: dict[int, ClefDefinition | None] = {}
        for measure_index, measure in enumerate(_children(part, "measure"), start=1):
            event_index = 0
            current_event_ids: list[str] = []
            for child in measure:
                child_name = _local_name(child.tag)
                if child_name == "attributes":
                    for clef in _children(child, "clef"):
                        staff_number, definition = _parse_clef(clef)
                        active_clefs[staff_number] = definition
                    continue
                if child_name != "note":
                    continue

                is_chord_tone = _first_child(child, "chord") is not None
                if not is_chord_tone:
                    if len(current_event_ids) > 1:
                        chord_events.append(tuple(current_event_ids))
                    current_event_ids = []
                    event_index += 1

                pitch_element = _first_child(child, "pitch")
                if pitch_element is None:
                    continue
                pitch, step, octave = _parse_pitch(child)
                musicxml_id = child.attrib.get("id")
                staff_number = _optional_int(child, "staff", 1)
                voice = _optional_int(child, "voice", 1)
                note = MusicXmlNote(
                    musicxml_id=musicxml_id,
                    pitch=pitch,
                    step=step,
                    octave=octave,
                    part=part_index,
                    measure=measure_index,
                    musicxml_staff_number=staff_number,
                    voice=voice,
                    clef=active_clefs.get(staff_number),
                    active_clefs=dict(active_clefs),
                    event_index=event_index,
                    is_chord_tone=is_chord_tone,
                )
                notes.append(note)
                if musicxml_id is not None:
                    current_event_ids.append(musicxml_id)
                    if not is_chord_tone:
                        stream = (part_index, measure_index, staff_number, voice)
                        non_chord_streams[stream].append(musicxml_id)
            if len(current_event_ids) > 1:
                chord_events.append(tuple(current_event_ids))

    return ParsedMusicXml(
        notes=tuple(notes),
        chord_events=tuple(chord_events),
        non_chord_streams={key: tuple(value) for key, value in non_chord_streams.items()},
    )


def expected_staff_position(
    note: MusicXmlNote, *, musicxml_staff_number: int | None = None
) -> int | None:
    clef = (
        note.clef if musicxml_staff_number is None else note.active_clefs.get(musicxml_staff_number)
    )
    if clef is None or clef.sign not in CLEF_REFERENCE_PITCHES or not 1 <= clef.line <= 5:
        return None
    reference_step, reference_octave = CLEF_REFERENCE_PITCHES[clef.sign]
    reference_octave += clef.octave_change
    reference_pitch_index = reference_octave * 7 + DIATONIC_STEPS[reference_step]
    note_pitch_index = note.octave * 7 + DIATONIC_STEPS[note.step]
    reference_staff_position = 2 * clef.line - 1
    return reference_staff_position + note_pitch_index - reference_pitch_index


def _mapping(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return value


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _has_notehead_geometry(group: dict[str, Any]) -> bool:
    ellipses = group.get("notehead_ellipses")
    contours = group.get("notehead_contours")
    bbox = group.get("bbox")
    return (
        (
            (isinstance(ellipses, list) and bool(ellipses))
            or (isinstance(contours, list) and bool(contours))
        )
        and isinstance(bbox, list)
        and len(bbox) == 4
    )


def _visual_center(group: dict[str, Any]) -> tuple[float, float] | None:
    center = group.get("center")
    if not isinstance(center, list) or len(center) != 2:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in center):
        return None
    point = (float(center[0]), float(center[1]))
    return point if all(math.isfinite(value) for value in point) else None


def _notehead_size(group: dict[str, Any]) -> tuple[float, float] | None:
    ellipses = group.get("notehead_ellipses")
    if isinstance(ellipses, list) and ellipses:
        ellipse = _mapping(ellipses[0])
        if ellipse is not None:
            rx = ellipse.get("rx")
            ry = ellipse.get("ry")
            if (
                isinstance(rx, (int, float))
                and not isinstance(rx, bool)
                and isinstance(ry, (int, float))
                and not isinstance(ry, bool)
                and math.isfinite(rx)
                and math.isfinite(ry)
                and rx > 0
                and ry > 0
            ):
                return 2 * float(rx), 2 * float(ry)
    bbox = group.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        return None
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        return None
    return width, height


def _pixel_centers_prove_adjacent_staff_position(
    group: dict[str, Any],
    sidecar_note: dict[str, Any],
    expected_position: int,
    *,
    groups_by_id: dict[str, dict[str, Any]],
    groups_by_musicxml_id: dict[str, list[str]],
    sidecar_notes_by_id: dict[str, dict[str, Any]],
    xml_notes_by_id: dict[str, MusicXmlNote],
) -> bool:
    """Recognize one narrow staff-grid rounding false positive from source geometry.

    A skewed final grid sample can round two visibly adjacent notes to the same
    integer staff position. Suppress that mismatch only when structural, canonical
    links on the same physical staff independently establish the local pixel step,
    and a nearby correctly rounded note at the reported position is exactly one
    pixel-space step away in the MusicXML-required direction.
    """
    actual_position = _integer(group.get("staff_position"))
    target_center = _visual_center(group)
    if (
        actual_position is None
        or abs(actual_position - expected_position) != 1
        or target_center is None
        or group.get("visual_status") != "canonical"
        or _string(sidecar_note.get("alignment_method")) != "structural"
    ):
        return False
    staff_group_index = _integer(group.get("staff_group_index"))
    staff_index = _integer(group.get("staff_index"))
    if staff_group_index is None or staff_index is None:
        return False

    anchors: list[tuple[float, float, int, float, float]] = []
    for anchor in groups_by_id.values():
        if (
            anchor is group
            or anchor.get("visual_status") != "canonical"
            or _integer(anchor.get("staff_group_index")) != staff_group_index
            or _integer(anchor.get("staff_index")) != staff_index
        ):
            continue
        musicxml_id = _string(anchor.get("musicxml_id"))
        if musicxml_id is None or groups_by_musicxml_id.get(musicxml_id) != [
            anchor.get("visual_group_id")
        ]:
            continue
        xml_note = xml_notes_by_id.get(musicxml_id)
        anchor_note = sidecar_notes_by_id.get(musicxml_id)
        if (
            xml_note is None
            or anchor_note is None
            or xml_note.musicxml_staff_number - 1 != staff_index
            or _string(anchor_note.get("alignment_method")) != "structural"
        ):
            continue
        anchor_position = _integer(anchor.get("staff_position"))
        anchor_expected = expected_staff_position(xml_note)
        anchor_center = _visual_center(anchor)
        anchor_size = _notehead_size(anchor)
        if (
            anchor_position is None
            or anchor_position != anchor_expected
            or anchor_center is None
            or anchor_size is None
        ):
            continue
        anchors.append(
            (
                anchor_center[0],
                anchor_center[1],
                anchor_position,
                anchor_size[0],
                anchor_size[1],
            )
        )

    target_size = _notehead_size(group)
    if target_size is None or len(anchors) < 3:
        return False
    typical_width = median([target_size[0], *(anchor[3] for anchor in anchors)])
    typical_height = median([target_size[1], *(anchor[4] for anchor in anchors)])
    pair_x_tolerance = 3 * typical_width
    step_candidates: list[float] = []
    for index, first in enumerate(anchors):
        for second in anchors[index + 1 :]:
            position_delta = first[2] - second[2]
            if position_delta == 0 or abs(first[0] - second[0]) > pair_x_tolerance:
                continue
            step = abs((first[1] - second[1]) / position_delta)
            if 0.25 * typical_height <= step <= 1.5 * typical_height:
                step_candidates.append(step)
    if len(step_candidates) < 3:
        return False
    staff_step = median(step_candidates)
    residual_tolerance = max(1.5, 0.25 * staff_step)
    neighbor_x_tolerance = 3 * typical_width
    expected_delta = expected_position - actual_position
    for anchor_x, anchor_y, anchor_position, _width, _height in anchors:
        if (
            anchor_position != actual_position
            or abs(anchor_x - target_center[0]) > neighbor_x_tolerance
        ):
            continue
        expected_y = anchor_y - expected_delta * staff_step
        if (
            abs(target_center[1] - expected_y) <= residual_tolerance
            and 0.65 * staff_step <= abs(target_center[1] - anchor_y) <= 1.35 * staff_step
        ):
            return True
    return False


def _format_member_assignments(members: list[dict[str, str | None]], field: str) -> str:
    return ", ".join(
        f"{member['musicxml_id']}/{member['visual_group_id']}="
        f"{member[field] if member[field] is not None else 'null'}"
        for member in members
    )


def evaluate_musicxml_sidecar(musicxml: str, sidecar: dict[str, Any]) -> VisualEvalReport:
    version = sidecar.get("version")
    if version != SIDECAR_VERSION:
        raise UnsupportedSidecarVersionError(
            f"Visual sidecar version {SIDECAR_VERSION} is required, got {version!r}"
        )
    parsed_xml = parse_musicxml(musicxml)
    divergences: list[Divergence] = []

    def add(
        kind: str,
        message: str,
        *,
        musicxml_id: str | None = None,
        visual_group_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        divergences.append(
            Divergence(
                kind,
                message,
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_id,
                details=details or {},
            )
        )

    xml_notes_by_id: dict[str, MusicXmlNote] = {}
    for note in parsed_xml.notes:
        if note.musicxml_id is None:
            add(
                "contract_error",
                "Pitched MusicXML note is missing its HOMR id",
                details={"part": note.part, "measure": note.measure},
            )
            continue
        if note.musicxml_id in xml_notes_by_id:
            add(
                "contract_error",
                "Duplicate pitched MusicXML note id",
                musicxml_id=note.musicxml_id,
            )
            continue
        xml_notes_by_id[note.musicxml_id] = note

    raw_notes = sidecar.get("notes")
    raw_groups = sidecar.get("visual_groups")
    if not isinstance(raw_notes, list):
        add("contract_error", "Sidecar notes must be an array")
        raw_notes = []
    if not isinstance(raw_groups, list):
        add("contract_error", "Sidecar visual_groups must be an array")
        raw_groups = []

    sidecar_notes_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_note in enumerate(raw_notes):
        sidecar_note = _mapping(raw_note)
        if sidecar_note is None:
            add("contract_error", "Sidecar note must be an object", details={"index": index})
            continue
        musicxml_id = _string(sidecar_note.get("musicxml_id"))
        if musicxml_id is None:
            add(
                "contract_error",
                "Sidecar note must have a string musicxml_id",
                details={"index": index},
            )
            continue
        if musicxml_id in sidecar_notes_by_id:
            add("contract_error", "Duplicate sidecar note id", musicxml_id=musicxml_id)
            continue
        sidecar_notes_by_id[musicxml_id] = sidecar_note

    groups_by_id: dict[str, dict[str, Any]] = {}
    diagnostic_visual_group_ids: list[str] = []
    groups_by_musicxml_id: dict[str, list[str]] = defaultdict(list)
    for index, raw_group in enumerate(raw_groups):
        group = _mapping(raw_group)
        if group is None:
            add(
                "contract_error",
                "Visual group must be an object",
                details={"index": index},
            )
            continue
        visual_group_id = _string(group.get("visual_group_id"))
        if visual_group_id is None:
            add(
                "contract_error",
                "Visual group must have a string visual_group_id",
                details={"index": index},
            )
            continue
        if visual_group_id in groups_by_id:
            add(
                "contract_error",
                "Duplicate visual group id",
                visual_group_id=visual_group_id,
            )
            continue
        groups_by_id[visual_group_id] = group

        visual_status = _string(group.get("visual_status"))
        group_musicxml_id = group.get("musicxml_id")
        if visual_status not in SUPPORTED_VISUAL_STATUSES:
            add(
                "contract_error",
                "Visual group has an unsupported visual_status",
                visual_group_id=visual_group_id,
                details={"visual_status": visual_status},
            )
        if visual_status == "diagnostic":
            diagnostic_visual_group_ids.append(visual_group_id)
            if group_musicxml_id is not None:
                add(
                    "contract_error",
                    "Diagnostic visual group must not link to MusicXML",
                    visual_group_id=visual_group_id,
                )
            continue
        if visual_status in LINKED_VISUAL_STATUSES:
            if not isinstance(group_musicxml_id, str):
                add(
                    "contract_error",
                    "Canonical or fallback visual group must have one musicxml_id",
                    visual_group_id=visual_group_id,
                )
            else:
                groups_by_musicxml_id[group_musicxml_id].append(visual_group_id)
            if _string(group.get("moment_id")) is None:
                add(
                    "contract_error",
                    "Linked visual group must have a moment_id",
                    musicxml_id=group_musicxml_id if isinstance(group_musicxml_id, str) else None,
                    visual_group_id=visual_group_id,
                )

    xml_ids = set(xml_notes_by_id)
    sidecar_ids = set(sidecar_notes_by_id)
    for musicxml_id in sorted(xml_ids - sidecar_ids):
        add(
            "missing_sidecar_note",
            "Pitched MusicXML note is absent from sidecar notes",
            musicxml_id=musicxml_id,
        )
    for musicxml_id in sorted(sidecar_ids - xml_ids):
        add(
            "extra_sidecar_note",
            "Sidecar note id is absent from MusicXML",
            musicxml_id=musicxml_id,
        )

    valid_group_by_musicxml_id: dict[str, dict[str, Any]] = {}
    for musicxml_id in sorted(xml_ids & sidecar_ids):
        xml_note = xml_notes_by_id[musicxml_id]
        sidecar_note = sidecar_notes_by_id[musicxml_id]
        sidecar_pitch = _string(sidecar_note.get("pitch"))
        if sidecar_pitch != xml_note.pitch:
            add(
                "pitch_divergence",
                "Sidecar pitch does not match MusicXML pitch",
                musicxml_id=musicxml_id,
                details={"musicxml_pitch": xml_note.pitch, "sidecar_pitch": sidecar_pitch},
            )

        context_fields = {
            "part": xml_note.part,
            "measure": xml_note.measure,
            "musicxml_staff_number": xml_note.musicxml_staff_number,
            "voice": xml_note.voice,
        }
        for field_name, expected_value in context_fields.items():
            if _integer(sidecar_note.get(field_name)) != expected_value:
                add(
                    "contract_error",
                    f"Sidecar note {field_name} does not match MusicXML",
                    musicxml_id=musicxml_id,
                    details={
                        "expected": expected_value,
                        "actual": sidecar_note.get(field_name),
                    },
                )

        visual_group_id_value = sidecar_note.get("visual_group_id")
        if visual_group_id_value is None:
            add(
                "missing_visual_note",
                "MusicXML note has no linked visual group",
                musicxml_id=musicxml_id,
            )
            continue
        visual_group_id = _string(visual_group_id_value)
        if visual_group_id is None or visual_group_id not in groups_by_id:
            add(
                "missing_visual_note",
                "MusicXML note references a missing visual group",
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_id,
            )
            continue
        group = groups_by_id[visual_group_id]
        if group.get("musicxml_id") != musicxml_id:
            add(
                "contract_error",
                "MusicXML note and visual group inverse links disagree",
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_id,
                details={"group_musicxml_id": group.get("musicxml_id")},
            )
            continue
        linked_group_ids = groups_by_musicxml_id.get(musicxml_id, [])
        if linked_group_ids != [visual_group_id]:
            add(
                "contract_error",
                "MusicXML note must be linked by exactly one visual group",
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_id,
                details={"linked_visual_group_ids": linked_group_ids},
            )
            continue
        if not _has_notehead_geometry(group):
            add(
                "missing_visual_note",
                "Linked visual group has no usable notehead geometry",
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_id,
            )
            continue

        staff_index = _integer(group.get("staff_index"))
        expected_staff_index = xml_note.musicxml_staff_number - 1
        repair_actions = group.get("repair_actions")
        has_cross_staff_action = (
            isinstance(repair_actions, list) and CROSS_STAFF_REPAIR_ACTION in repair_actions
        )
        has_cross_staff_alignment = (
            _string(sidecar_note.get("alignment_method")) == CROSS_STAFF_ALIGNMENT_METHOD
        )
        claims_cross_staff_repair = has_cross_staff_action or has_cross_staff_alignment
        valid_cross_staff_repair = (
            has_cross_staff_action
            and has_cross_staff_alignment
            and group.get("visual_status") == "fallback"
            and staff_index is not None
            and staff_index != expected_staff_index
        )
        if claims_cross_staff_repair and not valid_cross_staff_repair:
            add(
                "contract_error",
                "Cross-staff repair metadata is incomplete or inconsistent",
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_id,
                details={
                    "alignment_method": sidecar_note.get("alignment_method"),
                    "repair_actions": repair_actions,
                    "visual_status": group.get("visual_status"),
                    "musicxml_staff_index": expected_staff_index,
                    "visual_staff_index": staff_index,
                },
            )
        elif staff_index != expected_staff_index and not valid_cross_staff_repair:
            add(
                "contract_error",
                "Visual group physical staff does not match MusicXML staff",
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_id,
                details={"expected": expected_staff_index, "actual": staff_index},
            )

        actual_staff_position = _integer(group.get("staff_position"))
        pitch_staff_number = (
            staff_index + 1
            if valid_cross_staff_repair and staff_index is not None
            else xml_note.musicxml_staff_number
        )
        expected_position = expected_staff_position(
            xml_note, musicxml_staff_number=pitch_staff_number
        )
        if expected_position is None:
            add(
                "unevaluable_visual_pitch",
                "Visual pitch cannot be evaluated without a supported active clef",
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_id,
            )
        elif actual_staff_position is None:
            add(
                "unevaluable_visual_pitch",
                "Visual group has no integer staff_position",
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_id,
            )
        elif actual_staff_position != expected_position and not (
            _pixel_centers_prove_adjacent_staff_position(
                group,
                sidecar_note,
                expected_position,
                groups_by_id=groups_by_id,
                groups_by_musicxml_id=groups_by_musicxml_id,
                sidecar_notes_by_id=sidecar_notes_by_id,
                xml_notes_by_id=xml_notes_by_id,
            )
        ):
            add(
                "visual_pitch_divergence",
                "Visual staff position does not match MusicXML pitch and clef",
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_id,
                details={"expected": expected_position, "actual": actual_staff_position},
            )
        valid_group_by_musicxml_id[musicxml_id] = group

    for musicxml_id, visual_group_ids in sorted(groups_by_musicxml_id.items()):
        if len(visual_group_ids) > 1:
            add(
                "contract_error",
                "Multiple visual groups link to one MusicXML note",
                musicxml_id=musicxml_id,
                details={"visual_group_ids": sorted(visual_group_ids)},
            )
        if musicxml_id not in sidecar_notes_by_id:
            add(
                "contract_error",
                "Visual group links to a missing sidecar note",
                musicxml_id=musicxml_id,
                visual_group_id=visual_group_ids[0] if visual_group_ids else None,
            )

    for event_ids in parsed_xml.chord_events:
        groups = [
            valid_group_by_musicxml_id[musicxml_id]
            for musicxml_id in event_ids
            if musicxml_id in valid_group_by_musicxml_id
        ]
        by_staff: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for group in groups:
            staff_index = _integer(group.get("staff_index"))
            if staff_index is not None:
                by_staff[staff_index].append(group)
        for staff_index, staff_groups in sorted(by_staff.items()):
            if len(staff_groups) < 2:
                continue
            members = [
                {
                    "musicxml_id": _string(group.get("musicxml_id")),
                    "visual_group_id": _string(group.get("visual_group_id")),
                    "moment_id": _string(group.get("moment_id")),
                    "chord_id": _string(group.get("chord_id")),
                }
                for group in staff_groups
            ]
            member_ids = [
                musicxml_id
                for member in members
                if (musicxml_id := member["musicxml_id"]) is not None
            ]
            member_notes = [
                xml_notes_by_id[musicxml_id]
                for musicxml_id in member_ids
                if musicxml_id in xml_notes_by_id
            ]
            first_member_id = member_ids[0] if member_ids else None
            first_visual_group_id = members[0]["visual_group_id"] if members else None
            parts = sorted({note.part for note in member_notes})
            measures = sorted({note.measure for note in member_notes})
            staff_numbers = sorted({note.musicxml_staff_number for note in member_notes})
            voices = sorted({note.voice for note in member_notes})
            event_indexes = sorted({note.event_index for note in member_notes})
            location = (
                f"part {parts[0]}, measure {measures[0]}, MusicXML staff "
                f"{staff_numbers[0]}, voice {voices[0]}, event {event_indexes[0]}"
                if all(
                    len(values) == 1
                    for values in (parts, measures, staff_numbers, voices, event_indexes)
                )
                else f"physical staff index {staff_index}"
            )
            details = {
                "musicxml_ids": member_ids,
                "visual_group_ids": [member["visual_group_id"] for member in members],
                "members": members,
                "part_numbers": parts,
                "measure_numbers": measures,
                "musicxml_staff_numbers": staff_numbers,
                "physical_staff_index": staff_index,
                "voices": voices,
                "event_indexes": event_indexes,
            }

            moments = {_string(group.get("moment_id")) for group in staff_groups}
            if len(moments) != 1 or None in moments:
                add(
                    "contract_error",
                    f"Same-staff MusicXML chord at {location} has inconsistent moment_id "
                    f"assignments: {_format_member_assignments(members, 'moment_id')}; "
                    "members must share one non-null moment_id",
                    musicxml_id=first_member_id,
                    visual_group_id=first_visual_group_id,
                    details=details,
                )

    for stream, musicxml_ids in parsed_xml.non_chord_streams.items():
        seen_moments: dict[str, str] = {}
        for musicxml_id in musicxml_ids:
            group = valid_group_by_musicxml_id.get(musicxml_id)
            if group is None:
                continue
            moment_id = _string(group.get("moment_id"))
            if moment_id is None:
                continue
            previous_id = seen_moments.get(moment_id)
            if previous_id is not None:
                add(
                    "contract_error",
                    "Sequential non-chord notes in one voice share a moment_id",
                    musicxml_id=musicxml_id,
                    visual_group_id=_string(group.get("visual_group_id")),
                    details={
                        "previous_musicxml_id": previous_id,
                        "moment_id": moment_id,
                        "stream": list(stream),
                    },
                )
            else:
                seen_moments[moment_id] = musicxml_id

    chord_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups_by_id.values():
        chord_id = _string(group.get("chord_id"))
        if chord_id is not None and _string(group.get("musicxml_id")) is not None:
            chord_groups[chord_id].append(group)
    for chord_id, groups in chord_groups.items():
        moments = {_string(group.get("moment_id")) for group in groups}
        staff_group_indexes = {_integer(group.get("staff_group_index")) for group in groups}
        staff_indexes = {_integer(group.get("staff_index")) for group in groups}
        if (
            len(moments) != 1
            or None in moments
            or len(staff_group_indexes) != 1
            or None in staff_group_indexes
            or len(staff_indexes) != 1
            or None in staff_indexes
        ):
            add(
                "contract_error",
                "chord_id spans incompatible visual groups",
                details={
                    "chord_id": chord_id,
                    "visual_group_ids": sorted(
                        _string(group.get("visual_group_id")) or "" for group in groups
                    ),
                },
            )

    counts = {
        "musicxml_notes": len(xml_notes_by_id),
        "sidecar_notes": len(sidecar_notes_by_id),
        "visual_groups": len(groups_by_id),
        "linked_visual_groups": sum(
            1
            for group in groups_by_id.values()
            if _string(group.get("visual_status")) in LINKED_VISUAL_STATUSES
        ),
        "diagnostic_visual_groups": len(diagnostic_visual_group_ids),
        "divergences": len(divergences),
    }
    return VisualEvalReport(
        counts=counts,
        divergences=tuple(divergences),
        diagnostic_visual_group_ids=tuple(sorted(diagnostic_visual_group_ids)),
    )
