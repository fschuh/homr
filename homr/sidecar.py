import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from homr.model import Note
from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords


@dataclass(frozen=True)
class PreprocessingMetadata:
    source_image_size: tuple[int, int]
    autocrop_box: tuple[int, int, int, int]
    cropped_size: tuple[int, int]
    resized_size: tuple[int, int]
    resize_scale: tuple[float, float]
    prediction_size: tuple[int, int]

    def prediction_point_to_source(self, point: tuple[float, float]) -> tuple[float, float]:
        pred_w, pred_h = self.prediction_size
        resized_w, resized_h = self.resized_size
        crop_x, crop_y, _crop_w, _crop_h = self.autocrop_box
        x = point[0] * resized_w / pred_w
        y = point[1] * resized_h / pred_h
        return (x / self.resize_scale[0] + crop_x, y / self.resize_scale[1] + crop_y)

    def prediction_contour_to_source(self, contour: Any) -> list[list[float]]:
        points = np.asarray(contour).reshape(-1, 2)
        return [
            [round(x, 3), round(y, 3)]
            for x, y in (
                self.prediction_point_to_source((float(p[0]), float(p[1]))) for p in points
            )
        ]


@dataclass
class VisualGroup:
    visual_id: str
    staff_index: int
    staff_position: int
    prediction_center: tuple[float, float]
    transformer_center: tuple[float, float] | None
    notehead_contours: list[list[list[float]]]
    stem_contours: list[list[list[float]]]
    linked_musicxml_ids: list[str] = field(default_factory=list)

    @property
    def bbox(self) -> list[float]:
        points = [
            point for contour in self.notehead_contours + self.stem_contours for point in contour
        ]
        if not points:
            return []
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]


@dataclass
class MusicXmlNoteRecord:
    musicxml_id: str
    part: int
    measure: int
    staff: int
    voice: int
    pitch: str | None
    duration: str
    match_confidence: float
    visual_group_id: str | None


@dataclass
class VisualMatch:
    symbol: EncodedSymbol
    visual_id: str | None
    confidence: float


class SidecarCollector:
    def __init__(self, metadata: PreprocessingMetadata) -> None:
        self.metadata = metadata
        self.visual_groups: dict[str, VisualGroup] = {}
        self.matches_by_symbol_id: dict[int, VisualMatch] = {}
        self.musicxml_notes: list[MusicXmlNoteRecord] = []
        self.unmatched_visual_notes: set[str] = set()
        self._next_musicxml_note_id = 1

    def add_staff_visual_notes(
        self, staff_index: int, original_notes: list[Note], transformed_notes: list[Note]
    ) -> None:
        for original, transformed in zip(original_notes, transformed_notes, strict=False):
            if original.visual_id is None:
                continue
            notehead_contour = self.metadata.prediction_contour_to_source(original.box.polygon)
            stem_contours = []
            if original.stem is not None:
                stem_contours.append(
                    self.metadata.prediction_contour_to_source(original.stem.polygon)
                )
            self.visual_groups[original.visual_id] = VisualGroup(
                visual_id=original.visual_id,
                staff_index=staff_index,
                staff_position=original.position,
                prediction_center=original.center,
                transformer_center=transformed.center,
                notehead_contours=[notehead_contour],
                stem_contours=stem_contours,
            )
            self.unmatched_visual_notes.add(original.visual_id)

    def add_staff_matches(self, symbols: list[EncodedSymbol], staff_index: int) -> None:
        visual_groups = [
            group for group in self.visual_groups.values() if group.staff_index == staff_index
        ]
        visual_groups = sorted(
            visual_groups,
            key=lambda group: (
                group.transformer_center[0] if group.transformer_center is not None else 0,
                group.transformer_center[1] if group.transformer_center is not None else 0,
            ),
        )
        visual_cursor = 0
        for chord in sort_token_chords(symbols):
            note_symbols = [symbol for symbol in chord if symbol.rhythm.startswith("note")]
            if not note_symbols:
                continue
            group_size = len(note_symbols)
            candidate_group = visual_groups[visual_cursor : visual_cursor + group_size]
            visual_cursor += group_size
            for symbol, visual_group in zip(note_symbols, candidate_group, strict=False):
                confidence = self._score_match(symbol, visual_group)
                self.matches_by_symbol_id[id(symbol)] = VisualMatch(
                    symbol=symbol,
                    visual_id=visual_group.visual_id,
                    confidence=confidence,
                )
                self.unmatched_visual_notes.discard(visual_group.visual_id)
            for symbol in note_symbols[len(candidate_group) :]:
                self.matches_by_symbol_id[id(symbol)] = VisualMatch(
                    symbol=symbol,
                    visual_id=None,
                    confidence=0.0,
                )

    def create_musicxml_id(self) -> str:
        musicxml_id = f"homr-note-{self._next_musicxml_note_id}"
        self._next_musicxml_note_id += 1
        return musicxml_id

    def record_musicxml_note(
        self,
        musicxml_id: str,
        part: int,
        measure: int,
        staff: int,
        voice: int,
        symbol: EncodedSymbol,
    ) -> None:
        match = self.matches_by_symbol_id.get(id(symbol))
        visual_id = match.visual_id if match is not None else None
        confidence = match.confidence if match is not None else 0.0
        pitch = symbol.pitch if symbol.pitch not in ("_", ".") else None
        if visual_id is not None and visual_id in self.visual_groups:
            self.visual_groups[visual_id].linked_musicxml_ids.append(musicxml_id)
        self.musicxml_notes.append(
            MusicXmlNoteRecord(
                musicxml_id=musicxml_id,
                part=part,
                measure=measure,
                staff=staff,
                voice=voice,
                pitch=pitch,
                duration=symbol.rhythm,
                match_confidence=confidence,
                visual_group_id=visual_id,
            )
        )

    def unmatched_musicxml_notes(self) -> list[str]:
        return [note.musicxml_id for note in self.musicxml_notes if note.visual_group_id is None]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "source_image_size": list(self.metadata.source_image_size),
            "preprocessing": {
                "autocrop_box": list(self.metadata.autocrop_box),
                "cropped_size": list(self.metadata.cropped_size),
                "resized_size": list(self.metadata.resized_size),
                "resize_scale": [
                    round(self.metadata.resize_scale[0], 8),
                    round(self.metadata.resize_scale[1], 8),
                ],
                "prediction_size": list(self.metadata.prediction_size),
            },
            "notes": [record.__dict__ for record in self.musicxml_notes],
            "visual_groups": [
                {
                    "visual_group_id": group.visual_id,
                    "staff_index": group.staff_index,
                    "staff_position": group.staff_position,
                    "center": [
                        round(
                            self.metadata.prediction_point_to_source(group.prediction_center)[0], 3
                        ),
                        round(
                            self.metadata.prediction_point_to_source(group.prediction_center)[1], 3
                        ),
                    ],
                    "bbox": group.bbox,
                    "notehead_contours": group.notehead_contours,
                    "stem_contours": group.stem_contours,
                    "musicxml_ids": group.linked_musicxml_ids,
                }
                for group in sorted(self.visual_groups.values(), key=lambda g: g.visual_id)
            ],
            "unmatched_musicxml_notes": self.unmatched_musicxml_notes(),
            "unmatched_visual_notes": sorted(self.unmatched_visual_notes),
        }

    def _score_match(self, symbol: EncodedSymbol, visual_group: VisualGroup) -> float:
        score = 0.65
        if visual_group.transformer_center is not None and symbol.coordinates is not None:
            try:
                coords = np.asarray(symbol.coordinates).reshape(-1)
                if len(coords) >= 2:
                    dx = abs(float(coords[0]) - visual_group.transformer_center[0])
                    dy = abs(float(coords[1]) - visual_group.transformer_center[1])
                    score += max(0.0, 0.25 - (dx + dy) / 1000.0)
            except (TypeError, ValueError):
                pass
        return min(round(score, 3), 1.0)


def write_sidecar(path: str, collector: SidecarCollector) -> None:
    sidecar_path = Path(path)
    sidecar_path.write_text(json.dumps(collector.to_json_dict(), indent=2), encoding="utf-8")
