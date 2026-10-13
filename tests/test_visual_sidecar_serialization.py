import json
import tempfile
import unittest
from pathlib import Path

from homr.visual_sidecar.serialization import write_visual_sidecar


class TestVisualSidecarSerialization(unittest.TestCase):
    def test_writes_compact_json(self) -> None:
        document = {"version": 3, "visual_groups": [{"notehead_contours": [[[1, 2], [3, 4]]]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.homr.visual.json"
            write_visual_sidecar(str(path), document)
            text = path.read_text(encoding="utf-8")
        self.assertEqual(json.loads(text), document)
        self.assertEqual(text, json.dumps(document, separators=(",", ":")))
