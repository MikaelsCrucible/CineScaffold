from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from cinescaffold.viewer_manifest import build_viewer_manifest, inspect_glb


def _write_glb(path: Path, document: dict) -> None:
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    total_length = 12 + 8 + len(encoded)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
    )


class ViewerManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene_ir = {
            "schema_version": "0.1",
            "scene_id": "viewer-test",
            "timeline": {
                "frame_start": 1,
                "frame_end": 3,
                "frame_count": 3,
                "fps_numerator": 24,
                "fps_denominator": 1,
                "duration_seconds": 0.125,
                "time_domain": "half_open",
            },
            "entities": [
                {
                    "entity_id": "actor",
                    "parent_id": None,
                    "local_state_track": {
                        "mode": "per_frame",
                        "samples": [
                            {
                                "frame": 1,
                                "value": {
                                    "translation_m": [0, 0, 0],
                                    "rotation_quaternion_wxyz": [1, 0, 0, 0],
                                    "scale": [1, 1, 1],
                                    "render_visible": True,
                                },
                            },
                            {
                                "frame": 2,
                                "value": {
                                    "translation_m": [1, 0, 0],
                                    "rotation_quaternion_wxyz": [1, 0, 0, 0],
                                    "scale": [1, 1, 1],
                                    "render_visible": False,
                                },
                            },
                            {
                                "frame": 3,
                                "value": {
                                    "translation_m": [2, 0, 0],
                                    "rotation_quaternion_wxyz": [1, 0, 0, 0],
                                    "scale": [1, 1, 1],
                                    "render_visible": False,
                                },
                            },
                        ],
                    },
                }
            ],
            "camera": {
                "camera_id": "main",
                "state_track": {
                    "mode": "constant",
                    "value": {
                        "translation_m": [0, -5, 2],
                        "rotation_quaternion_wxyz": [1, 0, 0, 0],
                        "focal_length_mm": 50,
                    },
                },
            },
        }
        self.document = {
            "asset": {"version": "2.0"},
            "nodes": [
                {
                    "name": "CS_ENTITY_actor",
                    "extras": {
                        "cinescaffold_id": "actor",
                        "cinescaffold_kind": "entity",
                    },
                },
                {
                    "name": "CS_CAMERA_main",
                    "camera": 0,
                    "extras": {
                        "cinescaffold_id": "main",
                        "cinescaffold_kind": "camera",
                    },
                },
            ],
            "cameras": [{"type": "perspective", "perspective": {"yfov": 1.0}}],
            "animations": [
                {
                    "name": "Scene",
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "translation"}}
                    ],
                    "samplers": [{"input": 0, "output": 1}],
                }
            ],
        }

    def test_manifest_binds_ir_glb_nodes_timeline_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glb_path = Path(directory) / "scene.glb"
            _write_glb(glb_path, self.document)
            manifest = build_viewer_manifest(
                scene_ir=self.scene_ir,
                scene_ir_hash="sha256:ir",
                glb_path=glb_path,
                blender_version="5.2.1 LTS",
                runtime_validation={"passed": True, "violations": []},
            )

        self.assertEqual(manifest["schema_version"], "0.1")
        self.assertEqual(manifest["scene_ir"]["canonical_hash"], "sha256:ir")
        self.assertTrue(manifest["glb"]["sha256"].startswith("sha256:"))
        self.assertEqual(manifest["default_camera"]["node_index"], 1)
        self.assertEqual(manifest["entities"][0]["node_index"], 0)
        self.assertEqual(
            manifest["visibility_tracks"][0]["ranges"],
            [
                {"frame_start": 1, "frame_end_exclusive": 2, "visible": True},
                {"frame_start": 2, "frame_end_exclusive": 4, "visible": False},
            ],
        )
        self.assertEqual(manifest["fallback"]["on_glb_error"], "mp4")
        self.assertIn("visibility_requires_scene_ir", [x["code"] for x in manifest["warnings"]])

    def test_manifest_rejects_glb_without_stable_entity_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glb_path = Path(directory) / "scene.glb"
            document = {**self.document, "nodes": self.document["nodes"][1:]}
            _write_glb(glb_path, document)
            with self.assertRaisesRegex(ValueError, "Entity 映射不一致"):
                build_viewer_manifest(
                    scene_ir=self.scene_ir,
                    scene_ir_hash="sha256:ir",
                    glb_path=glb_path,
                    blender_version="5.2.1 LTS",
                    runtime_validation={"passed": True, "violations": []},
                )

    def test_inspector_rejects_declared_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            glb_path = Path(directory) / "bad.glb"
            _write_glb(glb_path, self.document)
            data = bytearray(glb_path.read_bytes())
            struct.pack_into("<I", data, 8, len(data) + 4)
            glb_path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "声明长度"):
                inspect_glb(glb_path)


if __name__ == "__main__":
    unittest.main()
