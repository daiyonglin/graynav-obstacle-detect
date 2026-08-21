from __future__ import annotations

import re
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DemoContractTest(unittest.TestCase):
    def test_object_names_are_not_installed_as_hud_status(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        package = (ROOT.parent / "sdk_overlay/ssne_ai_demo.mk").read_text(
            encoding="utf-8"
        )
        self.assertIn("NAV_*.ssbmp", cmake)
        self.assertNotIn("INFO_*.ssbmp", cmake)
        self.assertIn("NAV_*.ssbmp", package)
        self.assertNotIn("INFO_*.ssbmp", package)
        names = {path.name for path in (ROOT / "app_assets/osd").glob("NAV_*.ssbmp")}
        expected = {
            f"NAV_{distance}_{position}.ssbmp"
            for distance in ("NEAR", "MID", "FAR", "UNKNOWN")
            for position in ("LEFT", "FRONT", "RIGHT", "MULTI", "BLOCKED")
        }
        self.assertEqual(names, expected)
        for name in names:
            raw = (ROOT / "app_assets/osd" / name).read_bytes()
            magic, width, height, colors = struct.unpack("<IIII", raw[:16])
            self.assertEqual(magic, 0x5353424D)
            self.assertEqual(colors, 32)
            self.assertEqual(len(raw), 16 + width * height)

    def test_osd_and_serial_budgets_are_explicit(self) -> None:
        source = (ROOT / "src/utils.cpp").read_text(encoding="utf-8")
        demo = (ROOT / "demo_obstacle.cpp").read_text(encoding="utf-8")
        self.assertRegex(source, r"max_display_boxes\s*=\s*std::min<size_t>\(result.items.size\(\), 2\)")
        self.assertIn("stair_quads.size() > 3U", source)
        self.assertIn("zones=L:", demo)
        self.assertIn("dist=", demo)
        self.assertIn("risk=", demo)
        self.assertIn("[F", demo)

    def test_indoor8_cannot_use_person_part_bridge(self) -> None:
        tracker = (ROOT / "src/tracker.cpp").read_text(encoding="utf-8")
        self.assertIn("semantic::ModelClassCount() != 25", tracker)
        self.assertIn("const float evidence_decay = indoor8 ? 0.85f : 0.95f", tracker)
        self.assertIn("detection.score >= 0.45f", tracker)

    def test_ranging_keeps_metric_estimate_separate_from_safety_bound(self) -> None:
        ranging = (ROOT / "src/ranging.cpp").read_text(encoding="utf-8")
        stabilizer = (ROOT / "src/guidance_stabilizer.cpp").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('item->distance_source = "nearfield_cap"', ranging)
        self.assertIn("std::min(item->safe_distance_m, near_upper)", ranging)
        self.assertIn("distance_identity_for", stabilizer)
        self.assertIn("distance_history_.size() > 3U", stabilizer)

    def test_fault_packet_clears_stale_navigation_state(self) -> None:
        demo = (ROOT / "demo_obstacle.cpp").read_text(encoding="utf-8")
        self.assertIn('decision.primary_class = "abnormal"', demo)
        self.assertIn('decision->primary_class = "abnormal"', demo)
        self.assertIn('decision->object_label = "NONE"', demo)
        self.assertIn('voice=ABNORMAL', demo)

    def test_voice_uses_action_only_short_prompts(self) -> None:
        voice = (ROOT / "src/voice_notifier.cpp").read_text(encoding="utf-8")
        for long_prompt in (
            "person_stop",
            "obstacle_stop",
            "obstacle_slow",
            "stair_stop",
            "stair_slow",
            "possible_stair",
            "kPromptPersonStop",
            "kPromptObstacleStop",
            "kPromptStairStop",
        ):
            self.assertNotIn(long_prompt, voice)
        self.assertIn("pending_key_ = action", voice)
        self.assertIn('action == "turn_left"', voice)
        self.assertIn('action == "turn_right"', voice)


if __name__ == "__main__":
    unittest.main()
