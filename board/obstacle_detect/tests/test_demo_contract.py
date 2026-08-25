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
        normal = {
            f"NAV_{distance}_{position}.ssbmp"
            for distance in ("NEAR", "MID", "FAR", "UNKNOWN")
            for position in ("LEFT", "FRONT", "RIGHT", "MULTI", "BLOCKED")
        }
        wall = {
            f"NAV_WALL_{distance}_{position}.ssbmp"
            for distance in ("NEAR", "MID", "FAR", "UNKNOWN")
            for position in ("LEFT", "FRONT", "RIGHT", "MULTI", "BLOCKED")
        }
        expected = normal | wall
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
        run = (ROOT / "scripts/run.sh").read_text(encoding="utf-8")
        self.assertNotIn('item->distance_source = "nearfield_cap"', ranging)
        self.assertIn("std::min(item->safe_distance_m, near_upper)", ranging)
        self.assertIn('env_float("A1_RANGE_GEOMETRY_SCALE", 1.60f)', ranging)
        self.assertIn("A1_RANGE_GEOMETRY_SCALE:-1.60", run)
        self.assertIn('env_float("A1_RANGE_SIZE_SCALE", 1.00f)', ranging)
        self.assertIn("A1_RANGE_SIZE_SCALE:-1.00", run)
        self.assertIn('env_float("A1_RANGE_SAFETY_SCALE", 1.00f)', ranging)
        self.assertIn("A1_RANGE_SAFETY_SCALE:-1.00", run)
        self.assertIn("std::min(displayed_safe, planning_safe)", ranging)
        self.assertIn("distance_identity_for", stabilizer)
        self.assertIn("distance_history_.size() > 3U", stabilizer)
        self.assertIn('"person_partial_width"', ranging)

    def test_navigation_actions_use_display_distance_without_ttc(self) -> None:
        planner = (ROOT / "src/avoidance_planner.cpp").read_text(encoding="utf-8")
        tracker = (ROOT / "src/tracker.cpp").read_text(encoding="utf-8")
        header = (ROOT / "include/avoidance_planner.hpp").read_text(
            encoding="utf-8"
        )
        run = (ROOT / "scripts/run.sh").read_text(encoding="utf-8")
        self.assertIn("if (item.distance_m >= 0.0f) return item.distance_m", planner)
        self.assertIn('"center_near_distance"', planner)
        self.assertNotIn("StopTtcSeconds", planner)
        self.assertNotIn("min_ttc", header)
        self.assertNotIn("StopTtcSeconds", tracker)
        self.assertNotIn("A1_TTC_STOP_S", run)
        self.assertIn("stop_below=", planner)
        self.assertIn("slow_below=", planner)
        self.assertIn("ttc=diagnostic_only", planner)
        self.assertIn('A1_RANGE_URGENT_M:-0.60', run)
        self.assertIn('A1_RANGE_NEAR_M:-0.80', run)
        self.assertIn('A1_RANGE_WARNING_M:-1.50', run)
        semantic = (ROOT / "src/semantic_config.cpp").read_text(encoding="utf-8")
        self.assertIn('env_float("A1_RANGE_URGENT_M", 0.60f)', semantic)
        self.assertIn('env_float("A1_RANGE_NEAR_M", 0.80f)', semantic)
        self.assertIn('env_float("A1_RANGE_WARNING_M", 1.50f)', semantic)
        fusion = (ROOT / "src/surface_fusion.cpp").read_text(encoding="utf-8")
        self.assertIn("depth_m < semantic::NearDistanceM()", fusion)
        self.assertIn("depth_m < semantic::WarningDistanceM()", fusion)
        self.assertNotIn("if (depth_m < 1.25f)", fusion)
        self.assertNotIn("if (depth_m < 2.20f)", fusion)

    def test_cover_fault_is_dark_only_and_white_wall_safe(self) -> None:
        demo = (ROOT / "demo_obstacle.cpp").read_text(encoding="utf-8")
        run = (ROOT / "scripts/run.sh").read_text(encoding="utf-8")
        self.assertIn("const bool hard_black_cover", demo)
        self.assertIn("stats.cover_candidate = hard_black_cover", demo)
        self.assertNotIn("hard_bright_cover", demo)
        self.assertNotIn("hard_flat_frame", demo)
        self.assertNotIn("cover_score >= score_threshold", demo)
        self.assertIn("freeze_observable", demo)
        self.assertIn("A1_COVER_BLACK_MEAN_MAX:-45", run)
        self.assertIn("A1_COVER_BLACK_RATIO_PCT:-80", run)

    def test_range_filter_uses_all_distance_median_and_asymmetric_release(self) -> None:
        tracker = (ROOT / "src/tracker.cpp").read_text(encoding="utf-8")
        header = (ROOT / "include/tracker.hpp").read_text(encoding="utf-8")
        self.assertIn("std::array<float, 5> inverse_depth_history", header)
        self.assertIn("track->inverse_depth_count >= 3", tracker)
        self.assertNotIn("measured_distance > 2.0f", tracker)
        self.assertIn("track->pending_range_count < 3", tracker)
        self.assertIn("accepted_residual < 0.0f", tracker)

    def test_partial_person_does_not_use_full_height_or_scale_anchor(self) -> None:
        ranging = (ROOT / "src/ranging.cpp").read_text(encoding="utf-8")
        fusion = (ROOT / "src/surface_fusion.cpp").read_text(encoding="utf-8")
        self.assertIn("reliable_person_full_extent", ranging)
        self.assertIn("fx_ * physical_width / pixel_width", ranging)
        self.assertIn('item.distance_source == "person_partial_width"', fusion)
        self.assertIn('item.distance_source = "person_partial_depth_fused"', fusion)

    def test_field_zone_and_wall_guidance_defaults(self) -> None:
        semantic = (ROOT / "src/semantic_config.cpp").read_text(encoding="utf-8")
        fusion = (ROOT / "src/surface_fusion.cpp").read_text(encoding="utf-8")
        run = (ROOT / "scripts/run.sh").read_text(encoding="utf-8")
        self.assertIn('A1_SECTOR_LEFT_BOUND:-0.35', run)
        self.assertIn('A1_SECTOR_RIGHT_BOUND:-0.65', run)
        self.assertIn('env_float("A1_SECTOR_LEFT_BOUND", 0.35f)', semantic)
        self.assertIn('env_float("A1_SECTOR_RIGHT_BOUND", 0.65f)', semantic)
        self.assertIn('A1_ENABLE_WALL_GUIDANCE:-0', run)
        self.assertIn('env_float("A1_ENABLE_WALL_GUIDANCE", 0.0f)', semantic)
        self.assertIn("const bool wall_guidance", fusion)
        demo = (ROOT / "demo_obstacle.cpp").read_text(encoding="utf-8")
        self.assertIn("semantic::WallGuidanceEnabled()", demo)

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

    def test_voice_cadence_follows_latest_action(self) -> None:
        voice = (ROOT / "src/voice_notifier.cpp").read_text(encoding="utf-8")
        run = (ROOT / "scripts/run.sh").read_text(encoding="utf-8")
        self.assertIn('getenv_int("A1_VOICE_COOLDOWN_MS", 0)', voice)
        self.assertIn('if (action == "clear") return clear_repeat_ms_', voice)
        self.assertNotIn("86400000", voice)
        self.assertIn('getenv_int("A1_VOICE_STOP_REPEAT_MS", 500)', voice)
        self.assertIn('getenv_int("A1_VOICE_FAULT_REPEAT_MS", 0)', voice)
        self.assertIn('getenv_int("A1_VOICE_ACTION_PROMPT_MS", 1000)', voice)
        self.assertIn("fixed_frame_ ? action_prompt_ms_", voice)
        self.assertIn('getenv_int("A1_VOICE_STOP_FOLLOWUP_HOLD_MS", 0)', voice)
        self.assertIn('A1_VOICE_COOLDOWN_MS:-0', run)
        self.assertIn('A1_VOICE_STOP_REPEAT_MS:-500', run)
        self.assertIn('A1_VOICE_FAULT_REPEAT_MS:-0', run)
        self.assertIn('A1_VOICE_ACTION_PROMPT_MS:-1000', run)
        self.assertIn('A1_VOICE_STOP_FOLLOWUP_HOLD_MS:-0', run)
        self.assertIn('A1_VOICE_DIAG:-0', run)
        self.assertIn('getenv_int("A1_VOICE_INTERVAL_FRAMES", 1)', voice)
        self.assertIn('getenv_bool("A1_VOICE_REQUIRE_ACK", false)', voice)
        self.assertIn('getenv_bool("A1_VOICE_QUERY_IDLE", false)', voice)
        self.assertIn('getenv_bool("A1_VOICE_DIAG", false)', voice)
        self.assertIn('A1_VOICE_INTERVAL_FRAMES:-1', run)

    def test_normal_nav_packet_is_emitted_each_inference_frame(self) -> None:
        demo = (ROOT / "demo_obstacle.cpp").read_text(encoding="utf-8")
        run = (ROOT / "scripts/run.sh").read_text(encoding="utf-8")
        self.assertIn('A1_OUTPUT_INTERVAL_FRAMES:-1', run)
        self.assertIn("const bool nav_every_frame = output_interval_frames == 1", demo)
        self.assertIn("nav_every_frame || nav_heartbeat_due", demo)
        self.assertIn("constexpr int kOutputIntervalFrames = 1", demo)
        self.assertIn("const int output_interval_frames = kOutputIntervalFrames", demo)
        self.assertIn("const int osd_interval_frames = 1", demo)
        self.assertIn("per_frame_nav_continuous_voice_side_turn_v2", demo)

    def test_confirmed_wall_has_distinct_static_hud(self) -> None:
        source = (ROOT / "src/utils.cpp").read_text(encoding="utf-8")
        self.assertIn('wall_only ? "NAV_WALL_" : "NAV_"', source)


if __name__ == "__main__":
    unittest.main()
