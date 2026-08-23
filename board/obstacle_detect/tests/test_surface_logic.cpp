#include "surface_fusion.hpp"
#include "surface_segmentation.hpp"
#include "guidance_stabilizer.hpp"
#include "avoidance_planner.hpp"
#include "ranging.hpp"
#include "tracker.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

std::vector<float> segmentation_logits(bool hazard)
{
    std::vector<float> logits(SURFACE_GRID_CELLS * SURFACE_CLASS_COUNT, -4.0f);
    for (int y = 0; y < SURFACE_GRID_SIZE; ++y) {
        for (int x = 0; x < SURFACE_GRID_SIZE; ++x) {
            logits[(y * SURFACE_GRID_SIZE + x) * SURFACE_CLASS_COUNT] = 4.0f;
        }
    }
    if (hazard) {
        for (int y = SURFACE_GRID_SIZE * 2 / 3; y < SURFACE_GRID_SIZE * 5 / 6; ++y) {
            for (int x = SURFACE_GRID_SIZE * 2 / 5; x < SURFACE_GRID_SIZE * 3 / 5; ++x) {
                const int base = (y * SURFACE_GRID_SIZE + x) * SURFACE_CLASS_COUNT;
                logits[base] = -4.0f;
                logits[base + STEP_OR_DROP] = 4.0f;
            }
        }
    }
    return logits;
}

std::vector<float> depth_logits(int active_bin)
{
    std::vector<float> logits(SURFACE_GRID_CELLS * DEPTH_BIN_COUNT, -4.0f);
    for (int cell = 0; cell < SURFACE_GRID_CELLS; ++cell) {
        logits[cell * DEPTH_BIN_COUNT + active_bin] = 4.0f;
    }
    return logits;
}

std::vector<float> ambiguous_depth_logits(int first_bin, int second_bin)
{
    std::vector<float> logits(SURFACE_GRID_CELLS * DEPTH_BIN_COUNT, -8.0f);
    for (int cell = 0; cell < SURFACE_GRID_CELLS; ++cell) {
        logits[cell * DEPTH_BIN_COUNT + first_bin] = 5.0f;
        logits[cell * DEPTH_BIN_COUNT + second_bin] = 5.0f;
    }
    return logits;
}

std::vector<float> uniform_segmentation_logits(int active_class)
{
    std::vector<float> logits(SURFACE_GRID_CELLS * SURFACE_CLASS_COUNT, -4.0f);
    for (int cell = 0; cell < SURFACE_GRID_CELLS; ++cell) {
        logits[cell * SURFACE_CLASS_COUNT + active_class] = 4.0f;
    }
    return logits;
}

std::vector<float> blocked_segmentation_logits()
{
    std::vector<float> logits = segmentation_logits(false);
    for (int y = SURFACE_GRID_SIZE / 3; y < SURFACE_GRID_SIZE; ++y) {
        for (int x = SURFACE_GRID_SIZE * 2 / 5; x < SURFACE_GRID_SIZE * 3 / 5; ++x) {
            const int base = (y * SURFACE_GRID_SIZE + x) * SURFACE_CLASS_COUNT;
            logits[base] = -4.0f;
            logits[base + BLOCKED_SURFACE] = 4.0f;
        }
    }
    return logits;
}

std::vector<float> hwc_to_chw(const std::vector<float>& hwc, int channels)
{
    std::vector<float> chw(hwc.size(), 0.0f);
    for (int cell = 0; cell < SURFACE_GRID_CELLS; ++cell) {
        for (int channel = 0; channel < channels; ++channel) {
            chw[channel * SURFACE_GRID_CELLS + cell] = hwc[cell * channels + channel];
        }
    }
    return chw;
}

std::vector<float> packed_scene_logits(bool hazard,
                                       int depth_bin,
                                       bool edge,
                                       bool depth_jump = true)
{
    const std::vector<float> seg = segmentation_logits(hazard);
    std::vector<float> depth = depth_logits(depth_bin);
    const int edge_row = SURFACE_GRID_SIZE * 3 / 4;
    if (depth_jump) {
        const int upper_bin = std::max(0, depth_bin - 3);
        const int lower_bin = std::min(DEPTH_BIN_COUNT - 1, depth_bin + 3);
        depth.assign(SURFACE_GRID_CELLS * DEPTH_BIN_COUNT, -4.0f);
        for (int y = 0; y < SURFACE_GRID_SIZE; ++y) {
            const int active_bin = y < edge_row ? upper_bin : lower_bin;
            for (int x = 0; x < SURFACE_GRID_SIZE; ++x) {
                const int cell = y * SURFACE_GRID_SIZE + x;
                depth[cell * DEPTH_BIN_COUNT + active_bin] = 4.0f;
            }
        }
    }
    std::vector<float> packed(SURFACE_GRID_CELLS * UNIFIED_SCENE_CHANNELS, -8.0f);
    for (int cell = 0; cell < SURFACE_GRID_CELLS; ++cell) {
        for (int c = 0; c < SURFACE_CLASS_COUNT; ++c) {
            packed[cell * UNIFIED_SCENE_CHANNELS + c] =
                seg[cell * SURFACE_CLASS_COUNT + c];
        }
        for (int c = 0; c < DEPTH_BIN_COUNT; ++c) {
            packed[cell * UNIFIED_SCENE_CHANNELS + SURFACE_CLASS_COUNT + c] =
                depth[cell * DEPTH_BIN_COUNT + c];
        }
    }
    if (edge) {
        const int row = edge_row;
        for (int y = row - 1; y <= row + 1; ++y) {
            for (int x = SURFACE_GRID_SIZE / 3; x < SURFACE_GRID_SIZE * 2 / 3; ++x) {
                packed[(y * SURFACE_GRID_SIZE + x) * UNIFIED_SCENE_CHANNELS +
                       STAIR_EDGE_CHANNEL] = 8.0f;
            }
        }
    }
    return packed;
}

}  // namespace

int main()
{
    unsetenv("A1_SECTOR_LEFT_BOUND");
    unsetenv("A1_SECTOR_RIGHT_BOUND");
    unsetenv("A1_ENABLE_WALL_GUIDANCE");
    // The unified head uses the fixed indoor8 order, not ROD25 or COCO80 IDs.
    assert(obstacle::semantic::RawLabel(0) == "person");
    assert(obstacle::semantic::RawLabel(1) == "chair");
    assert(obstacle::semantic::RawLabel(2) == "dining_table");
    assert(obstacle::semantic::RawLabel(3) == "backpack");
    assert(obstacle::semantic::SemanticClassFromRaw(0) == obstacle::semantic::PERSON);
    assert(obstacle::semantic::SemanticClassFromRaw(1) == obstacle::semantic::CHAIR_SEAT);
    assert(obstacle::semantic::SemanticClassFromRaw(2) == obstacle::semantic::TABLE_DESK);
    assert(obstacle::semantic::SemanticClassFromRaw(3) == obstacle::semantic::BAG_SUITCASE);
    assert(std::fabs(obstacle::semantic::SectorLeftBoundaryRatio() - 0.35f) < 0.001f);
    assert(std::fabs(obstacle::semantic::SectorRightBoundaryRatio() - 0.65f) < 0.001f);
    assert(!obstacle::semantic::WallGuidanceEnabled());

    obstacle::SurfaceDecisionFusion fusion;
    AvoidanceDecision detection;
    detection.action = "clear";
    SurfaceResult pending;
    pending.perception_degraded = false;
    AvoidanceDecision pending_fused = fusion.Fuse(detection, pending, 0);
    assert(pending_fused.action == "slow");
    assert(!pending_fused.perception_degraded);
    pending.perception_degraded = true;
    AvoidanceDecision degraded_fused = fusion.Fuse(detection, pending, 0);
    assert(degraded_fused.action == "clear");
    assert(degraded_fused.perception_degraded);

    // A large side object overlaps the narrow centre footprint by design, but
    // its primary position remains lateral.  It must produce an opposite turn
    // after normal action stabilization rather than an immediate STOP.
    auto planner_detection = [](float x1, float x2, const char* sector,
                                int64_t timestamp) {
        DetectionResult frame;
        frame.view_id = 0;
        frame.roi = {0, 0, 720, 1280};
        frame.timestamp_ms = timestamp;
        DetectionItem item;
        item.box = {x1, 650.0f, x2, 1240.0f};
        item.raw_class_id = 1;
        item.raw_label = "chair";
        item.class_id = obstacle::semantic::CHAIR_SEAT;
        item.label = "chair";
        item.semantic_class = "chair";
        item.sector = sector;
        item.distance_m = 0.90f;
        item.safe_distance_m = 0.75f;
        item.distance_confidence = 0.80f;
        item.distance_source = "fused";
        item.risk_level = "near";
        item.quality = "good";
        item.ttc_s = 0.90f;
        frame.items.push_back(item);
        return frame;
    };
    obstacle::AvoidancePlanner left_planner;
    left_planner.Initialize({720, 1280});
    left_planner.Update(planner_detection(40.0f, 340.0f, "left", 100), 0, 100);
    const AvoidanceDecision left_avoid = left_planner.Update(
        planner_detection(40.0f, 340.0f, "left", 200), 0, 200);
    assert(left_avoid.action == "turn_right");
    assert(left_avoid.recommended_direction == "right");

    obstacle::AvoidancePlanner right_planner;
    right_planner.Initialize({720, 1280});
    right_planner.Update(planner_detection(380.0f, 680.0f, "right", 100), 0, 100);
    const AvoidanceDecision right_avoid = right_planner.Update(
        planner_detection(380.0f, 680.0f, "right", 200), 0, 200);
    assert(right_avoid.action == "turn_left");
    assert(right_avoid.recommended_direction == "left");

    // Reproduce the board log: one right-side partial-person box may cover all
    // three footprint masks, but its centre and track identity are still on
    // the right.  It is one obstacle, not a three-corridor blockade.
    obstacle::AvoidancePlanner broad_right_planner;
    broad_right_planner.Initialize({720, 1280});
    broad_right_planner.Update(
        planner_detection(250.0f, 710.0f, "right", 100), 0, 100);
    const AvoidanceDecision broad_right_avoid = broad_right_planner.Update(
        planner_detection(250.0f, 710.0f, "right", 200), 0, 200);
    assert(broad_right_avoid.action == "turn_left");
    assert(broad_right_avoid.hazard_sector == "right");
    assert(broad_right_avoid.hazard_position == "RIGHT");
    assert(!broad_right_avoid.left.occupied);
    assert(!broad_right_avoid.center.occupied);
    assert(broad_right_avoid.right.occupied);

    // The same exclusive-zone contract applies at warning range.  This is the
    // exact state seen in the board log where MID RIGHT was displayed but the
    // stale stabilized action remained SLOW.
    obstacle::AvoidancePlanner warning_right_planner;
    warning_right_planner.Initialize({720, 1280});
    DetectionResult warning_right = planner_detection(250.0f, 710.0f, "right", 100);
    warning_right.items[0].distance_m = 1.95f;
    warning_right.items[0].safe_distance_m = 1.80f;
    warning_right.items[0].risk_level = "warning";
    warning_right.items[0].ttc_s = -1.0f;
    warning_right_planner.Update(warning_right, 0, 100);
    warning_right.timestamp_ms = 200;
    const AvoidanceDecision warning_right_avoid =
        warning_right_planner.Update(warning_right, 0, 200);
    assert(warning_right_avoid.action == "turn_left");
    assert(!warning_right_avoid.left.occupied);
    assert(!warning_right_avoid.center.occupied);
    assert(warning_right_avoid.right.occupied);

    // When alternating ROIs briefly retain a centre fragment plus the real
    // right-side body track, an empty left zone is already the only escape.
    // Do not wait for another view-level clearance flag and fall back to SLOW.
    obstacle::AvoidancePlanner center_right_planner;
    center_right_planner.Initialize({720, 1280});
    DetectionResult center_right = warning_right;
    DetectionItem center_fragment = center_right.items[0];
    center_fragment.box = {300.0f, 250.0f, 410.0f, 560.0f};
    center_fragment.sector = "center";
    center_fragment.track_id = 99;
    center_right.items.push_back(center_fragment);
    center_right_planner.Update(center_right, 0, 100);
    center_right.timestamp_ms = 200;
    const AvoidanceDecision center_right_avoid =
        center_right_planner.Update(center_right, 0, 200);
    assert(center_right_avoid.action == "turn_left");
    assert(!center_right_avoid.left.occupied);
    assert(center_right_avoid.center.occupied);
    assert(center_right_avoid.right.occupied);

    // Alternating UPPER/LOWER views may produce a face box and a torso box for
    // the same person.  They share one horizontal body column and must be
    // published as one entity rather than two obstacles.
    DetectionResult duplicate_person;
    DetectionItem body;
    body.raw_class_id = 0;
    body.raw_label = "person";
    body.class_id = obstacle::semantic::PERSON;
    body.label = "person";
    body.quality = "good";
    body.score = 0.55f;
    body.box = {420.0f, 600.0f, 700.0f, 1250.0f};
    DetectionItem face = body;
    face.score = 0.70f;
    face.box = {480.0f, 250.0f, 690.0f, 450.0f};
    duplicate_person.items.push_back(face);
    duplicate_person.items.push_back(body);
    utils::MultiTargetNMS(&duplicate_person, 0.45f, 8);
    assert(duplicate_person.items.size() == 1U);
    assert(duplicate_person.items[0].box[1] == 600.0f);

    obstacle::AvoidancePlanner center_planner;
    center_planner.Initialize({720, 1280});
    const AvoidanceDecision center_stop = center_planner.Update(
        planner_detection(240.0f, 480.0f, "center", 100), 0, 100);
    assert(center_stop.action == "stop");

    // Required walking sequence: the same centre obstacle progresses from
    // FAR -> MID -> NEAR, then the user probes until it moves to the right.
    // The action must be CLEAR -> SLOW -> STOP -> LEFT.  A deliberately tiny
    // safety bound and TTC prove that neither hidden field can stop a far
    // target anymore; the displayed distance is the action distance.
    obstacle::AvoidancePlanner staged_planner;
    staged_planner.Initialize({720, 1280});
    DetectionResult staged = planner_detection(240.0f, 480.0f, "center", 100);
    staged.items[0].distance_m = 2.60f;
    staged.items[0].safe_distance_m = 0.60f;
    staged.items[0].risk_level = "urgent";
    staged.items[0].ttc_s = 0.10f;
    assert(staged_planner.Update(staged, 0, 100).action == "clear");

    staged.timestamp_ms = 200;
    staged.items[0].distance_m = 1.80f;
    assert(staged_planner.Update(staged, 0, 200).action == "slow");

    staged.timestamp_ms = 300;
    staged.items[0].distance_m = 1.00f;
    assert(staged_planner.Update(staged, 0, 300).action == "stop");

    staged.timestamp_ms = 400;
    staged.items[0].box = {400.0f, 650.0f, 680.0f, 1240.0f};
    staged.items[0].sector = "right";
    const AvoidanceDecision probe_turn = staged_planner.Update(staged, 0, 400);
    assert(probe_turn.action == "turn_left");
    assert(probe_turn.recommended_direction == "left");
    assert(probe_turn.hazard_position == "RIGHT");

    // An unknown road mask is not positive evidence that the selected escape
    // side is blocked.  Preserve the object's turn unless that side has a
    // persistent blocked/step hazard.
    SurfaceResult unknown_turn_surface;
    unknown_turn_surface.valid = true;
    unknown_turn_surface.stale = false;
    unknown_turn_surface.primary_hazard = "unknown_other";
    AvoidanceDecision turn_detection = right_avoid;
    turn_detection.action = "turn_left";
    assert(fusion.Fuse(turn_detection, unknown_turn_surface, 300).action == "turn_left");

    // A close person on the right can make the background segmentation label
    // the empty left image area as blocked.  The named side-object action must
    // survive that generic mask; only persistent step/drop evidence vetoes it.
    SurfaceResult wall_like_left = unknown_turn_surface;
    wall_like_left.left.blocked_persistent = true;
    wall_like_left.left.safe_candidate = false;
    turn_detection.hazard_sector = "right";
    assert(fusion.Fuse(turn_detection, wall_like_left, 350).action == "turn_left");
    wall_like_left.left.persistent_hazard = true;
    assert(fusion.Fuse(turn_detection, wall_like_left, 360).action != "turn_left");

    obstacle::SurfaceSegmenter segmenter;
    SurfaceResult surface;
    const std::vector<float> hazard = segmentation_logits(true);
    const std::vector<float> depth = depth_logits(10);
    assert(segmenter.PostprocessLogits(hazard.data(), hazard.size(),
                                       depth.data(), depth.size(), true, 100, &surface));
    assert(!surface.center.persistent_hazard);

    // A wall/blocked surface is also temporal: it must not redirect the user
    // from a single noisy frame, but three of four observations latch it.
    obstacle::SurfaceSegmenter blocked_segmenter;
    SurfaceResult blocked_surface;
    const std::vector<float> blocked = blocked_segmentation_logits();
    for (int i = 0; i < 3; ++i) {
        assert(blocked_segmenter.PostprocessLogits(
            blocked.data(), blocked.size(), depth.data(), depth.size(), true,
            3000 + i * 100, &blocked_surface));
        assert(!blocked_surface.center.blocked_persistent);
    }
    assert(blocked_segmenter.PostprocessLogits(
        blocked.data(), blocked.size(), depth.data(), depth.size(), true,
        3300, &blocked_surface));
    assert(blocked_surface.center.blocked_persistent);
    assert(blocked_surface.primary_hazard == "blocked_surface");
    // 生产默认关闭 blocked_surface 对动作的覆盖：墙面分割仍可诊断，地面
    // 误分却不能再让一个无目标画面持续 SLOW。台阶测试在后文仍保持生效。
    const AvoidanceDecision blocked_fused =
        fusion.Fuse(detection, blocked_surface, 3300);
    assert(blocked_fused.action == "clear");
    assert(segmenter.PostprocessLogits(hazard.data(), hazard.size(),
                                       depth.data(), depth.size(), true, 200, &surface));
    assert(!surface.center.persistent_hazard);
    assert(segmenter.PostprocessLogits(hazard.data(), hazard.size(),
                                       depth.data(), depth.size(), true, 300, &surface));
    assert(!surface.center.persistent_hazard);
    assert(segmenter.PostprocessLogits(hazard.data(), hazard.size(),
                                       depth.data(), depth.size(), true, 400, &surface));
    assert(surface.center.persistent_hazard);
    assert(surface.primary_hazard == "step_or_drop");
    assert(surface.depth_level != "unknown");

    // Segmentation-only evidence is useful for diagnostics, but cannot be a
    // confirmed unified stair without edge and depth corroboration.
    assert(surface.stair_state == STAIR_NONE);

    const std::vector<float> clear = segmentation_logits(false);
    for (int i = 0; i < 3; ++i) {
        assert(segmenter.PostprocessLogits(clear.data(), clear.size(),
                                           depth.data(), depth.size(), true,
                                           500 + i * 100, &surface));
        assert(surface.center.persistent_hazard);
    }
    assert(segmenter.PostprocessLogits(clear.data(), clear.size(),
                                       depth.data(), depth.size(), true, 800, &surface));
    assert(!surface.center.persistent_hazard);

    // A whole-frame/all-corridor STEP prediction is a known coarse-mask
    // failure mode. Segmentation alone must not latch it as a real stair.
    obstacle::SurfaceSegmenter overfill_segmenter;
    SurfaceResult overfill_surface;
    const std::vector<float> overfill = uniform_segmentation_logits(STEP_OR_DROP);
    for (int i = 0; i < 6; ++i) {
        assert(overfill_segmenter.PostprocessLogits(
            overfill.data(), overfill.size(), depth.data(), depth.size(), true,
            900 + i * 100, &overfill_surface));
        assert(!overfill_surface.center.persistent_hazard);
        assert(overfill_surface.primary_hazard != "step_or_drop");
    }

    // The final E3 segmentation contract has four classes.  UNKNOWN_OTHER is
    // never a traversable path and must conservatively slow a clear detector.
    obstacle::SurfaceSegmenter unknown_segmenter;
    SurfaceResult unknown_surface;
    const std::vector<float> unknown_seg = uniform_segmentation_logits(UNKNOWN_OTHER);
    assert(unknown_segmenter.PostprocessLogits(
        unknown_seg.data(), unknown_seg.size(), depth.data(), depth.size(), true, 700,
        &unknown_surface));
    assert(unknown_surface.center.unknown_ratio > 0.99f);
    assert(!unknown_surface.center.safe_candidate);
    assert(unknown_surface.primary_hazard == "unknown_other");
    assert(unknown_surface.depth_level == "unknown");
    assert(unknown_surface.depth_ambiguous);
    const AvoidanceDecision unknown_fused = fusion.Fuse(detection, unknown_surface, 700);
    assert(unknown_fused.action == "slow");
    assert(unknown_fused.hazard_type == "unknown_other");

    // Ambiguous road depth cannot erase a reliable near object warning.
    AvoidanceDecision near_detection = detection;
    near_detection.depth_level = "near";
    near_detection.depth_confidence = 0.8f;
    near_detection.depth_ambiguous = false;
    near_detection.depth_source = "geometry";
    const AvoidanceDecision near_unknown_fused =
        fusion.Fuse(near_detection, unknown_surface, 701);
    assert(near_unknown_fused.depth_level == "near");
    assert(near_unknown_fused.depth_source == "geometry");

    // Equal evidence on opposite sides of a NEAR/MID boundary is explicitly
    // unknown.  A single strongly separated group is accepted immediately.
    obstacle::SurfaceSegmenter ambiguous_segmenter;
    SurfaceResult ambiguous_surface;
    const std::vector<float> ambiguous = ambiguous_depth_logits(6, 8);
    assert(ambiguous_segmenter.PostprocessLogits(
        clear.data(), clear.size(), ambiguous.data(), ambiguous.size(), true, 800,
        &ambiguous_surface));
    assert(ambiguous_surface.depth_level == "unknown");
    assert(ambiguous_surface.depth_ambiguous);
    assert(ambiguous_surface.depth_margin < 0.20f);

    obstacle::SurfaceSegmenter confident_segmenter;
    SurfaceResult confident_surface;
    for (int i = 0; i < 5; ++i) {
        assert(confident_segmenter.PostprocessLogits(
            clear.data(), clear.size(), depth.data(), depth.size(), true,
            1600 + i * 100, &confident_surface));
    }
    assert(confident_surface.depth_level == "far");
    assert(!confident_surface.depth_ambiguous);
    assert(confident_surface.depth_margin >= 0.20f);

    // HWC and CHW output layouts must decode to the same semantic result.
    obstacle::SurfaceSegmenter chw_segmenter;
    SurfaceResult chw_surface;
    const std::vector<float> clear_chw = hwc_to_chw(clear, SURFACE_CLASS_COUNT);
    const std::vector<float> depth_chw = hwc_to_chw(depth, DEPTH_BIN_COUNT);
    for (int i = 0; i < 5; ++i) {
        assert(chw_segmenter.PostprocessLogits(
            clear_chw.data(), clear_chw.size(), depth_chw.data(), depth_chw.size(), false,
            2200 + i * 100, &chw_surface));
    }
    assert(chw_surface.center.safe_candidate == confident_surface.center.safe_candidate);
    assert(chw_surface.depth_level == confident_surface.depth_level);
    assert(!chw_segmenter.PostprocessLogits(
        clear_chw.data(), clear_chw.size() - 1U,
        depth_chw.data(), depth_chw.size(), false, 1100, &chw_surface));

    // The production model packs segmentation, depth and stair edge into one
    // 21-channel output; semantic, horizontal-edge and depth-jump evidence must
    // persist for four LOWER observations before a confirmed stair is exposed.
    obstacle::SurfaceSegmenter packed_segmenter;
    SurfaceResult packed_surface;
    const std::vector<float> packed = packed_scene_logits(true, 10, true);
    assert(packed_segmenter.PostprocessPackedLogits(
        packed.data(), packed.size(), true, 1200, &packed_surface));
    assert(!packed_surface.stair_edge_persistent);
    assert(packed_segmenter.PostprocessPackedLogits(
        packed.data(), packed.size(), true, 1300, &packed_surface));
    assert(packed_surface.stair_state == STAIR_SUSPECTED);
    for (int i = 2; i < 4; ++i) {
        assert(packed_segmenter.PostprocessPackedLogits(
            packed.data(), packed.size(), true, 1200 + i * 100, &packed_surface));
    }
    assert(packed_surface.stair_edge_persistent);
    assert(packed_surface.stair_state == STAIR_CONFIRMED);
    assert(packed_surface.stair_edge_count > 0);
    assert(packed_surface.stair_edge_peak >= 0.55f);
    assert(packed_surface.stair_edge_span_ratio >= 0.45f);
    assert(packed_surface.stair_depth_jump_bins >= 2.0f);
    assert(packed_surface.stair_box_valid);
    assert(packed_surface.stair_edge_x2_norm > packed_surface.stair_edge_x1_norm);

    packed_surface.depth_level = "mid";
    packed_surface.depth_ambiguous = false;
    AvoidanceDecision stair_fused = fusion.Fuse(detection, packed_surface, 1600);
    assert(stair_fused.action == "stop");

    // A bed edge or chair back supplies one horizontal edge but neither a
    // bounded STEP region nor a depth discontinuity. It must never confirm.
    obstacle::SurfaceSegmenter furniture_edge_segmenter;
    SurfaceResult furniture_edge_surface;
    const std::vector<float> furniture_edge =
        packed_scene_logits(false, 10, true, false);
    for (int i = 0; i < 8; ++i) {
        assert(furniture_edge_segmenter.PostprocessPackedLogits(
            furniture_edge.data(), furniture_edge.size(), true,
            2000 + i * 100, &furniture_edge_surface));
        assert(furniture_edge_surface.stair_state != STAIR_CONFIRMED);
        assert(!furniture_edge_surface.stair_edge_persistent);
    }

    // If a candidate stair edge lies inside a stable person/furniture box, the
    // confirmation is conservatively downgraded to suspected.
    DetectionResult occluding_objects;
    DetectionItem chair;
    chair.class_id = obstacle::semantic::CHAIR_SEAT;
    chair.raw_label = "chair";
    chair.quality = "good";
    chair.box = {180.0f, 700.0f, 540.0f, 1260.0f};
    occluding_objects.items.push_back(chair);
    fusion.ApplyObjectOcclusion(occluding_objects, &packed_surface);
    assert(packed_surface.stair_edge_occluded_by_object);
    assert(packed_surface.stair_state == STAIR_SUSPECTED);
    assert(fusion.Fuse(detection, packed_surface, 1700).action == "slow");

    // OSD, UART and voice consume one stabilized decision. NEAR enters after
    // 2/3 evidence, requires 4/5 non-NEAR votes to leave, and STOP requires
    // four lower-risk observations to release.
    obstacle::GuidanceStabilizer guidance;
    AvoidanceDecision raw;
    raw.action = "clear";
    raw.cause = "PATH";
    raw.range = "FAR";
    raw.hazard_sector = "center";
    raw.object_label = "NONE";
    raw.scene_label = "PATH";
    raw.ai_ok = true;
    guidance.Update(raw, 0);
    raw.action = "stop";
    raw.cause = "OBJECT";
    raw.object_label = "PERSON";
    raw.range = "NEAR";
    raw.primary_class = "person";
    raw.distance_estimate_m = 1.10f;
    raw.center.occupied = true;
    raw.center.raw_label = "person";
    raw.center.distance_estimate_m = 1.10f;
    raw.center.safe_distance_m = 0.95f;
    raw.center.risk_level = "urgent";
    guidance.Update(raw, 100);
    const StableGuidance& near_stop = guidance.Update(raw, 200);
    assert(near_stop.action == "stop");
    assert(near_stop.range == "NEAR");
    assert(near_stop.primary_class == "person");
    assert(near_stop.center.object_class == "person");
    assert(std::fabs(near_stop.distance_estimate_m - 1.10f) < 0.01f);
    raw.action = "slow";
    raw.range = "MID";
    for (int i = 0; i < 3; ++i) {
        assert(guidance.Update(raw, 300 + i * 100).action == "stop");
    }
    assert(guidance.Update(raw, 600).action == "slow");
    assert(guidance.Current().range == "MID");

    // The planner has already stabilized a lateral escape.  Publishing it
    // through a second two-frame gate left the HUD and voice stuck at SLOW, so
    // a turn from SLOW is intentionally immediate here.
    obstacle::GuidanceStabilizer direction_guidance;
    AvoidanceDecision direction_raw;
    direction_raw.action = "slow";
    direction_raw.cause = "WARNING_RANGE";
    direction_raw.range = "MID";
    direction_raw.hazard_sector = "right";
    direction_raw.ai_ok = true;
    direction_guidance.Update(direction_raw, 0);
    direction_raw.action = "turn_left";
    direction_raw.cause = "RIGHT_PRIMARY_WARNING";
    assert(direction_guidance.Update(direction_raw, 100).action == "turn_left");
    direction_raw.cause = "RIGHT_WARNING_DIRECT";
    assert(direction_guidance.Update(direction_raw, 200).action == "turn_left");

    // A prior conservative STOP is released as soon as the already-stabilized
    // planner identifies a lateral escape.  Keeping another four-frame release
    // gate made both the HUD and speech repeat STOP after the person moved right.
    obstacle::GuidanceStabilizer stopped_direction_guidance;
    AvoidanceDecision stopped_direction_raw = direction_raw;
    stopped_direction_raw.action = "stop";
    stopped_direction_guidance.Update(stopped_direction_raw, 0);
    stopped_direction_guidance.Update(stopped_direction_raw, 100);
    stopped_direction_raw.action = "turn_left";
    assert(stopped_direction_guidance.Update(stopped_direction_raw, 200).action ==
           "turn_left");

    // The public distance filter must be scoped to one track. Switching from a
    // near person to a far chair cannot retain the old person's history.
    raw.action = "slow";
    raw.cause = "OBJECT";
    raw.primary_class = "person";
    raw.nearest_track_id = 11;
    raw.distance_estimate_m = 1.0f;
    guidance.Update(raw, 700);
    guidance.Update(raw, 800);
    raw.primary_class = "chair";
    raw.nearest_track_id = 12;
    raw.distance_estimate_m = 3.5f;
    const StableGuidance& switched_range = guidance.Update(raw, 900);
    assert(std::fabs(switched_range.distance_estimate_m - 3.5f) < 0.01f);

    // A covered camera is an immediate self-contained protection state. No
    // stale target, zone or metric distance may leak into the fault packet.
    raw.action = "system_fault";
    raw.ai_ok = false;
    const StableGuidance& fault = guidance.Update(raw, 1000);
    assert(fault.action == "system_fault");
    assert(fault.primary_class == "abnormal");
    assert(fault.object_label == "NONE");
    assert(fault.distance_estimate_m < 0.0f);
    assert(!fault.left.occupied && !fault.center.occupied && !fault.right.occupied);

    // Near-field occupancy is a conservative planning bound, not a discrete
    // metric reading. The displayed mean remains continuous while safe range
    // may be capped for collision avoidance.
    obstacle::RangingEstimator ranging;
    ranging.Initialize({720, 1280});
    DetectionItem near_chair;
    near_chair.raw_class_id = 1;
    near_chair.class_id = obstacle::semantic::CHAIR_SEAT;
    near_chair.raw_label = "chair";
    near_chair.label = "chair";
    near_chair.score = 0.80f;
    near_chair.quality = "good";
    near_chair.box = {200.0f, 500.0f, 520.0f, 1255.0f};
    ranging.Estimate(&near_chair);
    assert(near_chair.distance_m > 0.45f);
    assert(near_chair.safe_distance_m <= 0.451f);
    assert(near_chair.distance_source != "nearfield_cap");

    // 现场比例修正只作用于可解释的几何均值与不确定度。它不能改写相机
    // 高度/俯角，也不能把近场上界伪装成精确读数。
    setenv("A1_RANGE_GEOMETRY_SCALE", "1.00", 1);
    obstacle::RangingEstimator unscaled_ranging;
    unscaled_ranging.Initialize({720, 1280});
    DetectionItem reference_chair;
    reference_chair.raw_class_id = 1;
    reference_chair.class_id = obstacle::semantic::CHAIR_SEAT;
    reference_chair.raw_label = "chair";
    reference_chair.label = "chair";
    reference_chair.score = 0.80f;
    reference_chair.quality = "good";
    reference_chair.box = {210.0f, 560.0f, 510.0f, 900.0f};
    unscaled_ranging.Estimate(&reference_chair);
    const float unscaled_distance = reference_chair.distance_m;
    assert(unscaled_distance > 0.0f);
    setenv("A1_RANGE_GEOMETRY_SCALE", "1.60", 1);
    obstacle::RangingEstimator scaled_ranging;
    scaled_ranging.Initialize({720, 1280});
    scaled_ranging.Estimate(&reference_chair);
    assert(reference_chair.distance_m > unscaled_distance * 1.45f);
    assert(reference_chair.distance_m < unscaled_distance * 1.75f);
    // 1.60 倍仅校正展示均值；规划下界默认不得随之放大，否则 2m 左右
    // 的原始几何证据会被错误释放到 FAR/CLEAR。
    assert(reference_chair.safe_distance_m < reference_chair.distance_m / 1.35f);

    // Indoor8 局部人体不包含脚时不得套用 1.70m 全身高度和 1.60 地面比例。
    // 方形上半身框应使用可见宽度，距离必须对框宽变化明显响应。
    DetectionItem partial_person;
    partial_person.raw_class_id = 0;
    partial_person.class_id = obstacle::semantic::PERSON;
    partial_person.raw_label = "person";
    partial_person.label = "person";
    partial_person.score = 0.80f;
    partial_person.quality = "good";
    partial_person.box = {350.0f, 260.0f, 680.0f, 690.0f};
    scaled_ranging.Estimate(&partial_person);
    assert(partial_person.distance_source == "person_partial_width");
    assert(partial_person.distance_m > 0.50f);
    assert(partial_person.distance_m < 1.30f);
    // 框已占画面近一半宽度时，即使宽度先验存在较大方差，规划器仍须
    // 保留近场占用上界，不能因为展示均值或滤波历史而释放为 FAR。
    assert(partial_person.safe_distance_m <= 0.951f);
    const float close_partial_distance = partial_person.distance_m;
    partial_person.box = {430.0f, 300.0f, 610.0f, 520.0f};
    scaled_ranging.Estimate(&partial_person);
    assert(partial_person.distance_m > close_partial_distance * 1.45f);

    // Indoor8 must not inherit the legacy ROD25 person-part bridge. One chair
    // observation is ignored as noise, while two consecutive high-confidence
    // chair observations correct an old PERSON track.
    obstacle::ObstacleTracker class_tracker;
    class_tracker.Initialize({720, 1280});
    SurfaceResult tracker_surface;
    tracker_surface.valid = true;
    tracker_surface.stale = false;
    class_tracker.SetSurfaceResult(tracker_surface);
    auto raw_detection = [](int raw_class, float score, int64_t timestamp) {
        DetectionResult frame;
        frame.view_id = 0;
        frame.roi = {0, 0, 720, 1280};
        frame.timestamp_ms = timestamp;
        DetectionItem item;
        item.box = {210.0f, 600.0f, 510.0f, 1180.0f};
        item.score = score;
        item.raw_class_id = raw_class;
        item.raw_label = obstacle::semantic::RawLabel(raw_class);
        item.class_id = obstacle::semantic::SemanticClassFromRaw(raw_class);
        item.label = obstacle::semantic::SemanticLabel(item.class_id);
        item.semantic_class = item.label;
        item.quality = "good";
        frame.items.push_back(item);
        return frame;
    };
    class_tracker.Update(raw_detection(0, 0.75f, 100), 1);
    class_tracker.Update(raw_detection(0, 0.75f, 200), 2);
    assert(!class_tracker.StableResult().items.empty());
    assert(class_tracker.StableResult().items[0].raw_label == "person");
    class_tracker.Update(raw_detection(1, 0.65f, 300), 3);
    assert(class_tracker.StableResult().items[0].raw_label == "person");
    class_tracker.Update(raw_detection(1, 0.65f, 400), 4);
    assert(class_tracker.StableResult().items[0].raw_label == "chair");

    // A lower-confidence correction follows the ordinary three-observation
    // rule; it must not take the two-frame high-confidence shortcut.
    obstacle::ObstacleTracker ordinary_switch_tracker;
    ordinary_switch_tracker.Initialize({720, 1280});
    ordinary_switch_tracker.SetSurfaceResult(tracker_surface);
    ordinary_switch_tracker.Update(raw_detection(0, 0.75f, 100), 1);
    ordinary_switch_tracker.Update(raw_detection(0, 0.75f, 200), 2);
    ordinary_switch_tracker.Update(raw_detection(1, 0.40f, 300), 3);
    ordinary_switch_tracker.Update(raw_detection(1, 0.40f, 400), 4);
    assert(ordinary_switch_tracker.StableResult().items[0].raw_label == "person");
    ordinary_switch_tracker.Update(raw_detection(1, 0.40f, 500), 5);
    assert(ordinary_switch_tracker.StableResult().items[0].raw_label == "chair");

    obstacle::ObstacleTracker noise_tracker;
    noise_tracker.Initialize({720, 1280});
    noise_tracker.SetSurfaceResult(tracker_surface);
    noise_tracker.Update(raw_detection(0, 0.75f, 100), 1);
    noise_tracker.Update(raw_detection(0, 0.75f, 200), 2);
    noise_tracker.Update(raw_detection(1, 0.65f, 300), 3);
    noise_tracker.Update(raw_detection(0, 0.75f, 400), 4);
    assert(noise_tracker.StableResult().items[0].raw_label == "person");

    // Three reliable anchors establish a scale.  The public depth grid may then
    // affect an internal safe distance, while the UI still exposes only levels.
    surface.valid = true;
    surface.stale = false;
    surface.labels.fill(GROUND_CANDIDATE);
    surface.depth_m.fill(2.0f);
    surface.depth_cell_confidence.fill(0.8f);
    DetectionResult detections;
    DetectionItem person;
    person.raw_class_id = 0;
    person.box = {250.0f, 760.0f, 470.0f, 1220.0f};
    person.distance_m = 1.5f;
    person.safe_distance_m = 1.35f;
    person.distance_confidence = 0.8f;
    person.quality = "good";
    detections.items.push_back(person);
    obstacle::DepthRangeFusion depth_fusion;
    depth_fusion.Initialize({720, 1280});
    depth_fusion.Apply(&detections, &surface);
    depth_fusion.Apply(&detections, &surface);
    depth_fusion.Apply(&detections, &surface);
    assert(detections.items[0].depth_source == "fused");
    assert(detections.items[0].depth_consistent);
    assert(detections.items[0].depth_level == "mid");

    std::cout << "surface-depth logic tests passed" << std::endl;
    return 0;
}
