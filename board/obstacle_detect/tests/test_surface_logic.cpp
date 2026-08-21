#include "surface_fusion.hpp"
#include "surface_segmentation.hpp"
#include "guidance_stabilizer.hpp"
#include "avoidance_planner.hpp"
#include "ranging.hpp"
#include "tracker.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace utils {
float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b)
{
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    return intersection / std::max(1.0f, area_a + area_b - intersection);
}
}  // namespace utils

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
    // The unified head uses the fixed indoor8 order, not ROD25 or COCO80 IDs.
    assert(obstacle::semantic::RawLabel(0) == "person");
    assert(obstacle::semantic::RawLabel(1) == "chair");
    assert(obstacle::semantic::RawLabel(2) == "dining_table");
    assert(obstacle::semantic::RawLabel(3) == "backpack");
    assert(obstacle::semantic::SemanticClassFromRaw(0) == obstacle::semantic::PERSON);
    assert(obstacle::semantic::SemanticClassFromRaw(1) == obstacle::semantic::CHAIR_SEAT);
    assert(obstacle::semantic::SemanticClassFromRaw(2) == obstacle::semantic::TABLE_DESK);
    assert(obstacle::semantic::SemanticClassFromRaw(3) == obstacle::semantic::BAG_SUITCASE);

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

    obstacle::AvoidancePlanner center_planner;
    center_planner.Initialize({720, 1280});
    const AvoidanceDecision center_stop = center_planner.Update(
        planner_detection(240.0f, 480.0f, "center", 100), 0, 100);
    assert(center_stop.action == "stop");

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
