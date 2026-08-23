#include "../include/surface_fusion.hpp"

#include "../include/semantic_config.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace obstacle {
namespace {

float clampf(float value, float lo, float hi)
{
    return std::max(lo, std::min(hi, value));
}

float median(std::vector<float>* values)
{
    if (values == NULL || values->empty()) return -1.0f;
    const size_t middle = values->size() / 2;
    std::nth_element(values->begin(), values->begin() + middle, values->end());
    float result = (*values)[middle];
    if ((values->size() & 1U) == 0U) {
        std::nth_element(values->begin(), values->begin() + middle - 1, values->end());
        result = 0.5f * (result + (*values)[middle - 1]);
    }
    return result;
}

int level_severity(const std::string& level)
{
    if (level == "near") return 3;
    if (level == "mid") return 2;
    if (level == "far") return 1;
    return 0;
}

}  // namespace

bool SurfaceDecisionFusion::IsSafe(const SurfaceCorridor& corridor)
{
    return corridor.safe_candidate && !corridor.persistent_hazard;
}

float SurfaceDecisionFusion::SafeScore(const SurfaceCorridor& corridor)
{
    return corridor.ground_ratio - corridor.blocked_ratio -
           2.0f * corridor.step_ratio - 0.75f * corridor.unknown_ratio;
}

void SurfaceDecisionFusion::ApplyObjectOcclusion(
    const DetectionResult& detections,
    SurfaceResult* surface) const
{
    if (surface == NULL || surface->stair_state == STAIR_NONE ||
        surface->stair_edge_count <= 0) {
        return;
    }
    const float frame_width = 720.0f;
    const float lower_roi_top = 1280.0f - frame_width;
    const float edge_y = lower_roi_top + frame_width *
        static_cast<float>(surface->stair_edge_rows[0]) / SURFACE_GRID_SIZE;
    const float corridor_x1 = frame_width * 0.30f;
    const float corridor_x2 = frame_width * 0.70f;
    for (const DetectionItem& item : detections.items) {
        const bool named_rigid = item.class_id == semantic::PERSON ||
            semantic::IsFurnitureLikeSemantic(item.class_id);
        if (!named_rigid || item.quality == "coarse") continue;
        const float overlap = std::max(0.0f,
            std::min(item.box[2], corridor_x2) -
            std::max(item.box[0], corridor_x1));
        const bool spans_corridor = overlap / (corridor_x2 - corridor_x1) >= 0.35f;
        const bool contains_edge = edge_y >= item.box[1] - 12.0f &&
                                   edge_y <= item.box[3] + 12.0f;
        if (!spans_corridor || !contains_edge) continue;
        surface->stair_edge_occluded_by_object = true;
        if (surface->stair_state == STAIR_CONFIRMED) {
            surface->stair_state = STAIR_SUSPECTED;
            surface->stair_edge_persistent = false;
            surface->center.persistent_hazard = false;
            surface->primary_hazard = "possible_step";
            surface->primary_sector = "center";
        }
        return;
    }
}

AvoidanceDecision SurfaceDecisionFusion::Fuse(const AvoidanceDecision& detection,
                                               const SurfaceResult& surface,
                                               int64_t now_ms) const
{
    AvoidanceDecision fused = detection;
    (void)now_ms;
    if (surface.perception_degraded) {
        fused.perception_degraded = true;
        fused.perception_source = "detection_degraded_surface_depth";
        fused.hazard_type = surface.primary_hazard;
        fused.hazard_sector = surface.primary_sector;
        fused.surface_confidence = surface.confidence;
        return fused;
    }
    if (!surface.valid || surface.stale) {
        fused.perception_degraded = false;
        fused.perception_source = surface.stale ? "detection+surface_depth_stale" :
                                                 "detection+surface_depth_pending";
        fused.hazard_type = "unknown";
        fused.hazard_sector = "center";
        fused.surface_confidence = 0.0f;
        if (detection.action == "clear") {
            fused.action = "slow";
            fused.prompt += " fusion=surface_depth_unavailable_slow";
        }
        return fused;
    }

    const bool wall_guidance = semantic::WallGuidanceEnabled();
    fused.perception_degraded = false;
    fused.perception_source = "detection+surface_depth";
    fused.hazard_type = surface.primary_hazard;
    if (detection.action == "clear") {
        fused.hazard_sector = surface.primary_sector;
    }
    fused.surface_confidence = surface.confidence;
    // blocked/unknown 仍保留在 SurfaceResult 供诊断，但默认不能再把地面误分
    // 直接升级成 SLOW/STOP。台阶/落差始终独立生效。
    if (!wall_guidance && surface.primary_hazard == "blocked_surface") {
        fused.hazard_type = detection.hazard_type;
        fused.hazard_sector = detection.hazard_sector;
    }
    // Keep a reliable object/geometry level when the road model is ambiguous.
    // A valid SurfaceDepth level replaces it only when object depth is absent
    // or the road evidence is more dangerous.
    const bool accept_surface_depth = !surface.depth_ambiguous &&
        surface.depth_level != "unknown" &&
        (fused.depth_level == "unknown" ||
         level_severity(surface.depth_level) > level_severity(fused.depth_level));
    if (accept_surface_depth) {
        fused.depth_level = surface.depth_level;
        fused.depth_confidence = surface.depth_confidence;
        fused.depth_margin = surface.depth_margin;
        fused.depth_ambiguous = surface.depth_ambiguous;
        fused.depth_source = surface.depth_source;
        fused.depth_consistent = surface.depth_consistent;
        fused.approaching = fused.approaching || surface.approaching;
    }

    // System health and an urgent detection/TTC retain absolute priority.
    if (detection.action == "system_fault" || detection.action == "stop") {
        return fused;
    }

    const bool left_safe = wall_guidance
        ? IsSafe(surface.left) : !surface.left.persistent_hazard;
    const bool center_safe = wall_guidance
        ? IsSafe(surface.center) : !surface.center.persistent_hazard;
    const bool right_safe = wall_guidance
        ? IsSafe(surface.right) : !surface.right.persistent_hazard;
    // UNKNOWN is not the same as a measured hazard.  Reject a turn only when
    // the selected side contains a temporally persistent step or blocked
    // surface; otherwise preserve the object planner's side-avoidance action.
    const bool left_explicitly_unsafe = surface.left.persistent_hazard ||
        (wall_guidance && surface.left.blocked_persistent);
    const bool right_explicitly_unsafe = surface.right.persistent_hazard ||
        (wall_guidance && surface.right.blocked_persistent);
    const bool center_drop = surface.stair_state == STAIR_CONFIRMED &&
        surface.center.persistent_hazard && surface.stair_edge_persistent;
    const bool possible_step = surface.stair_state == STAIR_SUSPECTED;
    const bool center_blocked = wall_guidance &&
        surface.center.blocked_persistent;
    const bool center_unknown = !center_drop && !center_blocked &&
        (surface.center.unknown_ratio >= 0.30f ||
         (wall_guidance && !center_safe));
    const bool near_surface = surface.depth_level == "near";
    const bool named_object_side_escape =
        (detection.action == "turn_left" && detection.hazard_sector == "right") ||
        (detection.action == "turn_right" && detection.hazard_sector == "left") ||
        ((detection.action == "turn_left" || detection.action == "turn_right") &&
         detection.hazard_sector == "multi");
    const bool selected_side_has_step =
        (detection.action == "turn_left" && surface.left.persistent_hazard) ||
        (detection.action == "turn_right" && surface.right.persistent_hazard);

    std::string reason = "surface_clear";
    if (center_drop) {
        if (fused.depth_level == "near" || fused.depth_level == "mid") {
            fused.action = "stop";
            reason = "confirmed_step_near_mid_stop";
        } else {
            fused.action = "slow";
            reason = "confirmed_step_far_unknown_slow";
        }
    } else if (named_object_side_escape && !selected_side_has_step) {
        // A background wall-like mask commonly covers the visible image behind
        // a close person.  It must not erase an explicit object-centre based
        // side-avoidance action.  Confirmed step/drop evidence on the selected
        // side still retains veto power.
        fused.hazard_sector = detection.hazard_sector;
        reason = "named_object_side_escape_preserved";
    } else if (detection.action == "turn_left" && left_explicitly_unsafe) {
        fused.action = right_safe ? "turn_right" :
            (near_surface && right_explicitly_unsafe ? "stop" : "slow");
        reason = "surface_reject_left_turn";
    } else if (detection.action == "turn_right" && right_explicitly_unsafe) {
        fused.action = left_safe ? "turn_left" :
            (near_surface && left_explicitly_unsafe ? "stop" : "slow");
        reason = "surface_reject_right_turn";
    } else if (detection.action != "clear") {
        // 命名目标已经形成稳定导航动作时，人物/家具背后的 blocked mask
        // 不能覆盖成难以理解的“墙面阻挡”。
        fused.hazard_sector = detection.hazard_sector;
        reason = "detection_action_preserved";
    } else if (center_blocked) {
        if (near_surface && !left_safe && !right_safe) {
            fused.action = "stop";
            reason = "surface_wall_near";
        } else if (left_safe || right_safe) {
            if (left_safe && right_safe) {
                fused.action = SafeScore(surface.left) >= SafeScore(surface.right)
                    ? "turn_left" : "turn_right";
            } else {
                fused.action = left_safe ? "turn_left" : "turn_right";
            }
            reason = "surface_blocked_side_avoid";
        } else {
            fused.action = "slow";
            reason = "surface_blocked_uncertain_side";
        }
    } else if (possible_step) {
        fused.action = "slow";
        fused.hazard_type = "possible_step";
        fused.hazard_sector = "center";
        reason = "possible_step_slow";
    } else if (center_unknown) {
        fused.action = "slow";
        fused.hazard_type = fused.hazard_type == "none"
            ? "unknown_other" : fused.hazard_type;
        fused.hazard_sector = "center";
        reason = "surface_center_unknown";
    } else {
        fused.action = "clear";
    }

    std::ostringstream prompt;
    prompt << detection.prompt
           << " fusion=" << reason
           << " surface=" << fused.hazard_type
           << " sector=" << fused.hazard_sector
           << " depth=" << fused.depth_level;
    fused.prompt = prompt.str();
    return fused;
}

DepthRangeFusion::DepthRangeFusion()
    : image_shape_{720, 1280}
{
}

void DepthRangeFusion::Initialize(const std::array<int, 2>& image_shape)
{
    image_shape_ = image_shape;
    scale_anchors_.clear();
}

std::string DepthRangeFusion::LevelFromDepth(float depth_m)
{
    if (depth_m <= 0.0f || !std::isfinite(depth_m)) return "unknown";
    if (depth_m < 1.25f) return "near";
    if (depth_m < 2.20f) return "mid";
    return "far";
}

bool DepthRangeFusion::IsReliableAnchor(const DetectionItem& item) const
{
    if (item.distance_m <= 0.0f || item.distance_confidence < 0.35f ||
        item.quality == "coarse") {
        return false;
    }
    if (semantic::ModelClassCount() == 80) {
        return item.raw_class_id == 0 || item.raw_class_id == 13 ||
               item.raw_class_id == 56 || item.raw_class_id == 57 ||
               item.raw_class_id == 60;
    }
    if (semantic::ModelClassCount() == 25) {
        return item.raw_class_id == 3 || item.raw_class_id == 17 ||
               item.raw_class_id == 23;
    }
    if (semantic::ModelClassCount() == 8) {
        return item.raw_class_id == 0 || item.raw_class_id == 1 ||
               item.raw_class_id == 2 || item.raw_class_id == 6 ||
               item.raw_class_id == 7;
    }
    return false;
}

void DepthRangeFusion::AddScaleAnchor(float scale)
{
    if (!std::isfinite(scale) || scale < 0.35f || scale > 3.0f) return;
    scale_anchors_.push_back(scale);
    while (scale_anchors_.size() > 12U) scale_anchors_.pop_front();
}

float DepthRangeFusion::StableScale() const
{
    if (scale_anchors_.size() < 3U) return -1.0f;
    std::vector<float> values(scale_anchors_.begin(), scale_anchors_.end());
    return median(&values);
}

bool DepthRangeFusion::SampleBox(const DetectionItem& item,
                                 const SurfaceResult& surface,
                                 float* depth_m,
                                 float* confidence) const
{
    if (depth_m == NULL || confidence == NULL || !surface.valid || surface.stale) return false;
    const int frame_w = image_shape_[0];
    const int frame_h = image_shape_[1];
    const int roi_size = std::min(frame_w, frame_h);
    const int roi_y = frame_h - roi_size;

    const float x1f = clampf(item.box[0], 0.0f, static_cast<float>(frame_w - 1));
    const float x2f = clampf(item.box[2], 0.0f, static_cast<float>(frame_w - 1));
    const float lower_half = 0.5f * (item.box[1] + item.box[3]);
    const float y1f = clampf(lower_half, static_cast<float>(roi_y), static_cast<float>(frame_h - 1));
    const float y2f = clampf(item.box[3], static_cast<float>(roi_y), static_cast<float>(frame_h - 1));
    if (x2f <= x1f || y2f <= y1f) return false;

    const int gx1 = std::max(0, std::min(SURFACE_GRID_SIZE - 1,
        static_cast<int>(x1f * SURFACE_GRID_SIZE / std::max(1, roi_size))));
    const int gx2 = std::max(gx1, std::min(SURFACE_GRID_SIZE - 1,
        static_cast<int>(x2f * SURFACE_GRID_SIZE / std::max(1, roi_size))));
    const int gy1 = std::max(0, std::min(SURFACE_GRID_SIZE - 1,
        static_cast<int>((y1f - roi_y) * SURFACE_GRID_SIZE / std::max(1, roi_size))));
    const int gy2 = std::max(gy1, std::min(SURFACE_GRID_SIZE - 1,
        static_cast<int>((y2f - roi_y) * SURFACE_GRID_SIZE / std::max(1, roi_size))));

    std::vector<float> values;
    std::vector<float> confidences;
    for (int gy = gy1; gy <= gy2; ++gy) {
        for (int gx = gx1; gx <= gx2; ++gx) {
            const int index = gy * SURFACE_GRID_SIZE + gx;
            // Do not let a wall behind a small detected object dominate its range.
            if (surface.labels[index] == BLOCKED_SURFACE) continue;
            const float value = surface.depth_m[index];
            const float conf = surface.depth_cell_confidence[index];
            if (value > 0.0f && std::isfinite(value) && conf >= 0.20f) {
                values.push_back(value);
                confidences.push_back(conf);
            }
        }
    }
    if (values.size() < 3U) return false;
    *depth_m = median(&values);
    *confidence = median(&confidences);
    return *depth_m > 0.0f;
}

void DepthRangeFusion::Apply(DetectionResult* result, SurfaceResult* surface)
{
    if (result == NULL || surface == NULL) return;
    for (size_t i = 0; i < result->items.size(); ++i) {
        DetectionItem& item = result->items[i];
        item.approaching = item.approach_mps > 0.15f;
        float learned = -1.0f;
        float learned_conf = 0.0f;
        if (!SampleBox(item, *surface, &learned, &learned_conf)) {
            if (item.safe_distance_m > 0.0f) {
                item.depth_level = LevelFromDepth(item.safe_distance_m);
                item.depth_confidence = item.distance_confidence;
                item.depth_source = "geometry";
            }
            continue;
        }

        if (IsReliableAnchor(item)) AddScaleAnchor(item.distance_m / learned);
        const float scale = StableScale();
        const float scaled = scale > 0.0f ? learned * scale : learned;
        item.depth_level = LevelFromDepth(scaled);
        item.depth_confidence = scale > 0.0f
            ? clampf(0.35f + 0.55f * learned_conf, 0.0f, 0.90f)
            : clampf(0.65f * learned_conf, 0.0f, 0.40f);
        item.depth_source = scale > 0.0f ? "learned_scaled" : "learned_relative";

        const float geometry = item.distance_m;
        if (geometry > 0.0f && scale > 0.0f) {
            const float relative_difference = std::fabs(geometry - scaled) /
                std::max(0.30f, std::min(geometry, scaled));
            item.depth_consistent = relative_difference <= 0.40f;
            if (item.depth_consistent) {
                const float conservative = std::min(
                    item.safe_distance_m > 0.0f ? item.safe_distance_m : geometry,
                    scaled);
                item.safe_distance_m = conservative;
                item.depth_level = LevelFromDepth(conservative);
                item.depth_confidence = clampf(
                    0.5f * item.distance_confidence + 0.5f * item.depth_confidence,
                    0.0f, 0.95f);
                item.depth_source = "fused";
            } else {
                // Conflict is resolved toward safety, never by blind averaging.
                const float conservative = std::min(
                    item.safe_distance_m > 0.0f ? item.safe_distance_m : geometry,
                    scaled);
                item.safe_distance_m = conservative;
                item.depth_level = LevelFromDepth(conservative);
                item.depth_confidence = std::min(item.distance_confidence,
                                                 item.depth_confidence) * 0.5f;
                item.depth_source = "conflict";
            }
        } else if (geometry <= 0.0f && scale > 0.0f && scale_anchors_.size() >= 3U &&
                   item.depth_confidence >= 0.45f) {
            // Only a calibrated model is allowed to create an internal metric
            // planner value.  It remains hidden from Aurora and speech.
            item.distance_m = scaled;
            item.safe_distance_m = scaled * 0.85f;
            item.distance_confidence = item.depth_confidence * 0.75f;
            item.distance_source = "learned_scaled";
            item.risk_level = item.depth_level == "near" ? "near" :
                              item.depth_level == "mid" ? "warning" : "far";
        }
    }
    const float scale = StableScale();
    if (surface->valid && !surface->stale && !surface->depth_ambiguous &&
        surface->center_depth_m > 0.0f) {
        if (scale > 0.0f) {
            surface->center_depth_m *= scale;
            surface->depth_level = LevelFromDepth(surface->center_depth_m);
            surface->depth_source = "learned_scaled";
            surface->depth_consistent = true;
            surface->depth_confidence = clampf(surface->depth_confidence + 0.15f, 0.0f, 0.90f);
        } else if (surface->depth_level != "unknown") {
            surface->depth_source = "learned_relative";
            surface->depth_confidence = std::min(surface->depth_confidence, 0.40f);
        }
        surface->proximity = surface->depth_level;
    }
}

}  // namespace obstacle
