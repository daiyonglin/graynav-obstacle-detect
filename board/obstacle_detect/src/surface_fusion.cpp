#include "../include/surface_fusion.hpp"

#include <sstream>

namespace obstacle {

bool SurfaceDecisionFusion::IsSafe(const SurfaceCorridor& corridor)
{
    return corridor.safe_candidate && !corridor.persistent_hazard;
}

float SurfaceDecisionFusion::SafeScore(const SurfaceCorridor& corridor)
{
    return corridor.ground_ratio - corridor.blocked_ratio -
           2.0f * corridor.step_ratio - 2.5f * corridor.pothole_ratio;
}

AvoidanceDecision SurfaceDecisionFusion::Fuse(const AvoidanceDecision& detection,
                                               const SurfaceResult& surface,
                                               int64_t now_ms) const
{
    AvoidanceDecision fused = detection;
    (void)now_ms;
    if (surface.perception_degraded) {
        fused.perception_degraded = true;
        fused.perception_source = "detection_degraded_surface";
        fused.hazard_type = surface.primary_hazard;
        fused.hazard_sector = surface.primary_sector;
        fused.surface_confidence = surface.confidence;
        return fused;
    }
    if (!surface.valid || surface.stale) {
        fused.perception_degraded = false;
        fused.perception_source = surface.stale ? "detection+surface_stale" :
                                                 "detection+surface_pending";
        fused.hazard_type = "unknown";
        fused.hazard_sector = "center";
        fused.surface_confidence = 0.0f;
        if (detection.action != "system_fault" && detection.action != "stop" &&
            detection.action == "clear") {
            fused.action = "slow";
            fused.prompt += " fusion=surface_unavailable_slow";
        }
        return fused;
    }

    fused.perception_degraded = false;
    fused.perception_source = "detection+surface";
    fused.hazard_type = surface.primary_hazard;
    fused.hazard_sector = surface.primary_sector;
    fused.surface_confidence = surface.confidence;

    // 系统异常和检测 TTC 紧急停车拥有最高优先级。
    if (detection.action == "system_fault" || detection.action == "stop") {
        return fused;
    }

    const bool left_safe = IsSafe(surface.left);
    const bool center_safe = IsSafe(surface.center);
    const bool right_safe = IsSafe(surface.right);
    const bool center_drop = surface.center.persistent_hazard &&
        (surface.center.step_ratio >= 0.03f || surface.center.pothole_ratio >= 0.03f);
    const bool center_blocked = surface.center.blocked_ratio >= 0.35f;

    std::string reason = "surface_clear";
    if (center_drop) {
        if (left_safe || right_safe) {
            if (left_safe && right_safe) {
                fused.action = SafeScore(surface.left) >= SafeScore(surface.right)
                    ? "turn_left" : "turn_right";
            } else {
                fused.action = left_safe ? "turn_left" : "turn_right";
            }
            reason = "surface_drop_side_avoid";
        } else {
            fused.action = "stop";
            reason = "surface_drop_no_safe_side";
        }
    } else if (center_blocked) {
        if (left_safe || right_safe) {
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
    } else if (detection.action == "turn_left" && !left_safe) {
        fused.action = right_safe ? "turn_right" : "stop";
        reason = "surface_reject_left_turn";
    } else if (detection.action == "turn_right" && !right_safe) {
        fused.action = left_safe ? "turn_left" : "stop";
        reason = "surface_reject_right_turn";
    } else if (detection.action != "clear") {
        reason = "detection_action_preserved";
    } else if (!center_safe) {
        fused.action = "slow";
        fused.hazard_type = fused.hazard_type == "none" ? "unknown" : fused.hazard_type;
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
           << " proximity=" << surface.proximity;
    fused.prompt = prompt.str();
    return fused;
}

}  // namespace obstacle
