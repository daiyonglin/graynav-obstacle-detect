#include "../include/guidance_stabilizer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <vector>

namespace obstacle {
namespace {

std::string upper_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
}

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string direction_for_action(const std::string& action)
{
    if (action == "turn_left") return "left";
    if (action == "turn_right") return "right";
    if (action == "stop" || action == "system_fault") return "hold";
    return "forward";
}

std::string position_for_sector(const std::string& sector)
{
    const std::string upper = upper_copy(sector);
    if (upper == "LEFT") return "LEFT";
    if (upper == "RIGHT") return "RIGHT";
    if (upper == "MULTI") return "MULTI";
    if (upper == "BLOCKED" || upper == "WIDE") return "BLOCKED";
    return "FRONT";
}

std::string risk_for(const AvoidanceDecision& raw, const std::string& action)
{
    if (!raw.ai_ok || action == "system_fault") return "FAULT";
    if (action == "stop") return "URGENT";
    if (action == "clear") return "SAFE";
    if (upper_copy(raw.risk) == "UNKNOWN" || upper_copy(raw.cause) == "UNKNOWN") {
        return "UNKNOWN";
    }
    return "WARNING";
}

std::string distance_identity_for(const AvoidanceDecision& raw)
{
    const std::string cause = upper_copy(raw.cause);
    if (cause == "STAIR" || cause == "STEP_CHECK" || cause == "BLOCKED") {
        return "scene:" + cause + ":" + upper_copy(raw.hazard_position);
    }
    if (raw.nearest_track_id >= 0) {
        return "track:" + std::to_string(raw.nearest_track_id) + ":" +
            lower_copy(raw.primary_class);
    }
    return "scene:" + cause + ":" + upper_copy(raw.hazard_position);
}

GuidanceZone guidance_zone(const ZoneStatus& zone)
{
    GuidanceZone out;
    out.occupied = zone.occupied;
    out.object_class = zone.occupied
        ? lower_copy(zone.raw_label.empty() ? zone.label : zone.raw_label) : "clear";
    if (out.object_class.empty()) out.object_class = zone.occupied ? "obstacle" : "clear";
    out.distance_estimate_m = zone.distance_estimate_m;
    out.safe_distance_m = zone.safe_distance_m >= 0.0f
        ? zone.safe_distance_m : zone.distance_m;
    out.risk = zone.occupied ? upper_copy(zone.risk_level) : "SAFE";
    return out;
}

}  // namespace

GuidanceStabilizer::GuidanceStabilizer()
{
    Reset();
}

void GuidanceStabilizer::Reset()
{
    stable_ = StableGuidance();
    initialized_ = false;
    range_history_.clear();
    distance_history_.clear();
    distance_identity_.clear();
    pending_sector_.clear();
    pending_sector_count_ = 0;
    pending_action_.clear();
    pending_cause_.clear();
    pending_action_count_ = 0;
    stop_release_count_ = 0;
    last_object_seen_ms_ = -100000;
}

std::string GuidanceStabilizer::NormalizeRange(const std::string& value)
{
    const std::string upper = upper_copy(value);
    return upper == "NEAR" || upper == "MID" || upper == "FAR"
        ? upper : "UNKNOWN";
}

std::string GuidanceStabilizer::NormalizeSector(const std::string& value)
{
    const std::string upper = upper_copy(value);
    if (upper == "LEFT") return "left";
    if (upper == "RIGHT") return "right";
    if (upper == "MULTI") return "multi";
    if (upper == "BLOCKED" || upper == "WIDE") return "blocked";
    return "center";
}

int GuidanceStabilizer::CountRecent(const std::deque<std::string>& values,
                                    const std::string& value,
                                    size_t recent)
{
    int count = 0;
    const size_t begin = values.size() > recent ? values.size() - recent : 0U;
    for (size_t i = begin; i < values.size(); ++i) {
        if (values[i] == value) ++count;
    }
    return count;
}

std::string GuidanceStabilizer::Majority(const std::deque<std::string>& values,
                                         const std::string& fallback)
{
    if (values.empty()) return fallback;
    std::map<std::string, int> counts;
    for (const std::string& value : values) ++counts[value];
    std::string best = fallback;
    int best_count = -1;
    for (const auto& entry : counts) {
        if (entry.second > best_count ||
            (entry.second == best_count && entry.first == fallback)) {
            best = entry.first;
            best_count = entry.second;
        }
    }
    return best;
}

void GuidanceStabilizer::UpdateActionAndCause(const AvoidanceDecision& raw)
{
    const std::string candidate_action = raw.action;
    const std::string candidate_cause = upper_copy(raw.cause);
    const bool immediate_fault = candidate_action == "system_fault" || !raw.ai_ok;
    if (immediate_fault) {
        stable_.action = candidate_action;
        stable_.cause = candidate_cause;
        stable_.scene_label = "AI_FAIL";
        stable_.object_label = "NONE";
        stable_.primary_class = "abnormal";
        stable_.range = "UNKNOWN";
        stable_.sector = "center";
        stable_.hazard_position = "FRONT";
        stable_.left = GuidanceZone();
        stable_.center = GuidanceZone();
        stable_.right = GuidanceZone();
        stable_.distance_estimate_m = -1.0f;
        distance_history_.clear();
        distance_identity_.clear();
        pending_action_count_ = 0;
        stop_release_count_ = 0;
        return;
    }

    const bool stop_to_lateral_escape = stable_.action == "stop" &&
        (candidate_action == "turn_left" || candidate_action == "turn_right");
    if (stop_to_lateral_escape) {
        // The planner has already confirmed that the obstacle is lateral and
        // selected the opposite escape direction.  Do not apply another four
        // samples of STOP release hysteresis in the presentation layer.
        stable_.action = candidate_action;
        stable_.cause = candidate_cause;
        stable_.scene_label = upper_copy(raw.scene_label);
        pending_action_count_ = 0;
        stop_release_count_ = 0;
        return;
    }
    if (stable_.action == "stop" && candidate_action != "stop") {
        if (++stop_release_count_ < 4) return;
        stop_release_count_ = 0;
        stable_.action = candidate_action;
        stable_.cause = candidate_cause;
        stable_.scene_label = upper_copy(raw.scene_label);
        pending_action_count_ = 0;
        return;
    } else {
        stop_release_count_ = 0;
    }

    if (candidate_action == stable_.action) {
        pending_action_count_ = 0;
        // Cause is descriptive metadata.  It may legitimately alternate
        // between two planner rules while the user-facing action stays the
        // same; update it without restarting the action stability gate.
        stable_.cause = candidate_cause;
        stable_.scene_label = upper_copy(raw.scene_label);
        return;
    }
    if (candidate_action == pending_action_) {
        ++pending_action_count_;
        pending_cause_ = candidate_cause;
    } else {
        pending_action_ = candidate_action;
        pending_cause_ = candidate_cause;
        pending_action_count_ = 1;
    }

    // 台阶确认本身已包含较长时序，系统故障也已在上方立即处理；其余变化
    // 至少连续两帧才进入面向人的输出。STOP 同样要求两帧，避免单帧测距噪声。
    // The planner already applies temporal hysteresis.  Once it has converted
    // SLOW into an explicit lateral escape, publish that turn immediately;
    // applying a second two-frame gate here made alternating ROIs keep the HUD
    // and speech stuck at SLOW despite a persistent side obstacle.
    const bool lateral_escape_from_slow = stable_.action == "slow" &&
        (candidate_action == "turn_left" || candidate_action == "turn_right");
    const int needed = (candidate_cause == "STAIR" || lateral_escape_from_slow) ? 1 : 2;
    if (pending_action_count_ >= needed) {
        stable_.action = candidate_action;
        stable_.cause = pending_cause_;
        stable_.scene_label = upper_copy(raw.scene_label);
        pending_action_count_ = 0;
    }
}

void GuidanceStabilizer::UpdateRange(const std::string& candidate)
{
    range_history_.push_back(candidate);
    while (range_history_.size() > 5U) range_history_.pop_front();
    if (candidate == "NEAR" && CountRecent(range_history_, "NEAR", 3U) >= 2) {
        stable_.range = "NEAR";
        return;
    }
    if (stable_.range == "NEAR") {
        const int non_near = static_cast<int>(range_history_.size()) -
            CountRecent(range_history_, "NEAR", range_history_.size());
        if (range_history_.size() < 5U || non_near < 4) return;
    }
    const std::string majority = Majority(range_history_, stable_.range);
    if (CountRecent(range_history_, majority, range_history_.size()) >= 3) {
        stable_.range = majority;
    }
}

void GuidanceStabilizer::UpdateSector(const std::string& candidate)
{
    if (candidate == stable_.sector) {
        pending_sector_count_ = 0;
        return;
    }
    if (candidate == pending_sector_) ++pending_sector_count_;
    else {
        pending_sector_ = candidate;
        pending_sector_count_ = 1;
    }
    if (pending_sector_count_ >= 3) {
        stable_.sector = candidate;
        pending_sector_count_ = 0;
    }
}

void GuidanceStabilizer::UpdateObject(const AvoidanceDecision& raw, int64_t now_ms)
{
    const std::string candidate = upper_copy(raw.object_label);
    if (!candidate.empty() && candidate != "NONE" && raw.confidence >= 0.12f) {
        stable_.object_label = candidate;
        last_object_seen_ms_ = now_ms;
    } else if (now_ms - last_object_seen_ms_ > 600) {
        stable_.object_label = "NONE";
    }
}

void GuidanceStabilizer::UpdateDistance(const AvoidanceDecision& raw)
{
    if (!raw.ai_ok || stable_.action == "clear" || stable_.action == "system_fault") {
        stable_.distance_estimate_m = -1.0f;
        distance_history_.clear();
        distance_identity_.clear();
        return;
    }
    const float candidate = raw.distance_estimate_m;
    if (!(candidate > 0.0f) || !std::isfinite(candidate)) return;

    const std::string identity = distance_identity_for(raw);
    if (identity != distance_identity_) {
        distance_identity_ = identity;
        distance_history_.clear();
        stable_.distance_estimate_m = -1.0f;
    }
    distance_history_.push_back(candidate);
    while (distance_history_.size() > 3U) distance_history_.pop_front();
    std::vector<float> sorted(distance_history_.begin(), distance_history_.end());
    std::sort(sorted.begin(), sorted.end());
    float median = sorted[sorted.size() / 2U];
    if (sorted.size() % 2U == 0U) {
        median = 0.5f * (sorted[sorted.size() / 2U - 1U] + median);
    }
    if (stable_.distance_estimate_m > 0.0f && !raw.approaching) {
        median = std::max(stable_.distance_estimate_m * 0.70f,
            std::min(stable_.distance_estimate_m * 1.30f, median));
    }
    if (stable_.distance_estimate_m > 0.0f) {
        // Approach evidence must be reflected quickly for safety.  Increasing
        // distance is intentionally slower because a single shortened box can
        // otherwise make a nearby obstacle look suddenly far away.
        const bool getting_closer = median < stable_.distance_estimate_m;
        const float alpha = raw.approaching ? 0.65f : (getting_closer ? 0.50f : 0.30f);
        stable_.distance_estimate_m =
            (1.0f - alpha) * stable_.distance_estimate_m + alpha * median;
    } else {
        stable_.distance_estimate_m = median;
    }
}

const StableGuidance& GuidanceStabilizer::Update(const AvoidanceDecision& raw,
                                                  int64_t now_ms)
{
    if (!initialized_) {
        stable_.action = raw.action;
        stable_.cause = upper_copy(raw.cause);
        stable_.range = NormalizeRange(raw.range);
        stable_.sector = NormalizeSector(raw.hazard_sector);
        stable_.object_label = upper_copy(raw.object_label);
        stable_.scene_label = upper_copy(raw.scene_label);
        stable_.recommended_direction = direction_for_action(stable_.action);
        stable_.hazard_position = position_for_sector(stable_.sector);
        stable_.primary_class = lower_copy(raw.primary_class.empty()
            ? raw.object_label : raw.primary_class);
        if (stable_.primary_class.empty() || stable_.primary_class == "none") {
            stable_.primary_class = lower_copy(raw.cause);
        }
        stable_.risk = risk_for(raw, stable_.action);
        stable_.left = guidance_zone(raw.left);
        stable_.center = guidance_zone(raw.center);
        stable_.right = guidance_zone(raw.right);
        stable_.confidence = raw.confidence;
        stable_.ai_ok = raw.ai_ok;
        stable_.timestamp_ms = static_cast<uint64_t>(std::max<int64_t>(0, now_ms));
        last_object_seen_ms_ = stable_.object_label != "NONE" ? now_ms : -100000;
        initialized_ = true;
        range_history_.push_back(stable_.range);
        UpdateDistance(raw);
        return stable_;
    }

    UpdateActionAndCause(raw);
    if (stable_.action == "system_fault" || !raw.ai_ok) {
        stable_.recommended_direction = "hold";
        stable_.risk = "FAULT";
        stable_.confidence = raw.confidence;
        stable_.ai_ok = false;
        stable_.timestamp_ms = static_cast<uint64_t>(std::max<int64_t>(0, now_ms));
        return stable_;
    }
    UpdateRange(NormalizeRange(raw.range));
    UpdateSector(NormalizeSector(raw.hazard_sector));
    UpdateObject(raw, now_ms);
    stable_.recommended_direction = direction_for_action(stable_.action);
    stable_.hazard_position = position_for_sector(stable_.sector);
    const std::string candidate_class = lower_copy(raw.primary_class.empty()
        ? raw.object_label : raw.primary_class);
    if (!candidate_class.empty() && candidate_class != "none") {
        stable_.primary_class = candidate_class;
    } else if (stable_.action == "clear") {
        stable_.primary_class = "none";
    }
    stable_.risk = risk_for(raw, stable_.action);
    stable_.left = guidance_zone(raw.left);
    stable_.center = guidance_zone(raw.center);
    stable_.right = guidance_zone(raw.right);
    UpdateDistance(raw);
    stable_.confidence = 0.75f * stable_.confidence + 0.25f * raw.confidence;
    stable_.ai_ok = raw.ai_ok;
    stable_.timestamp_ms = static_cast<uint64_t>(std::max<int64_t>(0, now_ms));
    return stable_;
}

}  // namespace obstacle
