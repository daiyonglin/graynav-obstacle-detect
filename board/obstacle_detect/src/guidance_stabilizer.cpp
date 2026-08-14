#include "../include/guidance_stabilizer.hpp"

#include <algorithm>
#include <cctype>
#include <map>

namespace obstacle {
namespace {

std::string upper_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return value;
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

    if (candidate_action == stable_.action && candidate_cause == stable_.cause) {
        pending_action_count_ = 0;
        stable_.scene_label = upper_copy(raw.scene_label);
        return;
    }
    if (candidate_action == pending_action_ && candidate_cause == pending_cause_) {
        ++pending_action_count_;
    } else {
        pending_action_ = candidate_action;
        pending_cause_ = candidate_cause;
        pending_action_count_ = 1;
    }

    // 台阶确认本身已包含较长时序，系统故障也已在上方立即处理；其余变化
    // 至少连续两帧才进入面向人的输出。STOP 同样要求两帧，避免单帧测距噪声。
    const int needed = candidate_cause == "STAIR" ? 1 : 2;
    if (pending_action_count_ >= needed) {
        stable_.action = candidate_action;
        stable_.cause = candidate_cause;
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
        stable_.confidence = raw.confidence;
        stable_.ai_ok = raw.ai_ok;
        stable_.timestamp_ms = static_cast<uint64_t>(std::max<int64_t>(0, now_ms));
        last_object_seen_ms_ = stable_.object_label != "NONE" ? now_ms : -100000;
        initialized_ = true;
        range_history_.push_back(stable_.range);
        return stable_;
    }

    UpdateActionAndCause(raw);
    UpdateRange(NormalizeRange(raw.range));
    UpdateSector(NormalizeSector(raw.hazard_sector));
    UpdateObject(raw, now_ms);
    stable_.confidence = 0.75f * stable_.confidence + 0.25f * raw.confidence;
    stable_.ai_ok = raw.ai_ok;
    stable_.timestamp_ms = static_cast<uint64_t>(std::max<int64_t>(0, now_ms));
    return stable_;
}

}  // namespace obstacle
