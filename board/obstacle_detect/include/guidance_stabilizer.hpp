#pragma once

#include "common.hpp"

#include <deque>
#include <string>

namespace obstacle {

/** @brief 将融合决策稳定成 OSD、串口和语音共用的唯一状态。 */
class GuidanceStabilizer {
public:
    GuidanceStabilizer();

    void Reset();
    const StableGuidance& Update(const AvoidanceDecision& raw, int64_t now_ms);
    const StableGuidance& Current() const { return stable_; }

private:
    static std::string NormalizeRange(const std::string& value);
    static std::string NormalizeSector(const std::string& value);
    static std::string Majority(const std::deque<std::string>& values,
                                const std::string& fallback);
    static int CountRecent(const std::deque<std::string>& values,
                           const std::string& value,
                           size_t recent);

    void UpdateActionAndCause(const AvoidanceDecision& raw);
    void UpdateRange(const std::string& candidate);
    void UpdateSector(const std::string& candidate);
    void UpdateObject(const AvoidanceDecision& raw, int64_t now_ms);

    StableGuidance stable_;
    bool initialized_;
    std::deque<std::string> range_history_;
    std::string pending_sector_;
    int pending_sector_count_;
    std::string pending_action_;
    std::string pending_cause_;
    int pending_action_count_;
    int stop_release_count_;
    int64_t last_object_seen_ms_;
};

}  // namespace obstacle
