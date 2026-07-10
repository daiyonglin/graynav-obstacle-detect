#pragma once

#include "common.hpp"

#include <array>
#include <string>

namespace obstacle {

/**
 * Produces a conservative navigation action from tracked, ranged obstacles.
 * Turns are emitted only after both ROI views have recently observed the
 * scene and the selected side corridor has verified clearance.
 */
class AvoidancePlanner {
public:
    AvoidancePlanner();

    void Initialize(const std::array<int, 2>& image_shape);
    AvoidanceDecision Update(const DetectionResult& result, int view_id, int64_t timestamp_ms);

public:
    struct Corridor {
        ZoneStatus zone;
        float clearance;
        float min_ttc;
        bool verified;
        Corridor() : clearance(8.0f), min_ttc(-1.0f), verified(false) {}
    };

private:
    bool IsActionHazard(const DetectionItem& item) const;
    void AddToCorridor(Corridor* corridor, const DetectionItem& item) const;
    std::string StabilizeAction(const std::string& desired, int64_t now_ms);

    std::array<int, 2> image_shape_;
    int64_t last_view_ms_[2];
    std::string stable_action_;
    std::string pending_action_;
    int pending_count_;
    int64_t stable_since_ms_;
    int64_t pending_since_ms_;
};

}  // namespace obstacle
