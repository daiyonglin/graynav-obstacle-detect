#pragma once

#include "common.hpp"

namespace obstacle {

/** @brief 将稳定目标决策与道路分割走廊合并为唯一 AvoidanceDecision。 */
class SurfaceDecisionFusion {
public:
    AvoidanceDecision Fuse(const AvoidanceDecision& detection,
                           const SurfaceResult& surface,
                           int64_t now_ms) const;

private:
    static bool IsSafe(const SurfaceCorridor& corridor);
    static float SafeScore(const SurfaceCorridor& corridor);
};

}  // namespace obstacle
