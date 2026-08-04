#pragma once

#include "common.hpp"

#include <array>
#include <deque>

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

/**
 * @brief Associates the cached 64x64 monocular-depth grid with stable detector
 * tracks and calibrates its scale only from conservative geometric anchors.
 *
 * The public RGB-D model is never treated as a centimetre sensor.  Without a
 * reliable anchor it only assigns near/mid/far.  A metric value may influence
 * the planner only after several recent person/furniture anchors agree.
 */
class DepthRangeFusion {
public:
    DepthRangeFusion();
    void Initialize(const std::array<int, 2>& image_shape);
    void Apply(DetectionResult* result, SurfaceResult* surface);

private:
    bool SampleBox(const DetectionItem& item,
                   const SurfaceResult& surface,
                   float* depth_m,
                   float* confidence) const;
    bool IsReliableAnchor(const DetectionItem& item) const;
    void AddScaleAnchor(float scale);
    float StableScale() const;
    static std::string LevelFromDepth(float depth_m);

    std::array<int, 2> image_shape_;
    std::deque<float> scale_anchors_;
};

}  // namespace obstacle
