#include "surface_fusion.hpp"
#include "surface_segmentation.hpp"

#include <cassert>
#include <iostream>
#include <vector>

namespace {

std::vector<float> logits_with_center_step(bool hazard)
{
    std::vector<float> logits(32 * 32 * 4, -4.0f);
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            logits[(y * 32 + x) * 4] = 4.0f;
        }
    }
    if (hazard) {
        for (int y = 20; y < 25; ++y) {
            for (int x = 14; x < 19; ++x) {
                logits[(y * 32 + x) * 4] = -4.0f;
                logits[(y * 32 + x) * 4 + STEP_OR_DROP] = 4.0f;
            }
        }
    }
    return logits;
}

}  // namespace

int main()
{
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

    obstacle::SurfaceSegmenter segmenter;
    SurfaceResult surface;
    const std::vector<float> hazard = logits_with_center_step(true);
    assert(segmenter.PostprocessLogits(hazard.data(), hazard.size(), true, 100, &surface));
    assert(!surface.center.persistent_hazard);
    assert(segmenter.PostprocessLogits(hazard.data(), hazard.size(), true, 200, &surface));
    assert(surface.center.persistent_hazard);
    assert(surface.primary_hazard == "step_or_drop");

    surface.left.safe_candidate = true;
    surface.right.safe_candidate = false;
    AvoidanceDecision fused = fusion.Fuse(detection, surface, 200);
    assert(fused.action == "turn_left");
    assert(fused.perception_source == "detection+surface");

    const std::vector<float> clear = logits_with_center_step(false);
    for (int i = 0; i < 3; ++i) {
        assert(segmenter.PostprocessLogits(clear.data(), clear.size(), true, 300 + i * 100, &surface));
        assert(surface.center.persistent_hazard);
    }
    assert(segmenter.PostprocessLogits(clear.data(), clear.size(), true, 600, &surface));
    assert(!surface.center.persistent_hazard);
    std::cout << "surface logic tests passed" << std::endl;
    return 0;
}
