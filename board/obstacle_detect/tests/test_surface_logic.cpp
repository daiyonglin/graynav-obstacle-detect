#include "surface_fusion.hpp"
#include "surface_segmentation.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

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
        for (int y = 40; y < 49; ++y) {
            for (int x = 27; x < 37; ++x) {
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
    const std::vector<float> hazard = segmentation_logits(true);
    const std::vector<float> depth = depth_logits(10);
    assert(segmenter.PostprocessLogits(hazard.data(), hazard.size(),
                                       depth.data(), depth.size(), true, 100, &surface));
    assert(!surface.center.persistent_hazard);
    assert(segmenter.PostprocessLogits(hazard.data(), hazard.size(),
                                       depth.data(), depth.size(), true, 200, &surface));
    assert(surface.center.persistent_hazard);
    assert(surface.primary_hazard == "step_or_drop");
    assert(surface.depth_level != "unknown");

    surface.left.safe_candidate = true;
    surface.right.safe_candidate = false;
    AvoidanceDecision fused = fusion.Fuse(detection, surface, 200);
    assert(fused.action == "turn_left");
    assert(fused.perception_source == "detection+surface_depth");

    const std::vector<float> clear = segmentation_logits(false);
    for (int i = 0; i < 3; ++i) {
        assert(segmenter.PostprocessLogits(clear.data(), clear.size(),
                                           depth.data(), depth.size(), true,
                                           300 + i * 100, &surface));
        assert(surface.center.persistent_hazard);
    }
    assert(segmenter.PostprocessLogits(clear.data(), clear.size(),
                                       depth.data(), depth.size(), true, 600, &surface));
    assert(!surface.center.persistent_hazard);

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
