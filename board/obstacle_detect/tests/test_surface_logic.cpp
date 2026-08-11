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

std::vector<float> ambiguous_depth_logits(int first_bin, int second_bin)
{
    std::vector<float> logits(SURFACE_GRID_CELLS * DEPTH_BIN_COUNT, -8.0f);
    for (int cell = 0; cell < SURFACE_GRID_CELLS; ++cell) {
        logits[cell * DEPTH_BIN_COUNT + first_bin] = 5.0f;
        logits[cell * DEPTH_BIN_COUNT + second_bin] = 5.0f;
    }
    return logits;
}

std::vector<float> uniform_segmentation_logits(int active_class)
{
    std::vector<float> logits(SURFACE_GRID_CELLS * SURFACE_CLASS_COUNT, -4.0f);
    for (int cell = 0; cell < SURFACE_GRID_CELLS; ++cell) {
        logits[cell * SURFACE_CLASS_COUNT + active_class] = 4.0f;
    }
    return logits;
}

std::vector<float> hwc_to_chw(const std::vector<float>& hwc, int channels)
{
    std::vector<float> chw(hwc.size(), 0.0f);
    for (int cell = 0; cell < SURFACE_GRID_CELLS; ++cell) {
        for (int channel = 0; channel < channels; ++channel) {
            chw[channel * SURFACE_GRID_CELLS + cell] = hwc[cell * channels + channel];
        }
    }
    return chw;
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
    assert(fused.depth_level == "unknown");
    assert(fused.depth_source == "step_override_far");

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

    // The final E3 segmentation contract has four classes.  UNKNOWN_OTHER is
    // never a traversable path and must conservatively slow a clear detector.
    obstacle::SurfaceSegmenter unknown_segmenter;
    SurfaceResult unknown_surface;
    const std::vector<float> unknown_seg = uniform_segmentation_logits(UNKNOWN_OTHER);
    assert(unknown_segmenter.PostprocessLogits(
        unknown_seg.data(), unknown_seg.size(), depth.data(), depth.size(), true, 700,
        &unknown_surface));
    assert(unknown_surface.center.unknown_ratio > 0.99f);
    assert(!unknown_surface.center.safe_candidate);
    assert(unknown_surface.primary_hazard == "unknown_other");
    assert(unknown_surface.depth_level == "unknown");
    assert(unknown_surface.depth_ambiguous);
    const AvoidanceDecision unknown_fused = fusion.Fuse(detection, unknown_surface, 700);
    assert(unknown_fused.action == "slow");
    assert(unknown_fused.hazard_type == "unknown_other");

    // Ambiguous road depth cannot erase a reliable near object warning.
    AvoidanceDecision near_detection = detection;
    near_detection.depth_level = "near";
    near_detection.depth_confidence = 0.8f;
    near_detection.depth_ambiguous = false;
    near_detection.depth_source = "geometry";
    const AvoidanceDecision near_unknown_fused =
        fusion.Fuse(near_detection, unknown_surface, 701);
    assert(near_unknown_fused.depth_level == "near");
    assert(near_unknown_fused.depth_source == "geometry");

    // Equal evidence on opposite sides of a NEAR/MID boundary is explicitly
    // unknown.  A single strongly separated group is accepted immediately.
    obstacle::SurfaceSegmenter ambiguous_segmenter;
    SurfaceResult ambiguous_surface;
    const std::vector<float> ambiguous = ambiguous_depth_logits(6, 8);
    assert(ambiguous_segmenter.PostprocessLogits(
        clear.data(), clear.size(), ambiguous.data(), ambiguous.size(), true, 800,
        &ambiguous_surface));
    assert(ambiguous_surface.depth_level == "unknown");
    assert(ambiguous_surface.depth_ambiguous);
    assert(ambiguous_surface.depth_margin < 0.20f);

    obstacle::SurfaceSegmenter confident_segmenter;
    SurfaceResult confident_surface;
    assert(confident_segmenter.PostprocessLogits(
        clear.data(), clear.size(), depth.data(), depth.size(), true, 900,
        &confident_surface));
    assert(confident_surface.depth_level == "far");
    assert(!confident_surface.depth_ambiguous);
    assert(confident_surface.depth_margin >= 0.20f);

    // HWC and CHW output layouts must decode to the same semantic result.
    obstacle::SurfaceSegmenter chw_segmenter;
    SurfaceResult chw_surface;
    const std::vector<float> clear_chw = hwc_to_chw(clear, SURFACE_CLASS_COUNT);
    const std::vector<float> depth_chw = hwc_to_chw(depth, DEPTH_BIN_COUNT);
    assert(chw_segmenter.PostprocessLogits(
        clear_chw.data(), clear_chw.size(), depth_chw.data(), depth_chw.size(), false, 1000,
        &chw_surface));
    assert(chw_surface.center.safe_candidate == confident_surface.center.safe_candidate);
    assert(chw_surface.depth_level == confident_surface.depth_level);
    assert(!chw_segmenter.PostprocessLogits(
        clear_chw.data(), clear_chw.size() - 1U,
        depth_chw.data(), depth_chw.size(), false, 1100, &chw_surface));

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
