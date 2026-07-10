#pragma once

#include "common.hpp"

#include <array>

namespace obstacle {

/**
 * Converts a full-frame detection box into conservative monocular range
 * evidence.  It fuses ground-plane geometry, class size priors and a
 * near-field upper bound while retaining uncertainty for navigation.
 */
class RangingEstimator {
public:
    RangingEstimator();

    void Initialize(const std::array<int, 2>& image_shape);
    void Estimate(DetectionItem* item) const;

private:
    struct EstimateValue {
        float mean;
        float sigma;
        bool valid;
        EstimateValue() : mean(-1.0f), sigma(1.0f), valid(false) {}
    };

    EstimateValue GroundEstimate(const DetectionItem& item, float* lateral_m) const;
    EstimateValue SizeEstimate(const DetectionItem& item) const;
    float NearFieldUpperBound(const DetectionItem& item) const;
    bool SizePrior(int raw_class_id, float* size_m, float* relative_sigma) const;

    std::array<int, 2> image_shape_;
    float fov_h_deg_;
    float fov_v_deg_;
    float camera_height_m_;
    float camera_pitch_deg_;
    float min_distance_m_;
    float max_distance_m_;
    float fx_;
    float fy_;
};

}  // namespace obstacle
