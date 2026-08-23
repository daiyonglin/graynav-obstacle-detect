#pragma once

#include "common.hpp"

#include <array>

namespace obstacle {

/**
 * @brief 将全图检测框转换为带不确定度的保守单目距离证据。
 *
 * 模块并行构造三类互补证据：框底部射线与地面的交点距离、类别真实尺寸先验、
 * 以及贴近画面底部的大目标所给出的近场距离上界。地面法与尺寸法一致时按
 * 逆方差融合；冲突时保留更可信来源并增大方差。最终向规划器提供
 * safe_distance = mean - sigma，而不是过于乐观的均值。
 */
class RangingEstimator {
public:
    RangingEstimator();

    void Initialize(const std::array<int, 2>& image_shape);
    void Estimate(DetectionItem* item) const;

private:
    /** 一个高斯近似测量：mean 为距离均值，sigma 为标准差，valid 表示证据可用。 */
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
    float ground_contact_offset_ratio_;
    float geometry_scale_;
    float safety_scale_;
    float min_distance_m_;
    float max_distance_m_;
    float fx_;
    float fy_;
};

}  // namespace obstacle
