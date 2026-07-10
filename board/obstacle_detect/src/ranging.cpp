#include "../include/ranging.hpp"

#include "../include/semantic_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace obstacle {
namespace {

const float kPi = 3.14159265358979323846f;

float env_float(const char* name, float fallback)
{
    const char* value = std::getenv(name);
    return (value == NULL || value[0] == '\0') ? fallback : static_cast<float>(std::atof(value));
}

float clampf(float value, float lo, float hi)
{
    return std::max(lo, std::min(hi, value));
}

float box_width(const DetectionItem& item)
{
    return std::max(1.0f, item.box[2] - item.box[0]);
}

float box_height(const DetectionItem& item)
{
    return std::max(1.0f, item.box[3] - item.box[1]);
}

int border_touches(const DetectionItem& item, int width, int height)
{
    int count = 0;
    if (item.box[0] <= 2.0f) ++count;
    if (item.box[1] <= 2.0f) ++count;
    if (item.box[2] >= width - 3.0f) ++count;
    if (item.box[3] >= height - 3.0f) ++count;
    return count;
}

std::string risk_from_safe_distance(float safe_distance)
{
    if (safe_distance < 0.0f) return "unknown";
    if (safe_distance < 0.80f) return "urgent";
    if (safe_distance < 1.05f) return "near";
    if (safe_distance < 2.00f) return "warning";
    return "far";
}

}  // namespace

RangingEstimator::RangingEstimator()
    : image_shape_{720, 1280},
      fov_h_deg_(49.7f),
      fov_v_deg_(78.9f),
      camera_height_m_(0.85f),
      camera_pitch_deg_(15.0f),
      min_distance_m_(0.20f),
      max_distance_m_(8.0f),
      fx_(1.0f),
      fy_(1.0f)
{
}

void RangingEstimator::Initialize(const std::array<int, 2>& image_shape)
{
    image_shape_ = image_shape;
    fov_h_deg_ = env_float("A1_CAM_FOV_H_DEG", 49.7f);
    fov_v_deg_ = env_float("A1_CAM_FOV_V_DEG", 78.9f);
    camera_height_m_ = env_float("A1_CAM_HEIGHT_M", 0.85f);
    camera_pitch_deg_ = env_float("A1_CAM_PITCH_DOWN_DEG", 15.0f);
    min_distance_m_ = env_float("A1_DIST_MIN_M", 0.20f);
    max_distance_m_ = env_float("A1_DIST_MAX_M", 8.0f);

    fx_ = 0.5f * image_shape_[0] /
          std::tan(0.5f * fov_h_deg_ * kPi / 180.0f);
    fy_ = 0.5f * image_shape_[1] /
          std::tan(0.5f * fov_v_deg_ * kPi / 180.0f);
}

RangingEstimator::EstimateValue RangingEstimator::GroundEstimate(
    const DetectionItem& item, float* lateral_m) const
{
    EstimateValue out;
    const float foot_x = 0.5f * (item.box[0] + item.box[2]);
    const float foot_y = clampf(item.box[3], 0.0f, static_cast<float>(image_shape_[1] - 1));
    const float cx = 0.5f * image_shape_[0];
    const float cy = 0.5f * image_shape_[1];
    const float ray_down = std::atan((foot_y - cy) / std::max(1.0f, fy_)) +
                           camera_pitch_deg_ * kPi / 180.0f;
    if (ray_down <= 0.75f * kPi / 180.0f) return out;

    const float z = camera_height_m_ / std::tan(ray_down);
    if (z < min_distance_m_ || z > max_distance_m_) return out;

    out.mean = z;
    const float bottom_ratio = foot_y / std::max(1.0f, static_cast<float>(image_shape_[1]));
    const float geometry_penalty = std::fabs(bottom_ratio - 0.72f);
    const float touch_penalty = 0.10f * border_touches(item, image_shape_[0], image_shape_[1]);
    out.sigma = clampf(0.10f + 0.16f * z + geometry_penalty + touch_penalty,
                       0.12f, 1.60f);
    out.valid = true;
    if (lateral_m != NULL) {
        *lateral_m = (foot_x - cx) * z / std::max(1.0f, fx_);
    }
    return out;
}

bool RangingEstimator::SizePrior(int raw_class_id, float* size_m, float* relative_sigma) const
{
    if (semantic::ModelClassCount() != 25 || size_m == NULL || relative_sigma == NULL) {
        return false;
    }
    switch (raw_class_id) {
        case 3:  *size_m = 1.70f; *relative_sigma = 0.18f; return true;  // person
        case 17: *size_m = 0.80f; *relative_sigma = 0.28f; return true;  // bench
        case 21: *size_m = 0.35f; *relative_sigma = 0.35f; return true;  // plant pot
        case 23: *size_m = 0.85f; *relative_sigma = 0.25f; return true;  // chair
        default: return false;
    }
}

RangingEstimator::EstimateValue RangingEstimator::SizeEstimate(const DetectionItem& item) const
{
    EstimateValue out;
    float physical_size = 0.0f;
    float relative_sigma = 0.0f;
    if (!SizePrior(item.raw_class_id, &physical_size, &relative_sigma)) return out;
    if (border_touches(item, image_shape_[0], image_shape_[1]) >= 2) return out;

    const float pixel_height = box_height(item);
    const float z = fy_ * physical_size / pixel_height;
    if (z < min_distance_m_ || z > max_distance_m_) return out;
    out.mean = z;
    out.sigma = clampf(z * relative_sigma + 0.08f, 0.12f, 2.0f);
    out.valid = true;
    return out;
}

float RangingEstimator::NearFieldUpperBound(const DetectionItem& item) const
{
    const float bottom = item.box[3] / std::max(1.0f, static_cast<float>(image_shape_[1]));
    const float wr = box_width(item) / std::max(1.0f, static_cast<float>(image_shape_[0]));
    const float hr = box_height(item) / std::max(1.0f, static_cast<float>(image_shape_[1]));
    if (bottom > 0.975f && (wr > 0.30f || hr > 0.32f)) return 0.45f;
    if (bottom > 0.945f && (wr > 0.18f || hr > 0.24f)) return 0.70f;
    if (bottom > 0.915f && wr > 0.12f) return 1.00f;
    return -1.0f;
}

void RangingEstimator::Estimate(DetectionItem* item) const
{
    if (item == NULL) return;
    item->distance_m = -1.0f;
    item->safe_distance_m = -1.0f;
    item->distance_sigma_m = -1.0f;
    item->distance_confidence = 0.0f;
    item->distance_source = "unknown";
    item->lateral_m = 0.0f;

    float lateral = 0.0f;
    EstimateValue ground = GroundEstimate(*item, &lateral);
    EstimateValue size = SizeEstimate(*item);
    const float near_upper = NearFieldUpperBound(*item);
    item->lateral_m = lateral;

    EstimateValue fused;
    if (ground.valid && size.valid) {
        const float residual = std::fabs(ground.mean - size.mean) /
            std::sqrt(ground.sigma * ground.sigma + size.sigma * size.sigma);
        if (residual <= 2.5f) {
            const float wg = 1.0f / (ground.sigma * ground.sigma);
            const float ws = 1.0f / (size.sigma * size.sigma);
            fused.mean = (wg * ground.mean + ws * size.mean) / (wg + ws);
            fused.sigma = std::sqrt(1.0f / (wg + ws));
            fused.valid = true;
            item->distance_source = "fused";
        } else {
            fused = ground;
            fused.sigma *= 1.35f;
            item->distance_source = "ground_reject_size";
        }
    } else if (ground.valid) {
        fused = ground;
        item->distance_source = "ground";
    } else if (size.valid) {
        fused = size;
        item->distance_source = "size";
    }

    if (fused.valid && near_upper > 0.0f && fused.mean > near_upper) {
        fused.mean = near_upper;
        fused.sigma = std::max(fused.sigma, 0.22f);
        item->distance_source = "nearfield_cap";
    }

    if (item->quality == "coarse" || item->score < 0.16f) {
        if (near_upper > 0.0f) {
            item->safe_distance_m = near_upper;
            item->distance_sigma_m = 0.35f;
            item->distance_confidence = 0.30f;
            item->distance_source = "nearfield_bound";
            item->risk_level = risk_from_safe_distance(near_upper);
        } else {
            item->risk_level = "unknown";
        }
        return;
    }

    if (!fused.valid) {
        if (near_upper > 0.0f) {
            item->safe_distance_m = near_upper;
            item->distance_sigma_m = 0.35f;
            item->distance_confidence = 0.35f;
            item->distance_source = "nearfield_bound";
            item->risk_level = risk_from_safe_distance(near_upper);
        } else {
            item->risk_level = "unknown";
        }
        return;
    }

    item->distance_m = clampf(fused.mean, min_distance_m_, max_distance_m_);
    item->distance_sigma_m = fused.sigma;
    item->safe_distance_m = clampf(item->distance_m - fused.sigma,
                                   min_distance_m_, max_distance_m_);
    item->distance_confidence = clampf(1.0f - fused.sigma / std::max(0.5f, item->distance_m),
                                       0.15f, 0.95f);
    item->risk_level = risk_from_safe_distance(item->safe_distance_m);
}

}  // namespace obstacle
