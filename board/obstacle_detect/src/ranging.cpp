#include "../include/ranging.hpp"

#include "../include/semantic_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace obstacle {
namespace {

/*
 * 单目测距采用“多证据 + 不确定度”设计：
 * 1. 框底部射线与地平面求交；2. 类别物理尺寸先验；3. 近场上界；
 * 4. 一致时逆方差融合。规划器使用 mean-sigma，而不是乐观均值。
 */

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
      camera_height_m_(0.71f),
      camera_pitch_deg_(15.0f),
      min_distance_m_(0.20f),
      max_distance_m_(8.0f),
      fx_(1.0f),
      fy_(1.0f)
{
}

void RangingEstimator::Initialize(const std::array<int, 2>& image_shape)
{
    // 相机安装参数均可通过环境变量标定；fx/fy 由 FOV 与完整画面尺寸换算。
    image_shape_ = image_shape;
    fov_h_deg_ = env_float("A1_CAM_FOV_H_DEG", 49.7f);
    fov_v_deg_ = env_float("A1_CAM_FOV_V_DEG", 78.9f);
    camera_height_m_ = env_float("A1_CAM_HEIGHT_M", 0.71f);
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
    /*
     * 取框底边中心近似目标落地点，将像素射线叠加相机向下俯仰角后与
     * y=地面求交。上半身/人脸框没有真实脚点，会被主动拒绝并交给尺寸
     * 先验，避免把人脸框底边误当成地面接触点而得到虚假的远距离。
     */
    EstimateValue out;
    const float foot_x = 0.5f * (item.box[0] + item.box[2]);
    const float foot_y = clampf(item.box[3], 0.0f, static_cast<float>(image_shape_[1] - 1));
    const float bottom_ratio = foot_y /
        std::max(1.0f, static_cast<float>(image_shape_[1]));
    const float cx = 0.5f * image_shape_[0];
    const float cy = 0.5f * image_shape_[1];
    const float ray_down = std::atan((foot_y - cy) / std::max(1.0f, fy_)) +
                           camera_pitch_deg_ * kPi / 180.0f;
    if (ray_down <= 0.75f * kPi / 180.0f) return out;

    const float z = camera_height_m_ / std::tan(ray_down);
    if (z < min_distance_m_ || z > max_distance_m_) return out;

    /*
     * person 框底部只有在“由该落地点距离反推的全身像素高度”与实际框高大致
     * 一致时才可视为脚点。上半身、腿部或人脸框的 visible_fraction 明显偏小，
     * 此时跳过地面测距，转交部分人体尺寸先验，避免得到虚假的 FAR。
     */
    if (item.raw_class_id == 3) {
        const float expected_full_height = fy_ * 1.70f / std::max(0.20f, z);
        const float visible_fraction = box_height(item) /
            std::max(1.0f, expected_full_height);
        if (visible_fraction < 0.52f) return out;
    }

    out.mean = z;
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
    // 先验同时给出均值和相对标准差，个体尺寸差异不会被伪装成精确测量。
    if (semantic::ModelClassCount() != 25 || size_m == NULL || relative_sigma == NULL) {
        return false;
    }
    switch (raw_class_id) {
        case 3:  *size_m = 1.70f; *relative_sigma = 0.18f; return true;  // person
        case 9:  *size_m = 0.55f; *relative_sigma = 0.45f; return true;  // dustbin/container
        case 17: *size_m = 0.80f; *relative_sigma = 0.28f; return true;  // bench
        case 20: *size_m = 0.70f; *relative_sigma = 0.40f; return true;  // traffic barrel
        case 21: *size_m = 0.35f; *relative_sigma = 0.35f; return true;  // plant pot
        case 23: *size_m = 0.85f; *relative_sigma = 0.25f; return true;  // chair
        default: return false;
    }
}

RangingEstimator::EstimateValue RangingEstimator::SizeEstimate(const DetectionItem& item) const
{
    // 完整目标使用物理高度；部分人体按外观比例选择头宽/肩宽/腿宽先验。
    EstimateValue out;
    const float pixel_width = box_width(item);
    const float pixel_height = box_height(item);
    const float aspect = pixel_width / pixel_height;
    if (item.raw_class_id == 3 && pixel_width >= 18.0f) {
        const float foot_y = clampf(item.box[3], 0.0f,
                                    static_cast<float>(image_shape_[1] - 1));
        const float cy = 0.5f * image_shape_[1];
        const float ray_down = std::atan((foot_y - cy) / std::max(1.0f, fy_)) +
            camera_pitch_deg_ * kPi / 180.0f;
        float visible_fraction = 0.0f;
        if (ray_down > 0.75f * kPi / 180.0f) {
            const float z_ground = camera_height_m_ / std::tan(ray_down);
            if (z_ground >= min_distance_m_ && z_ground <= max_distance_m_) {
                const float expected_full_height = fy_ * 1.70f / z_ground;
                visible_fraction = pixel_height / std::max(1.0f, expected_full_height);
            }
        }

        if (visible_fraction < 0.52f) {
            /*
             * 方形区域更接近头部，较窄高框更接近躯干/腿部。宽度先验仅用于
             * 部分人体，并赋予 35%~45% 的较大不确定度；规划最终使用 mean-sigma。
             */
            float physical_width = 0.42f;  // 成人肩宽/上半身可见宽度。
            float relative_sigma = 0.38f;
            if (aspect >= 0.72f) {
                physical_width = 0.18f;     // 成人头宽。
                relative_sigma = 0.32f;
            } else if (aspect < 0.32f) {
                physical_width = 0.28f;     // 双腿或窄身体区域宽度。
                relative_sigma = 0.45f;
            }
            const float z_partial = fx_ * physical_width / pixel_width;
            if (z_partial >= min_distance_m_ && z_partial <= max_distance_m_) {
                out.mean = z_partial;
                out.sigma = clampf(relative_sigma * z_partial + 0.10f,
                                   0.16f, 1.60f);
                out.valid = true;
                return out;
            }
        }
    }
    float physical_size = 0.0f;
    float relative_sigma = 0.0f;
    if (!SizePrior(item.raw_class_id, &physical_size, &relative_sigma)) return out;
    if (border_touches(item, image_shape_[0], image_shape_[1]) >= 2) return out;

    const float z = fy_ * physical_size / pixel_height;
    if (z < min_distance_m_ || z > max_distance_m_) return out;
    out.mean = z;
    out.sigma = clampf(z * relative_sigma + 0.08f, 0.12f, 2.0f);
    out.valid = true;
    return out;
}

float RangingEstimator::NearFieldUpperBound(const DetectionItem& item) const
{
    /*
     * 靠近画面底部且占比较大的框只提供“距离不超过某值”的上界。
     * 横跨两侧边界的饱和框没有可信物理宽度，禁止转换成 0.45m 紧急距离。
     */
    const float bottom = item.box[3] / std::max(1.0f, static_cast<float>(image_shape_[1]));
    const float wr = box_width(item) / std::max(1.0f, static_cast<float>(image_shape_[0]));
    const float hr = box_height(item) / std::max(1.0f, static_cast<float>(image_shape_[1]));
    const bool clips_both_horizontal_borders =
        item.box[0] <= 3.0f &&
        item.box[2] >= static_cast<float>(image_shape_[0] - 4);
    // A broad box clipped by both image borders has no trustworthy physical
    // width. Do not convert this regression artifact into a precise 0.45 m
    // emergency measurement.
    if (clips_both_horizontal_borders || wr > 0.90f || item.quality == "coarse") {
        return -1.0f;
    }
    if (item.raw_class_id == 3) {
        const float aspect = box_width(item) / box_height(item);
        const bool head_like = bottom < 0.78f && aspect > 0.55f && aspect < 1.65f;
        if (head_like && wr > 0.30f) return 0.70f;
        if (head_like && wr > 0.18f) return 1.10f;
    }
    if (bottom > 0.975f && (wr > 0.30f || hr > 0.32f)) return 0.45f;
    if (bottom > 0.945f && (wr > 0.18f || hr > 0.24f)) return 0.70f;
    if (bottom > 0.915f && wr > 0.12f) return 1.00f;
    return -1.0f;
}

void RangingEstimator::Estimate(DetectionItem* item) const
{
    /*
     * 融合步骤：先计算 ground/size，两者归一化残差 <=2.5 时按 1/sigma^2
     * 加权；冲突时保留地面结果并增大不确定度；最后应用近场上界。
     * coarse 或低分框不报告 distance_m，只保留风险上界或 unknown。
     */
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

    // 地面证据被判为部分人体而拒绝时，仍用尺寸距离恢复横向地面位置近似。
    if (!ground.valid && fused.valid) {
        const float center_x = 0.5f * (item->box[0] + item->box[2]);
        item->lateral_m = (center_x - 0.5f * image_shape_[0]) *
                          fused.mean / std::max(1.0f, fx_);
    }

    if (item->quality == "coarse" || item->score < 0.16f) {
        if (near_upper > 0.0f && item->quality != "coarse") {
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
