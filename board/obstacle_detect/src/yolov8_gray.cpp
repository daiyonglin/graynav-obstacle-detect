#include "../include/utils.hpp"
#include "../include/semantic_config.hpp"

#include <algorithm>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifndef A1_YOLO_INPUT_CHANNELS
#define A1_YOLO_INPUT_CHANNELS 3
#endif

/*
 * B3 单通道 YOLOv8n-DCE 的板端封装。该文件负责双 ROI 预处理、A1 NPU 调用、
 * 六个 raw head 的结构校验、DFL 解码、全图坐标反映射和多目标 NMS；它只产生
 * 单帧检测，不在这里保存轨迹或直接决定语音动作。
 */
namespace {

// YOLOv8 DFL 每条边由 16 个离散 bin 表示，因此回归头固定为 4*16=64 通道。
constexpr int REG_MAX = 16;
constexpr int REG_CHANNELS = 64;

constexpr int DISPLAY_PERSON = DISPLAY_CLASS_PERSON;
const int NUM_CLASSES = obstacle::semantic::ModelClassCount();

bool getenv_flag(const char* name, bool fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T';
}

int getenv_int(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

float getenv_float(const char* name, float fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return static_cast<float>(std::atof(value));
}

/*
 * 仅在平均亮度明显偏低且仍保留纹理时启用局部直方图增强；正常帧保持逐字节
 * 不变，极低方差的遮挡帧也不增强，以免把传感器噪声伪造成目标纹理。
 */
bool apply_adaptive_gray_lut(ssne_tensor_t* tensor, int frame_id)
{
    if (!getenv_flag("A1_ADAPTIVE_GRAY", true) || tensor == nullptr) return false;
    uint8_t* data = reinterpret_cast<uint8_t*>(get_data(*tensor));
    const uint32_t size = get_total_size(*tensor);
    const int width = static_cast<int>(get_width(*tensor));
    const int height = static_cast<int>(get_height(*tensor));
    if (data == nullptr || size < static_cast<uint32_t>(width * height) ||
        width <= 0 || height <= 0) return false;

    uint64_t sum = 0;
    uint64_t sum_sq = 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < size; i += 16) {
        const uint32_t v = data[i];
        sum += v;
        sum_sq += v * v;
        ++count;
    }
    if (count == 0) return false;
    const float mean = static_cast<float>(sum) / count;
    const float variance = std::max(0.0f,
        static_cast<float>(sum_sq) / count - mean * mean);
    const float stddev = std::sqrt(variance);

    const float dark_mean = getenv_float("A1_ADAPTIVE_GRAY_DARK_MEAN", 75.0f);
    if (mean >= dark_mean || stddev < 5.0f) return false;

    // 4x4 分块的裁剪局部直方图均衡用于恢复暗背景中的人体和障碍轮廓；
    // 再与原灰度图混合，避免分块接缝并限制分布偏移，保持接近训练灰度域。
    const int tiles_x = 4;
    const int tiles_y = 4;
    const int blend = std::max(20, std::min(80,
        getenv_int("A1_ADAPTIVE_GRAY_BLEND", 60)));
    for (int ty = 0; ty < tiles_y; ++ty) {
        const int y0 = ty * height / tiles_y;
        const int y1 = (ty + 1) * height / tiles_y;
        for (int tx = 0; tx < tiles_x; ++tx) {
            const int x0 = tx * width / tiles_x;
            const int x1 = (tx + 1) * width / tiles_x;
            int hist[256] = {0};
            const int pixels = std::max(1, (x1 - x0) * (y1 - y0));
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) ++hist[data[y * width + x]];
            }
            const int clip_limit = std::max(8, pixels / 64);
            int excess = 0;
            for (int i = 0; i < 256; ++i) {
                if (hist[i] > clip_limit) {
                    excess += hist[i] - clip_limit;
                    hist[i] = clip_limit;
                }
            }
            const int uniform = excess / 256;
            const int remainder = excess % 256;
            for (int i = 0; i < 256; ++i) hist[i] += uniform + (i < remainder ? 1 : 0);
            int cdf[256];
            int running = 0;
            for (int i = 0; i < 256; ++i) {
                running += hist[i];
                cdf[i] = running;
            }
            int cdf_min = 0;
            while (cdf_min < 255 && cdf[cdf_min] == 0) ++cdf_min;
            const int base = cdf[cdf_min];
            const int denom = std::max(1, pixels - base);
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    const int index = y * width + x;
                    const int original = data[index];
                    const int equalized = std::max(0, std::min(255,
                        (cdf[original] - base) * 255 / denom));
                    data[index] = static_cast<uint8_t>(
                        (original * (100 - blend) + equalized * blend + 50) / 100);
                }
            }
        }
    }

    if (getenv_flag("A1_ADAPTIVE_GRAY_DIAG", false) && frame_id % 300 == 0) {
        std::cout << "[YOLOV8GRAY][LIGHT] frame=" << frame_id
                  << " mean=" << mean << " std=" << stddev
                  << " mode=local_hist blend=" << blend << std::endl;
    }
    return true;
}

std::string getenv_string(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::string(value);
}

/*
 * ONNX 导出形状通常以 NCHW 描述，而 A1 运行时把转换后的 head 暴露为图像式
 * HWC 缓冲。默认按实测 HWC 读取，并保留环境变量覆盖用于转换一致性排查。
 */
bool read_model_output_as_hwc()
{
    static int mode = -1;
    if (mode < 0) {
        const std::string layout = getenv_string("A1_MODEL_OUTPUT_LAYOUT", "HWC");
        mode = (layout == "CHW" || layout == "chw") ? 0 : 1;
    }
    return mode != 0;
}

struct DistanceConfig {
    float fov_h_deg;
    float fov_v_deg;
    float camera_height_m;
    float camera_pitch_down_deg;
    float min_distance_m;
    float max_distance_m;

    DistanceConfig()
        : fov_h_deg(getenv_float("A1_CAM_FOV_H_DEG", 49.7f)),
          fov_v_deg(getenv_float("A1_CAM_FOV_V_DEG", 78.9f)),
          camera_height_m(getenv_float("A1_CAM_HEIGHT_M", 0.71f)),
          camera_pitch_down_deg(getenv_float("A1_CAM_PITCH_DOWN_DEG", 15.0f)),
          min_distance_m(getenv_float("A1_DIST_MIN_M", 0.2f)),
          max_distance_m(getenv_float("A1_DIST_MAX_M", 8.0f)) {}
};

struct BranchView {
    // 对 ssne_tensor_t 的无拷贝视图；统一保存通道、网格、stride、layout 和数据指针。
    float* data = nullptr;
    int out_idx = -1;
    int w = 0;
    int h = 0;
    int c = 0;
    int stride = 0;
    uint32_t raw_total = 0;
    uint32_t element_count = 0;
    uint8_t dtype = 0;
    uint8_t fmt = 0;
    bool is_cls = false;
};

struct ClassCandidate {
    int raw_cls = -1;
    float score = 0.0f;
};

inline float fast_sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float clampf(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

float box_area_ratio(const std::array<float, 4>& box, int img_w, int img_h)
{
    float w = std::max(0.0f, box[2] - box[0]);
    float h = std::max(0.0f, box[3] - box[1]);
    return (w * h) / std::max(1.0f, float(img_w * img_h));
}

float box_width_ratio(const std::array<float, 4>& box, int img_w)
{
    float w = std::max(0.0f, box[2] - box[0]);
    return w / std::max(1.0f, float(img_w));
}

float box_height_ratio(const std::array<float, 4>& box, int img_h)
{
    float h = std::max(0.0f, box[3] - box[1]);
    return h / std::max(1.0f, float(img_h));
}

float box_center_y_ratio(const std::array<float, 4>& box, int img_h)
{
    float cy = 0.5f * (box[1] + box[3]);
    return cy / std::max(1.0f, float(img_h));
}

int count_touch_borders(const std::array<float, 4>& box, int img_w, int img_h)
{
    int touch = 0;
    if (box[0] <= 2.f) touch++;
    if (box[1] <= 2.f) touch++;
    if (box[2] >= img_w - 3.f) touch++;
    if (box[3] >= img_h - 3.f) touch++;
    return touch;
}

bool is_small_far_candidate_box(const std::array<float, 4>& box,
                                int img_w,
                                int img_h)
{
    const float area_ratio = box_area_ratio(box, img_w, img_h);
    const float width_ratio = box_width_ratio(box, img_w);
    const float height_ratio = box_height_ratio(box, img_h);
    const float center_y = box_center_y_ratio(box, img_h);
    const float bottom_ratio = box[3] / std::max(1.0f, static_cast<float>(img_h));
    const int touch = count_touch_borders(box, img_w, img_h);

    return area_ratio >= 0.0005f && area_ratio <= 0.12f &&
           width_ratio >= 0.025f && width_ratio <= 0.55f &&
           height_ratio >= 0.020f && height_ratio <= 0.55f &&
           center_y >= 0.12f && bottom_ratio >= 0.18f &&
           touch <= 1;
}

bool is_supported_raw_class(int cls_id)
{
    return obstacle::semantic::IsSupportedRawClass(cls_id);
}

bool is_person_raw_class(int cls_id)
{
    return obstacle::semantic::SemanticClassFromRaw(cls_id) == DISPLAY_PERSON;
}

bool is_furniture_raw_class(int cls_id)
{
    return obstacle::semantic::IsFurnitureLikeRawClass(cls_id);
}

float candidate_threshold_for_class(int raw_cls)
{
    return obstacle::semantic::CandidateThreshold(raw_cls);
}

std::string top_class_counts_text(const std::vector<int>& counts, int limit)
{
    std::vector<int> ids;
    ids.reserve(counts.size());
    for (int i = 0; i < static_cast<int>(counts.size()); ++i) {
        if (counts[i] > 0) {
            ids.push_back(i);
        }
    }
    std::sort(ids.begin(), ids.end(), [&counts](int a, int b) {
        return counts[a] > counts[b];
    });
    std::string out;
    const int n = std::min(limit, static_cast<int>(ids.size()));
    for (int i = 0; i < n; ++i) {
        if (!out.empty()) out += ",";
        out += obstacle::semantic::RawLabel(ids[i]) + ":" + std::to_string(counts[ids[i]]);
    }
    return out.empty() ? "none" : out;
}

std::string sector_from_box(const std::array<float, 4>& box, int img_w)
{
    const float frame_width = std::max(1.0f, static_cast<float>(img_w));
    const float left_bound = obstacle::semantic::SectorLeftBoundaryRatio() * frame_width;
    const float right_bound = obstacle::semantic::SectorRightBoundaryRatio() * frame_width;
    const float width = std::max(1.0f, box[2] - box[0]);
    const float cx = 0.5f * (box[0] + box[2]);

    // 近乎占满全幅的框才作为 wide。窄视场中的普通大目标按框中心划入侧区，
    // 避免侧方人体或椅子被错误升级为覆盖整条通路的中心障碍。
    if (width / frame_width > obstacle::semantic::WideBoxRatio()) {
        return "wide";
    }
    if (cx < left_bound) return "left";
    if (cx > right_bound) return "right";
    return "center";
}

float estimate_ground_distance_m(const std::array<float, 4>& box,
                                 int img_h,
                                 const DistanceConfig& cfg)
{
    const float foot_y = clampf(box[3], 0.0f, static_cast<float>(img_h - 1));
    const float pi = 3.14159265358979323846f;
    const float cy = 0.5f * static_cast<float>(img_h);
    const float fy = cy / std::tan(0.5f * cfg.fov_v_deg * pi / 180.0f);
    const float ray_down_rad = std::atan((foot_y - cy) / std::max(1.0f, fy)) +
                               cfg.camera_pitch_down_deg * pi / 180.0f;

    if (ray_down_rad <= 0.5f * pi / 180.0f) {
        return -1.0f;
    }

    const float distance = cfg.camera_height_m / std::tan(ray_down_rad);

    if (distance < cfg.min_distance_m || distance > cfg.max_distance_m) {
        return -1.0f;
    }
    return distance;
}

float estimate_nearfield_distance_m(const std::array<float, 4>& box,
                                    int img_w,
                                    int img_h)
{
    const float bottom_ratio = box[3] / std::max(1.0f, static_cast<float>(img_h));
    const float width_ratio = box_width_ratio(box, img_w);
    const float height_ratio = box_height_ratio(box, img_h);
    const float area_ratio = box_area_ratio(box, img_w, img_h);

    if (bottom_ratio < 0.90f) {
        return -1.0f;
    }
    if (bottom_ratio > 0.97f && (width_ratio > 0.30f || height_ratio > 0.32f || area_ratio > 0.10f)) {
        return 0.45f;
    }
    if (bottom_ratio > 0.94f && (width_ratio > 0.18f || height_ratio > 0.24f || area_ratio > 0.045f)) {
        return 0.70f;
    }
    if (bottom_ratio > 0.91f && (width_ratio > 0.12f || area_ratio > 0.025f)) {
        return 1.00f;
    }
    return -1.0f;
}

float distance_confidence_for_source(const std::array<float, 4>& box,
                                     int img_w,
                                     int img_h,
                                     const std::string& source,
                                     float score)
{
    float confidence = clampf(0.25f + 0.55f * score, 0.20f, 0.85f);
    const float bottom_ratio = box[3] / std::max(1.0f, static_cast<float>(img_h));
    const float area_ratio = box_area_ratio(box, img_w, img_h);
    const int touch = count_touch_borders(box, img_w, img_h);

    if (source == "ground") {
        confidence += (bottom_ratio > 0.45f && bottom_ratio < 0.96f) ? 0.12f : -0.18f;
    } else if (source == "fused" || source == "fused_ground" || source == "fused_size") {
        confidence += 0.10f;
    } else if (source == "size") {
        confidence -= (touch > 0 || area_ratio > 0.35f) ? 0.25f : 0.05f;
    } else if (source == "nearfield" || source == "nearfield_cap") {
        confidence = 0.58f;
    } else {
        confidence = 0.0f;
    }

    if (touch >= 2) {
        confidence -= 0.12f;
    }
    return clampf(confidence, 0.0f, 1.0f);
}

// ROD25 没有独立的桌子或纸箱类别。以下原始类与室内刚性障碍外观最接近，
// 只有通过时序稳定和几何质量检查后，才允许作为 generic_obstacle 证据。
bool is_indoor_rigid_raw_class(int raw_cls)
{
    if (obstacle::semantic::ModelClassCount() != 25) return false;
    return raw_cls == 9 ||   // dustbin / box-like container
           raw_cls == 17 ||  // bench / table-like furniture
           raw_cls == 20 ||  // traffic barrel / large container
           raw_cls == 22 ||  // electrical box / cabinet-like object
           raw_cls == 23 ||  // chair
           raw_cls == 24;    // bicycle rack / rigid frame
}

std::string quality_from_box(const std::array<float, 4>& box,
                             int img_w,
                             int img_h,
                             int display_cls,
                             int raw_cls,
                             float score)
{
    const float area_ratio = box_area_ratio(box, img_w, img_h);
    const float width_ratio = box_width_ratio(box, img_w);
    const float height_ratio = box_height_ratio(box, img_h);
    const int touch = count_touch_borders(box, img_w, img_h);

    // 同时贴住左右边界的框不能代表可靠目标范围。量化后的 DFL 异常值常以
    // 这种方式饱和，若不拦截会形成高置信度、全幅宽的伪障碍框。
    const bool clips_both_horizontal_borders =
        box[0] <= 3.0f && box[2] >= static_cast<float>(img_w - 4);
    if (clips_both_horizontal_borders ||
        (width_ratio > 0.90f && touch >= 2)) {
        return "coarse";
    }

    if (obstacle::semantic::IsObstacleClass(display_cls) &&
        score < 0.28f &&
        (width_ratio > 0.88f || area_ratio > 0.70f) &&
        touch >= 2) {
        return "coarse";
    }

    // 宽且细节不足的框常来自货架、屏幕、栏杆或合并背景结构。它们仍可作为
    // 导航风险线索，但必须标为 coarse，使 NMS 和跟踪优先选择细粒度目标框。
    const bool indoor_rigid = is_indoor_rigid_raw_class(raw_cls);
    if (obstacle::semantic::IsObstacleClass(display_cls) &&
        !obstacle::semantic::IsFurnitureLikeSemantic(display_cls) &&
        !indoor_rigid &&
        score < 0.55f &&
        width_ratio > 0.62f &&
        height_ratio < 0.48f &&
        area_ratio < 0.45f) {
        return "coarse";
    }

    if (obstacle::semantic::IsObstacleClass(display_cls) &&
        display_cls != obstacle::semantic::CHAIR_SEAT &&
        !indoor_rigid &&
        score < 0.48f &&
        width_ratio > 0.72f) {
        return "coarse";
    }

    if (obstacle::semantic::IsObstacleClass(display_cls) &&
        score >= 0.20f &&
        is_small_far_candidate_box(box, img_w, img_h)) {
        return "good";
    }

    if (display_cls == DISPLAY_PERSON &&
        score >= 0.20f &&
        is_small_far_candidate_box(box, img_w, img_h)) {
        return "good";
    }

    if (score < 0.22f || touch >= 4 ||
        (obstacle::semantic::IsObstacleClass(display_cls) &&
         score < 0.28f &&
         (area_ratio > 0.65f || width_ratio > 0.92f || height_ratio > 0.92f))) {
        return "low";
    }
    return "good";
}

bool class_size_prior_m(int raw_class_id, float* physical_size_m, bool* use_height)
{
    *physical_size_m = 0.0f;
    *use_height = true;

    if (obstacle::semantic::ModelClassCount() == 25) {
        switch (raw_class_id) {
            case 3:  *physical_size_m = 1.70f; *use_height = true;  return true;  // person
            case 17: *physical_size_m = 0.80f; *use_height = true;  return true;  // bench
            case 21: *physical_size_m = 0.35f; *use_height = true;  return true;  // plant_pot
            case 23: *physical_size_m = 0.85f; *use_height = true;  return true;  // chair
            default: return false;
        }
    }

    switch (raw_class_id) {
        case 0:  *physical_size_m = 1.70f; *use_height = true;  return true;  // person
        case 39: *physical_size_m = 0.25f; *use_height = true;  return true;  // bottle
        case 41: *physical_size_m = 0.12f; *use_height = true;  return true;  // cup
        case 56: *physical_size_m = 0.85f; *use_height = true;  return true;  // chair
        case 57: *physical_size_m = 0.80f; *use_height = true;  return true;  // couch
        case 60: *physical_size_m = 0.75f; *use_height = true;  return true;  // dining table
        case 62: *physical_size_m = 0.80f; *use_height = false; return true;  // tv
        case 63: *physical_size_m = 0.35f; *use_height = false; return true;  // laptop
        case 65: *physical_size_m = 0.18f; *use_height = false; return true;  // remote
        case 66: *physical_size_m = 0.45f; *use_height = false; return true;  // keyboard
        case 67: *physical_size_m = 0.15f; *use_height = false; return true;  // cell phone
        case 73: *physical_size_m = 0.24f; *use_height = false; return true;  // book
        default: return false;
    }
}

float estimate_size_distance_m(const std::array<float, 4>& box,
                               int img_w,
                               int img_h,
                               int raw_class_id,
                               const DistanceConfig& cfg)
{
    float physical_size_m = 0.0f;
    bool use_height = true;
    if (!class_size_prior_m(raw_class_id, &physical_size_m, &use_height)) {
        return -1.0f;
    }

    const float pi = 3.14159265358979323846f;
    const float fx = (0.5f * static_cast<float>(img_w)) /
                     std::tan(0.5f * cfg.fov_h_deg * pi / 180.0f);
    const float fy = (0.5f * static_cast<float>(img_h)) /
                     std::tan(0.5f * cfg.fov_v_deg * pi / 180.0f);

    const float pixel_size = use_height ? std::max(1.0f, box[3] - box[1])
                                        : std::max(1.0f, box[2] - box[0]);
    const float focal = use_height ? fy : fx;
    const float distance = focal * physical_size_m / pixel_size;

    if (distance < cfg.min_distance_m || distance > cfg.max_distance_m) {
        return -1.0f;
    }
    return distance;
}

bool prefer_ground_distance(int raw_class_id)
{
    const int sem = obstacle::semantic::SemanticClassFromRaw(raw_class_id);
    return sem == obstacle::semantic::PERSON ||
           obstacle::semantic::IsFurnitureLikeSemantic(sem) ||
           sem == obstacle::semantic::GENERIC_OBSTACLE;
}

bool reject_size_distance_for_box(const std::array<float, 4>& box,
                                  int img_w,
                                  int img_h,
                                  int raw_class_id)
{
    const float area_ratio = box_area_ratio(box, img_w, img_h);
    const float width_ratio = box_width_ratio(box, img_w);
    const float height_ratio = box_height_ratio(box, img_h);
    const float bottom_ratio = box[3] / std::max(1.0f, static_cast<float>(img_h));
    const int touch = count_touch_borders(box, img_w, img_h);

    (void)raw_class_id;

    // 贴边或覆盖大部分画面的局部目标不满足尺寸先验假设，只保留为近场/风险线索。
    if (area_ratio > 0.45f || width_ratio > 0.82f || height_ratio > 0.82f ||
        touch >= 2 || bottom_ratio > 0.92f) {
        return true;
    }

    return false;
}

float fuse_distance_m(float ground_m, float size_m, std::string* source)
{
    const bool has_ground = ground_m >= 0.0f;
    const bool has_size = size_m >= 0.0f;

    if (has_ground && has_size) {
        *source = "fused";
        return std::min(ground_m, size_m);
    }
    if (has_ground) {
        *source = "ground";
        return ground_m;
    }
    if (has_size) {
        *source = "size";
        return size_m;
    }

    *source = "unknown";
    return -1.0f;
}

float fuse_distance_m(const std::array<float, 4>& box,
                      int img_w,
                      int img_h,
                      int raw_class_id,
                      float ground_m,
                      float size_m,
                      std::string* source)
{
    const float nearfield_m = estimate_nearfield_distance_m(box, img_w, img_h);

    if (size_m >= 0.0f && reject_size_distance_for_box(box, img_w, img_h, raw_class_id)) {
        size_m = -1.0f;
    }

    if (ground_m >= 0.0f && size_m >= 0.0f) {
        const float ratio = size_m / std::max(0.01f, ground_m);
        if (ratio >= 0.60f && ratio <= 1.70f) {
            *source = "fused";
            const int sem = obstacle::semantic::SemanticClassFromRaw(raw_class_id);
            const bool ground_preferred =
                sem == obstacle::semantic::PERSON ||
                obstacle::semantic::IsFurnitureLikeSemantic(sem) ||
                sem == obstacle::semantic::GENERIC_OBSTACLE;
            const float ground_w = ground_preferred ? 0.70f : 0.45f;
            float fused = ground_w * ground_m + (1.0f - ground_w) * size_m;
            if (nearfield_m >= 0.0f) {
                *source = "nearfield_cap";
                fused = std::min(fused, nearfield_m);
            }
            return fused;
        }
        if (prefer_ground_distance(raw_class_id)) {
            *source = "fused_ground";
            return nearfield_m >= 0.0f ? std::min(ground_m, nearfield_m) : ground_m;
        }
        *source = "fused_size";
        return nearfield_m >= 0.0f ? std::min(size_m, nearfield_m) : size_m;
    }

    float fused = fuse_distance_m(ground_m, size_m, source);
    if (nearfield_m >= 0.0f && fused >= 0.0f && nearfield_m < fused) {
        *source = "nearfield_cap";
        return nearfield_m;
    }
    if (fused < 0.0f && nearfield_m >= 0.0f) {
        *source = "nearfield";
        return nearfield_m;
    }

    return fused;
}

std::string risk_from_distance(float distance_m)
{
    if (distance_m < 0.0f) return "unknown";
    if (distance_m < 1.0f) return "near";
    if (distance_m < 2.0f) return "warning";
    return "far";
}

inline float read_branch_value(const BranchView& b, int c, int y, int x)
{
    if (read_model_output_as_hwc()) {
        return b.data[(y * b.w + x) * b.c + c];
    }
    return b.data[c * b.h * b.w + y * b.w + x];
}

// 仅对已通过分类阈值的 anchor 解码一条 DFL 边，避免为全部网格执行 softmax。
float decode_dfl_side(const BranchView& branch, int side, int y, int x)
{
    float max_logit = -FLT_MAX;
    for (int bin = 0; bin < REG_MAX; ++bin) {
        const float value = read_branch_value(branch, side * REG_MAX + bin, y, x);
        if (!std::isfinite(value)) return -1.0f;
        max_logit = std::max(max_logit, value);
    }
    float sum = 0.0f;
    float expectation = 0.0f;
    for (int bin = 0; bin < REG_MAX; ++bin) {
        const float value = read_branch_value(branch, side * REG_MAX + bin, y, x);
        const float weight = std::exp(value - max_logit);
        sum += weight;
        expectation += weight * static_cast<float>(bin);
    }
    return sum > 1e-6f ? expectation / sum : 0.0f;
}

float probability_to_logit(float probability)
{
    const float p = clampf(probability, 1e-4f, 1.0f - 1e-4f);
    return std::log(p / (1.0f - p));
}

bool infer_float_branch_shape(uint32_t raw_total,
                              uint32_t w,
                              uint32_t h,
                              uint8_t dtype,
                              uint32_t* element_count,
                              int* channels)
{
    if (dtype != SSNE_FLOAT32 || w == 0 || h == 0) {
        return false;
    }

    const uint32_t hw = w * h;
    if (hw == 0) {
        return false;
    }

    if (raw_total % hw == 0) {
        const uint32_t c = raw_total / hw;
        if (c == NUM_CLASSES || c == REG_CHANNELS) {
            *element_count = raw_total;
            *channels = static_cast<int>(c);
            return true;
        }
    }

    const uint32_t bytes_per_float = 4;
    const uint32_t hw_bytes = hw * bytes_per_float;
    if (hw_bytes != 0 && raw_total % hw_bytes == 0) {
        const uint32_t c = raw_total / hw_bytes;
        if (c == NUM_CLASSES || c == REG_CHANNELS) {
            *element_count = raw_total / bytes_per_float;
            *channels = static_cast<int>(c);
            return true;
        }
    }

    return false;
}

bool make_branch_view(ssne_tensor_t tensor,
                      int out_idx,
                      int det_width,
                      BranchView* out)
{
    const uint32_t w = get_width(tensor);
    const uint32_t h = get_height(tensor);
    const uint32_t raw_total = get_total_size(tensor);
    const uint8_t dtype = get_data_type(tensor);
    const uint8_t fmt = get_data_format(tensor);

    uint32_t element_count = 0;
    int channels = 0;
    if (!infer_float_branch_shape(raw_total, w, h, dtype, &element_count, &channels)) {
        out->data = reinterpret_cast<float*>(get_data(tensor));
        out->out_idx = out_idx;
        out->w = static_cast<int>(w);
        out->h = static_cast<int>(h);
        out->c = 0;
        out->stride = 0;
        out->raw_total = raw_total;
        out->element_count = 0;
        out->dtype = dtype;
        out->fmt = fmt;
        out->is_cls = false;
        return false;
    }

    out->data = reinterpret_cast<float*>(get_data(tensor));
    out->out_idx = out_idx;
    out->w = static_cast<int>(w);
    out->h = static_cast<int>(h);
    out->c = channels;
    out->stride = (w > 0) ? (det_width / static_cast<int>(w)) : 0;
    out->raw_total = raw_total;
    out->element_count = element_count;
    out->dtype = dtype;
    out->fmt = fmt;
    out->is_cls = channels == NUM_CLASSES;
    return out->data != nullptr && out->stride > 0;
}

void build_anchor_points(const std::vector<BranchView>& cls_branches,
                         std::vector<float>* anchor_x,
                         std::vector<float>* anchor_y,
                         std::vector<float>* stride_vec)
{
    anchor_x->clear();
    anchor_y->clear();
    stride_vec->clear();

    for (size_t s = 0; s < cls_branches.size(); ++s) {
        const BranchView& b = cls_branches[s];
        for (int y = 0; y < b.h; ++y) {
            for (int x = 0; x < b.w; ++x) {
                anchor_x->push_back(static_cast<float>(x) + 0.5f);
                anchor_y->push_back(static_cast<float>(y) + 0.5f);
                stride_vec->push_back(static_cast<float>(b.stride));
            }
        }
    }
}

bool pair_head_branches(std::vector<BranchView>* cls_branches,
                        std::vector<BranchView>* reg_branches)
{
    std::sort(cls_branches->begin(), cls_branches->end(),
              [](const BranchView& a, const BranchView& b) {
                  return (a.w * a.h) > (b.w * b.h);
              });

    std::vector<BranchView> paired_reg;
    paired_reg.reserve(cls_branches->size());

    for (size_t i = 0; i < cls_branches->size(); ++i) {
        const BranchView& cls = (*cls_branches)[i];
        bool found = false;
        for (size_t j = 0; j < reg_branches->size(); ++j) {
            const BranchView& reg = (*reg_branches)[j];
            if (reg.w == cls.w && reg.h == cls.h) {
                paired_reg.push_back(reg);
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    reg_branches->swap(paired_reg);
    return true;
}

bool validate_paired_heads(const std::vector<BranchView>& cls_branches,
                           const std::vector<BranchView>& reg_branches,
                           int det_width)
{
    if (cls_branches.size() != 3 || reg_branches.size() != 3) {
        return false;
    }

    const int expected_stride[3] = {8, 16, 32};
    for (int i = 0; i < 3; ++i) {
        const BranchView& cls = cls_branches[i];
        const BranchView& reg = reg_branches[i];
        if (cls.c != NUM_CLASSES || reg.c != REG_CHANNELS) {
            return false;
        }
        if (cls.w != reg.w || cls.h != reg.h) {
            return false;
        }
        if (cls.stride != expected_stride[i] || reg.stride != expected_stride[i]) {
            return false;
        }
        if (cls.w * cls.stride != det_width || cls.h * cls.stride != det_width) {
            return false;
        }
    }
    return true;
}

bool box_center_inside(const std::array<float, 4>& inner,
                       const std::array<float, 4>& outer)
{
    const float cx = 0.5f * (inner[0] + inner[2]);
    const float cy = 0.5f * (inner[1] + inner[3]);
    return cx >= outer[0] && cx <= outer[2] && cy >= outer[1] && cy <= outer[3];
}

int suppress_coarse_obstacle_boxes(DetectionResult* result, int img_w, int img_h)
{
    if (result == nullptr || result->items.size() < 2) {
        return 0;
    }

    std::vector<int> drop(result->items.size(), 0);
    for (size_t i = 0; i < result->items.size(); ++i) {
        const DetectionItem& big = result->items[i];
        if (!obstacle::semantic::IsObstacleClass(big.class_id) ||
            (big.quality != "low" && big.quality != "coarse")) {
            continue;
        }

        const float big_area = box_area_ratio(big.box, img_w, img_h);
        const float big_width = box_width_ratio(big.box, img_w);
        if (big_width < 0.70f && big_area < 0.40f) {
            continue;
        }

        for (size_t j = 0; j < result->items.size(); ++j) {
            if (i == j) {
                continue;
            }
            const DetectionItem& fine = result->items[j];

            const float fine_area = box_area_ratio(fine.box, img_w, img_h);
            const bool much_smaller = fine_area < big_area * 0.72f;
            const bool usable_score = fine.score >= 0.25f || fine.quality == "good";
            const bool person_inside = fine.class_id == DISPLAY_PERSON &&
                                       fine.score >= 0.20f &&
                                       box_center_inside(fine.box, big.box);
            const bool fine_obstacle_inside = obstacle::semantic::IsObstacleClass(fine.class_id) &&
                                              much_smaller &&
                                              usable_score &&
                                              fine.quality != "coarse" &&
                                              box_center_inside(fine.box, big.box);
            if (person_inside || fine_obstacle_inside) {
                drop[i] = 1;
                break;
            }
        }
    }

    if (std::find(drop.begin(), drop.end(), 1) == drop.end()) {
        return 0;
    }

    std::vector<DetectionItem> kept;
    kept.reserve(result->items.size());
    int dropped = 0;
    for (size_t i = 0; i < result->items.size(); ++i) {
        if (!drop[i]) {
            kept.push_back(result->items[i]);
        } else {
            dropped++;
        }
    }
    result->items.swap(kept);
    return dropped;
}

} // namespace

void YOLOV8GRAY::BuildClassNames()
{
    if (NUM_CLASSES == obstacle::semantic::NUM_SEMANTIC_CLASSES) {
        class_names_.clear();
        for (int i = 0; i < obstacle::semantic::NUM_SEMANTIC_CLASSES; ++i) {
            class_names_.push_back(obstacle::semantic::SemanticLabel(i));
        }
        return;
    }

    class_names_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
}

std::string YOLOV8GRAY::ClassIdToLabel(int class_id) const
{
    return obstacle::semantic::RawLabel(class_id);
}

void YOLOV8GRAY::Initialize(std::string& model_path,
                            std::array<int, 2>* in_img_shape,
                            std::array<int, 2>* in_det_shape)
{
    /*
     * 初始化阶段同时固定三套坐标：传感器全图、当前方形 ROI、384x384 模型输入。
     * UPPER/LOWER ROI 的 letterbox 参数独立保存，后处理才能精确反变换回 Aurora。
     */
    img_shape = *in_img_shape;
    det_shape = *in_det_shape;
    output_shape = {
        getenv_int("A1_FULL_FRAME_WIDTH", 720),
        getenv_int("A1_FULL_FRAME_HEIGHT", 1280)
    };

    nms_threshold = 0.60f;
    top_k = std::max(40, getenv_int("A1_NMS_TOP_K", 300));
    keep_top_k = std::max(6, getenv_int("A1_NMS_KEEP_TOP_K", 40));
    dual_roi_ = getenv_flag("A1_DUAL_ROI", true);
    predict_count_ = 0;
    active_view_ = 0;

    BuildClassNames();

    const int roi_size = std::min(img_shape[0], img_shape[1]);
    const int upper_y = std::max(0, getenv_int("A1_ROI_UPPER_Y", 0));
    const int lower_default = std::max(0, img_shape[1] - roi_size);
    const int lower_y = std::max(0, std::min(img_shape[1] - roi_size,
                                             getenv_int("A1_ROI_LOWER_Y", lower_default)));
    roi_[0] = {0, upper_y, img_shape[0], std::min(img_shape[1], upper_y + roi_size)};
    roi_[1] = dual_roi_ ? std::array<int, 4>{0, lower_y, img_shape[0], lower_y + roi_size}
                        : std::array<int, 4>{0, 0, img_shape[0], img_shape[1]};
    if (!dual_roi_) roi_[0] = roi_[1];

    for (int view = 0; view < 2; ++view) {
        const int roi_w = roi_[view][2] - roi_[view][0];
        const int roi_h = roi_[view][3] - roi_[view][1];
        lb_info_[view].src_w = roi_w;
        lb_info_[view].src_h = roi_h;
        lb_info_[view].dst_w = det_shape[0];
        lb_info_[view].dst_h = det_shape[1];
        lb_info_[view].scale = std::min(static_cast<float>(det_shape[0]) / roi_w,
                                        static_cast<float>(det_shape[1]) / roi_h);
        const int resize_w = static_cast<int>(std::round(roi_w * lb_info_[view].scale));
        const int resize_h = static_cast<int>(std::round(roi_h * lb_info_[view].scale));
        lb_info_[view].pad_x = (det_shape[0] - resize_w) / 2;
        lb_info_[view].pad_y = (det_shape[1] - resize_h) / 2;
    }

    std::cout << "[YOLOV8GRAY][INFO] Initialize start" << std::endl;
    std::cout << "[YOLOV8GRAY][INFO] model path: " << model_path << std::endl;
    std::cout << "[YOLOV8GRAY][INFO] src shape: " << img_shape[0] << "x" << img_shape[1] << std::endl;
    std::cout << "[YOLOV8GRAY][INFO] output shape: " << output_shape[0] << "x" << output_shape[1]
              << ", dual_roi=" << (dual_roi_ ? 1 : 0) << std::endl;
    std::cout << "[YOLOV8GRAY][INFO] det shape: " << det_shape[0] << "x" << det_shape[1] << std::endl;
    std::cout << "[YOLOV8GRAY][INFO] model class count=" << NUM_CLASSES
              << ", semantic classes=" << obstacle::semantic::NUM_SEMANTIC_CLASSES
              << ", input channels=" << A1_YOLO_INPUT_CHANNELS
              << std::endl;
    for (int view = 0; view < 2; ++view) {
        std::cout << "[YOLOV8GRAY][INFO] view=" << view
                  << " roi=(" << roi_[view][0] << "," << roi_[view][1]
                  << "," << roi_[view][2] << "," << roi_[view][3] << ")"
                  << " scale=" << lb_info_[view].scale
                  << " pad=(" << lb_info_[view].pad_x << "," << lb_info_[view].pad_y << ")"
                  << std::endl;
    }

    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);

    int mean[3] = {0, 0, 0};
    int stdv[3] = {0, 0, 0};
    int is_uint8 = 0;
    int dtype = -1;

    ssne_get_model_normalize_params(model_id, mean, stdv, &is_uint8);
    ssne_get_model_input_dtype(model_id, &dtype);

    std::cout << "[YOLOV8GRAY][INFO] model normalize mean=("
              << mean[0] << "," << mean[1] << "," << mean[2] << ") std=("
              << stdv[0] << "," << stdv[1] << "," << stdv[2] << ")"
              << ", is_uint8=" << is_uint8
              << ", input_dtype=" << dtype
              << std::endl;

    const uint32_t det_w = static_cast<uint32_t>(det_shape[0]);
    const uint32_t det_h = static_cast<uint32_t>(det_shape[1]);
    const uint8_t input_format = (A1_YOLO_INPUT_CHANNELS == 1) ? SSNE_Y_8 : SSNE_BGR;
    inputs[0] = create_tensor(det_w, det_h, input_format, SSNE_BUF_AI);
    std::cout << "[YOLOV8GRAY][INFO] create input tensor format="
              << ((A1_YOLO_INPUT_CHANNELS == 1) ? "SSNE_Y_8" : "SSNE_BGR")
              << std::endl;

    for (int view = 0; view < 2; ++view) {
        pipe_offline_[view] = GetAIPreprocessPipe();
        Clear(pipe_offline_[view]);
        const int resize_w = static_cast<int>(std::round(lb_info_[view].src_w * lb_info_[view].scale));
        const int resize_h = static_cast<int>(std::round(lb_info_[view].src_h * lb_info_[view].scale));
        const int pad_right = det_shape[0] - resize_w - lb_info_[view].pad_x;
        const int pad_bottom = det_shape[1] - resize_h - lb_info_[view].pad_y;
        int ret = SetCrop(pipe_offline_[view],
                          static_cast<uint16_t>(roi_[view][0]),
                          static_cast<uint16_t>(roi_[view][1]),
                          static_cast<uint16_t>(roi_[view][2]),
                          static_cast<uint16_t>(roi_[view][3]));
        std::cout << "[YOLOV8GRAY][INFO] view=" << view << " SetCrop ret=" << ret << std::endl;
        ret = SetPadding2(pipe_offline_[view],
                          static_cast<uint16_t>(lb_info_[view].pad_x),
                          static_cast<uint16_t>(lb_info_[view].pad_y),
                          static_cast<uint16_t>(pad_right),
                          static_cast<uint16_t>(pad_bottom), 114);
        std::cout << "[YOLOV8GRAY][INFO] view=" << view << " SetPadding2 ret=" << ret << std::endl;
        ret = SetNormalize(pipe_offline_[view], model_id);
        std::cout << "[YOLOV8GRAY][INFO] view=" << view << " SetNormalize ret=" << ret << std::endl;
    }
}

bool YOLOV8GRAY::Preprocess(ssne_tensor_t* img_in, ssne_tensor_t* input_tensor)
{
    // A1 离线管线一次完成当前 ROI crop、等比例缩放、114 padding 和模型归一化。
    int ret = RunAiPreprocessPipe(pipe_offline_[active_view_], *img_in, *input_tensor);
    if (ret != 0) {
        std::cout << "[YOLOV8GRAY][ERROR] RunAiPreprocessPipe failed, ret=" << ret << std::endl;
        return false;
    }
    if (A1_YOLO_INPUT_CHANNELS == 1) {
        apply_adaptive_gray_lut(input_tensor, predict_count_);
    }

    static bool dumped[2] = {false, false};
    if (getenv_flag("A1_DUMP_PREPROCESS_ONCE", false) && !dumped[active_view_]) {
        const std::string dump_base = getenv_string("A1_PREPROCESS_DUMP_PATH", "/tmp/yolov8_input");
        const std::string dump_path = dump_base + "_view" +
                                      std::to_string(active_view_) + ".bin";
        int save_ret = save_tensor_buffer(*input_tensor, dump_path.c_str());
        std::cout << "[YOLOV8GRAY][DEBUG] preprocess dump ret=" << save_ret
                  << ", path=" << dump_path << std::endl;
        dumped[active_view_] = true;
    }
    return true;
}

void YOLOV8GRAY::MapBoxToOriginalImage(std::array<float, 4>& box)
{
    // 先去 letterbox padding/scale，再加 ROI 左上角偏移，最后裁剪到传感器全图。
    const LetterboxInfo& lb = lb_info_[active_view_];
    float x1 = (box[0] - static_cast<float>(lb.pad_x)) / lb.scale;
    float y1 = (box[1] - static_cast<float>(lb.pad_y)) / lb.scale;
    float x2 = (box[2] - static_cast<float>(lb.pad_x)) / lb.scale;
    float y2 = (box[3] - static_cast<float>(lb.pad_y)) / lb.scale;

    x1 = clampf(x1, 0.0f, static_cast<float>(lb.src_w - 1));
    y1 = clampf(y1, 0.0f, static_cast<float>(lb.src_h - 1));
    x2 = clampf(x2, 0.0f, static_cast<float>(lb.src_w - 1));
    y2 = clampf(y2, 0.0f, static_cast<float>(lb.src_h - 1));

    box[0] = clampf(x1 + static_cast<float>(roi_[active_view_][0]),
                    0.0f,
                    static_cast<float>(output_shape[0] - 1));
    box[1] = clampf(y1 + static_cast<float>(roi_[active_view_][1]),
                    0.0f,
                    static_cast<float>(output_shape[1] - 1));
    box[2] = clampf(x2 + static_cast<float>(roi_[active_view_][0]),
                    0.0f,
                    static_cast<float>(output_shape[0] - 1));
    box[3] = clampf(y2 + static_cast<float>(roi_[active_view_][1]),
                    0.0f,
                    static_cast<float>(output_shape[1] - 1));
}

bool YOLOV8GRAY::Postprocess(DetectionResult* result, float conf_threshold)
{
    /*
     * 后处理顺序固定为：识别并配对六个 head -> 扫描分类 logit -> 对通过阈值的
     * anchor 解 DFL -> 几何/语义质量过滤 -> 全图映射 -> MultiTargetNMS。
     * 任一 head 的通道、stride、dtype 或布局不满足契约即整帧失败，禁止带错位框运行。
     */
    static int postprocess_frame_count = 0;
    ++postprocess_frame_count;
    const bool debug_post = getenv_flag("A1_DEBUG_POSTPROCESS", false);
    const int debug_interval = std::max(1, getenv_int("A1_DEBUG_POSTPROCESS_INTERVAL", 30));

    std::vector<BranchView> cls_branches;
    std::vector<BranchView> reg_branches;
    cls_branches.reserve(3);
    reg_branches.reserve(3);

    static bool meta_printed = false;
    bool invalid_output = false;

    for (int i = 0; i < 6; ++i) {
        BranchView b;
        const bool ok = make_branch_view(outputs[i], i, det_shape[0], &b);

        if (!meta_printed) {
            std::cout << "[YOLOV8GRAY][META] out" << i
                      << ": ok=" << (ok ? 1 : 0)
                      << ", w=" << b.w
                      << ", h=" << b.h
                      << ", raw_total=" << b.raw_total
                      << ", elements=" << b.element_count
                      << ", dtype=" << static_cast<int>(b.dtype)
                      << ", fmt=" << static_cast<int>(b.fmt)
                      << ", inferred_c=" << b.c
                      << ", stride=" << b.stride
                      << std::endl;
        }

        if (!ok) {
            invalid_output = true;
            continue;
        }

        if (b.is_cls) {
            cls_branches.push_back(b);
        } else if (b.c == REG_CHANNELS) {
            reg_branches.push_back(b);
        }
    }

    meta_printed = true;

    const bool paired = pair_head_branches(&cls_branches, &reg_branches);
    const bool valid_heads = paired && validate_paired_heads(cls_branches, reg_branches, det_shape[0]);
    if (invalid_output || !valid_heads) {
        std::cout << "[YOLOV8GRAY][ERROR] output branch grouping failed: cls="
                  << cls_branches.size() << ", reg=" << reg_branches.size()
                  << ". Expected 3 cls heads with " << NUM_CLASSES
                  << " channels and 3 DFL heads with " << REG_CHANNELS
                  << " channels at strides 8/16/32. Check dtype/shape/order of the converted .m1model."
                  << std::endl;
        result->Clear();
        return false;
    }

    if (debug_post && (postprocess_frame_count == 1 || postprocess_frame_count % debug_interval == 0)) {
        std::cout << "[YOLOV8GRAY][DEBUG] paired heads:";
        for (size_t i = 0; i < cls_branches.size(); ++i) {
            std::cout << " s" << i
                      << "(cls_out=" << cls_branches[i].out_idx
                      << ",reg_out=" << reg_branches[i].out_idx
                      << ",wh=" << cls_branches[i].w << "x" << cls_branches[i].h
                      << ",stride=" << cls_branches[i].stride << ")";
        }
        std::cout << std::endl;
    }

    static bool once = false;
    if (!once) {
        std::cout << "[YOLOV8GRAY][META] decode = "
                  << (read_model_output_as_hwc() ? "HWC" : "CHW")
                  << " + sigmoid + DFL + anchor decode + reverse letterbox" << std::endl;
        once = true;
    }

    int total_points = 0;
    for (size_t s = 0; s < cls_branches.size(); ++s) {
        total_points += cls_branches[s].w * cls_branches[s].h;
    }

    result->Clear();
    result->items.reserve(static_cast<size_t>(top_k));

    const int frame_w = output_shape[0];
    const int frame_h = output_shape[1];
    std::vector<int> debug_best_raw_counts;
    std::vector<int> debug_threshold_raw_counts;
    std::vector<int> debug_kept_raw_counts;
    if (debug_post) {
        debug_best_raw_counts.assign(NUM_CLASSES, 0);
        debug_threshold_raw_counts.assign(NUM_CLASSES, 0);
        debug_kept_raw_counts.assign(NUM_CLASSES, 0);
    }

    static std::vector<float> threshold_logits;
    int invalid_dfl_count = 0;
    if (threshold_logits.size() != static_cast<size_t>(NUM_CLASSES)) {
        threshold_logits.resize(NUM_CLASSES);
        for (int raw_cls = 0; raw_cls < NUM_CLASSES; ++raw_cls) {
            threshold_logits[raw_cls] =
                probability_to_logit(candidate_threshold_for_class(raw_cls));
        }
    }

    // 一个 anchor 完成 DFL 后的统一出口：构造语义、反映射，并拒绝饱和横框等伪框。
    const auto append_candidate = [&](int raw_cls,
                                      float score,
                                      const std::array<float, 4>& decoded_box) {
        const int display_cls = obstacle::semantic::SemanticClassFromRaw(raw_cls);
        DetectionItem item;
        item.box = decoded_box;
        item.score = score;
        item.class_id = display_cls;
        item.raw_class_id = raw_cls;
        item.raw_label = ClassIdToLabel(raw_cls);
        item.label = obstacle::semantic::SemanticLabel(display_cls);
        item.semantic_class = item.label;
        item.risk_weight = obstacle::semantic::RiskWeight(display_cls);
        MapBoxToOriginalImage(item.box);

        const float bw = item.box[2] - item.box[0];
        const float bh = item.box[3] - item.box[1];
        if (bw < 8.0f || bh < 10.0f) return;
        const float area_ratio = box_area_ratio(item.box, frame_w, frame_h);
        const float width_ratio = box_width_ratio(item.box, frame_w);
        const float height_ratio = box_height_ratio(item.box, frame_h);
        const float center_y = box_center_y_ratio(item.box, frame_h);
        const float bottom_ratio =
            item.box[3] / std::max(1.0f, static_cast<float>(frame_h));
        const int touch = count_touch_borders(item.box, frame_w, frame_h);
        if (area_ratio > 0.98f || (width_ratio > 0.98f && height_ratio > 0.98f)) return;

        const std::array<int, 4>& active_roi = roi_[active_view_];
        const float roi_w = std::max(1, active_roi[2] - active_roi[0]);
        const float roi_h = std::max(1, active_roi[3] - active_roi[1]);
        const float roi_width_ratio = bw / roi_w;
        const float roi_height_ratio = bh / roi_h;
        const bool roi_left = item.box[0] <= active_roi[0] + 12.0f;
        const bool roi_right = item.box[2] >= active_roi[2] - 12.0f;
        const bool roi_top = item.box[1] <= active_roi[1] + 12.0f;
        const bool roi_bottom = item.box[3] >= active_roi[3] - 12.0f;
        const bool roi_saturated_box =
            roi_width_ratio > 0.94f ||
            (roi_width_ratio > 0.82f && roi_height_ratio > 0.72f) ||
            (roi_height_ratio > 0.94f && roi_width_ratio > 0.65f) ||
            (roi_left && roi_right && roi_width_ratio > 0.88f) ||
            (roi_top && roi_bottom && roi_width_ratio > 0.72f);
        if (roi_saturated_box) {
            ++result->coarse_drop_count;
            if (debug_post &&
                postprocess_frame_count % debug_interval == 0 &&
                result->coarse_drop_count <= 3) {
                std::cout << "[YOLOV8GRAY][DROP] roi_saturated raw="
                          << item.raw_label << " conf=" << item.score
                          << " roi_ratio=" << roi_width_ratio << "x" << roi_height_ratio
                          << " box=(" << item.box[0] << "," << item.box[1]
                          << "," << item.box[2] << "," << item.box[3] << ")"
                          << " view=" << active_view_ << std::endl;
            }
            return;
        }

        const bool clips_both_horizontal_borders =
            item.box[0] <= 3.0f && item.box[2] >= static_cast<float>(frame_w - 4);
        const bool saturated_wide_box =
            clips_both_horizontal_borders && width_ratio > 0.985f;
        if (saturated_wide_box) {
            ++result->coarse_drop_count;
            if (debug_post &&
                postprocess_frame_count % debug_interval == 0 &&
                result->coarse_drop_count <= 3) {
                std::cout << "[YOLOV8GRAY][DROP] saturated_wide raw="
                          << item.raw_label << " conf=" << item.score
                          << " box=(" << item.box[0] << "," << item.box[1]
                          << "," << item.box[2] << "," << item.box[3] << ")"
                          << " view=" << active_view_ << std::endl;
            }
            return;
        }

        const bool wide_flat_midframe =
            width_ratio > 0.42f && height_ratio < 0.55f &&
            bottom_ratio < 0.88f && width_ratio > height_ratio * 1.05f;
        // 不按室内人体框的长宽比过滤普通 person，只剔除极弱且异常宽的背景响应，
        // 以保留腿部、躯干等不完整人体观测。
        if (item.class_id == DISPLAY_PERSON && score < 0.35f &&
            width_ratio > 0.65f && wide_flat_midframe) {
            return;
        }
        if (item.class_id != DISPLAY_PERSON && center_y < 0.05f && score < 0.24f) {
            return;
        }
        if (score < 0.22f && (area_ratio > 0.85f ||
            (width_ratio > 0.98f && touch >= 3))) {
            return;
        }

        const bool implausible_indoor_vehicle =
            obstacle::semantic::IsVehicleSemantic(item.class_id) &&
            bottom_ratio < 0.74f &&
            (center_y < 0.45f || height_ratio > width_ratio * 1.15f) &&
            score < 0.70f;
        if (implausible_indoor_vehicle ||
            (obstacle::semantic::IsVehicleSemantic(item.class_id) &&
             wide_flat_midframe && score < 0.55f)) {
            item.class_id = obstacle::semantic::GENERIC_OBSTACLE;
            item.label = obstacle::semantic::SemanticLabel(item.class_id);
            item.semantic_class = item.label;
            item.risk_weight = obstacle::semantic::RiskWeight(item.class_id);
        }

        item.sector = sector_from_box(item.box, frame_w);
        item.quality = quality_from_box(item.box, frame_w, frame_h,
                                        item.class_id, raw_cls, item.score);
        if (clips_both_horizontal_borders ||
            (width_ratio > 0.90f && touch >= 2)) {
            item.quality = "coarse";
        }
        const bool unreliable_wide_coarse =
            obstacle::semantic::IsObstacleClass(item.class_id) &&
            !is_indoor_rigid_raw_class(raw_cls) &&
            wide_flat_midframe && width_ratio > 0.62f &&
            bottom_ratio < 0.82f && score < 0.40f;
        if (unreliable_wide_coarse) {
            ++result->coarse_drop_count;
            return;
        }
        item.risk_level = "unknown";
        result->items.push_back(item);
        if (debug_post && raw_cls >= 0 &&
            raw_cls < static_cast<int>(debug_kept_raw_counts.size())) {
            ++debug_kept_raw_counts[raw_cls];
        }
    };

    /*
     * 原地扫描分类 tensor，每个 anchor 保留 top-1，并额外保护可能被其他类别压过的
     * person。只有分类过阈值才计算四边 DFL，显著减少空场景中的指数运算。
     */
    for (size_t scale = 0; scale < cls_branches.size(); ++scale) {
        const BranchView& cb = cls_branches[scale];
        const BranchView& rb = reg_branches[scale];
        for (int y = 0; y < cb.h; ++y) {
            for (int x = 0; x < cb.w; ++x) {
                int best_cls = -1;
                int person_cls = -1;
                float best_logit = -FLT_MAX;
                float person_logit = -FLT_MAX;
                for (int raw_cls = 0; raw_cls < NUM_CLASSES; ++raw_cls) {
                    if (!is_supported_raw_class(raw_cls)) continue;
                    const float logit = read_branch_value(cb, raw_cls, y, x);
                    if (!std::isfinite(logit)) continue;
                    if (logit > best_logit) {
                        best_logit = logit;
                        best_cls = raw_cls;
                    }
                    if (is_person_raw_class(raw_cls) && logit > person_logit) {
                        person_logit = logit;
                        person_cls = raw_cls;
                    }
                }
                if (debug_post && best_cls >= 0) {
                    ++debug_best_raw_counts[best_cls];
                }

                const bool keep_best = best_cls >= 0 &&
                    best_logit >= threshold_logits[best_cls];
                const float best_score = keep_best ? fast_sigmoid(best_logit) : 0.0f;
                const float person_score = person_cls >= 0
                    ? fast_sigmoid(person_logit) : 0.0f;
                // ROD25 对被截断的人体响应偏弱。保留相对可信的 person 次优分支，
                // 后续仍需经过几何过滤和至少两帧轨迹确认，不把弱单帧直接用于规划。
                const bool keep_person = person_cls >= 0 && person_cls != best_cls &&
                    person_logit >= threshold_logits[person_cls] &&
                    person_score >= std::max(0.08f, best_score * 0.30f);
                if (!keep_best && !keep_person) continue;

                if (debug_post) {
                    if (keep_best) ++debug_threshold_raw_counts[best_cls];
                    if (keep_person) ++debug_threshold_raw_counts[person_cls];
                }

                const float l = decode_dfl_side(rb, 0, y, x);
                const float t = decode_dfl_side(rb, 1, y, x);
                const float r = decode_dfl_side(rb, 2, y, x);
                const float bottom = decode_dfl_side(rb, 3, y, x);
                if (l < 0.0f || t < 0.0f || r < 0.0f || bottom < 0.0f) {
                    ++invalid_dfl_count;
                    continue;
                }
                const float anchor_x = static_cast<float>(x) + 0.5f;
                const float anchor_y = static_cast<float>(y) + 0.5f;
                const float stride = static_cast<float>(cb.stride);
                std::array<float, 4> box = {
                    clampf((anchor_x - l) * stride, 0.0f,
                           static_cast<float>(det_shape[0] - 1)),
                    clampf((anchor_y - t) * stride, 0.0f,
                           static_cast<float>(det_shape[1] - 1)),
                    clampf((anchor_x + r) * stride, 0.0f,
                           static_cast<float>(det_shape[0] - 1)),
                    clampf((anchor_y + bottom) * stride, 0.0f,
                           static_cast<float>(det_shape[1] - 1))
                };
                if (box[2] <= box[0] || box[3] <= box[1]) continue;
                if (keep_best) append_candidate(best_cls, best_score, box);
                if (keep_person) append_candidate(person_cls, person_score, box);
            }
        }
    }

    if (invalid_dfl_count > 32) {
        std::cout << "[YOLOV8GRAY][ERROR] non-finite DFL values="
                  << invalid_dfl_count << std::endl;
        result->Clear();
        return false;
    }

    result->raw_candidate_count = static_cast<int>(result->items.size());
    if (result->items.empty()) {
        if (debug_post && (postprocess_frame_count == 1 || postprocess_frame_count % debug_interval == 0)) {
            std::cout << "[YOLOV8GRAY][DEBUG] frame=" << postprocess_frame_count
                      << " total_points=" << total_points
                      << " raw_candidates=0 post_nms=0"
                      << " conf_threshold=" << conf_threshold << std::endl;
            if (!debug_best_raw_counts.empty()) {
                std::cout << "[YOLOV8GRAY][DEBUG] raw_best_top="
                          << top_class_counts_text(debug_best_raw_counts, 6)
                          << " raw_threshold_top="
                          << top_class_counts_text(debug_threshold_raw_counts, 6)
                          << " raw_kept_top=none"
                          << std::endl;
            }
        }
        return true;
    }

    utils::MultiTargetNMS(result, nms_threshold, top_k);
    result->post_nms_count = static_cast<int>(result->items.size());
    result->coarse_drop_count += suppress_coarse_obstacle_boxes(result, frame_w, frame_h);
    utils::SortDetectionResult(result);

    if (static_cast<int>(result->items.size()) > keep_top_k) {
        result->items.resize(keep_top_k);
    }

    if (debug_post && (postprocess_frame_count == 1 || postprocess_frame_count % debug_interval == 0)) {
        std::cout << "[YOLOV8GRAY][DEBUG] frame=" << postprocess_frame_count
                  << " total_points=" << total_points
                  << " raw_candidates=" << result->raw_candidate_count
                  << " post_nms=" << result->post_nms_count
                  << " kept=" << result->items.size()
                  << " conf_threshold=" << conf_threshold
                  << " nms_threshold=" << nms_threshold
                  << " top_k=" << top_k
                  << " keep_top_k=" << keep_top_k
                  << std::endl;
        if (!debug_best_raw_counts.empty()) {
            std::cout << "[YOLOV8GRAY][DEBUG] raw_best_top="
                      << top_class_counts_text(debug_best_raw_counts, 6)
                      << " raw_threshold_top="
                      << top_class_counts_text(debug_threshold_raw_counts, 6)
                      << " raw_kept_top="
                      << top_class_counts_text(debug_kept_raw_counts, 6)
                      << std::endl;
        }
    }
    return true;
}

bool YOLOV8GRAY::Predict(ssne_tensor_t* img_in,
                         DetectionResult* result,
                         float conf_threshold)
{
    /*
     * 单帧同步推理入口。每次只选择一个 ROI，依次执行预处理、NPU inference、
     * 六输出获取和 CPU 后处理，并记录各阶段耗时供 SystemHealth 统计。
     */
    if (result == nullptr) {
        std::cout << "[YOLOV8GRAY][ERROR] result is nullptr" << std::endl;
        return false;
    }

    const auto elapsed_ms = [](const std::chrono::steady_clock::time_point& start) {
        return std::chrono::duration_cast<std::chrono::duration<float, std::milli> >(
            std::chrono::steady_clock::now() - start).count();
    };
    active_view_ = dual_roi_ ? (predict_count_ & 1) : 0;
    ++predict_count_;
    std::chrono::steady_clock::time_point stage_start = std::chrono::steady_clock::now();
    if (!Preprocess(img_in, &inputs[0])) {
        result->Clear();
        return false;
    }
    last_timing_.preprocess_ms = elapsed_ms(stage_start);

    stage_start = std::chrono::steady_clock::now();
    if (ssne_inference(model_id, 1, inputs)) {
        std::cout << "[YOLOV8GRAY][ERROR] ssne_inference failed" << std::endl;
        result->Clear();
        return false;
    }
    last_timing_.inference_ms = elapsed_ms(stage_start);

    stage_start = std::chrono::steady_clock::now();
    if (ssne_getoutput(model_id, 6, outputs)) {
        std::cout << "[YOLOV8GRAY][ERROR] ssne_getoutput failed" << std::endl;
        result->Clear();
        return false;
    }
    last_timing_.output_ms = elapsed_ms(stage_start);

    static bool dumped_heads = false;
    if (getenv_flag("A1_DUMP_HEADS_ONCE", false) && !dumped_heads) {
        const std::string dump_dir = getenv_string("A1_HEAD_DUMP_PREFIX", "/tmp/yolov8_head");
        for (int i = 0; i < 6; ++i) {
            const std::string path = dump_dir + "_out" + std::to_string(i) + ".bin";
            const int save_ret = save_tensor_buffer(outputs[i], path.c_str());
            std::cout << "[YOLOV8GRAY][DEBUG] head dump out=" << i
                      << " ret=" << save_ret << " path=" << path << std::endl;
        }
        dumped_heads = true;
    }

    stage_start = std::chrono::steady_clock::now();
    const bool ok = Postprocess(result, conf_threshold);
    last_timing_.postprocess_ms = elapsed_ms(stage_start);
    result->view_id = active_view_;
    result->roi = roi_[active_view_];
    result->timestamp_ms = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    return ok;
}

void YOLOV8GRAY::Release()
{
    release_tensor(inputs[0]);

    for (int i = 0; i < 6; ++i) {
        release_tensor(outputs[i]);
    }

    for (int view = 0; view < 2; ++view) {
        if (pipe_offline_[view] != NULL) {
            ReleaseAIPreprocessPipe(pipe_offline_[view]);
            pipe_offline_[view] = NULL;
        }
    }
}
