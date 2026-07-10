#include "../include/utils.hpp"

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

namespace {

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

std::string getenv_string(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::string(value);
}

// ONNX head outputs are NCHW, while converted .m1model tensors are exposed
// through image-like runtime buffers. Most A1 head6 conversions need HWC
// reads; keep a runtime override for board diagnosis if boxes drift.
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
          camera_height_m(getenv_float("A1_CAM_HEIGHT_M", 0.85f)),
          camera_pitch_down_deg(getenv_float("A1_CAM_PITCH_DOWN_DEG", 15.0f)),
          min_distance_m(getenv_float("A1_DIST_MIN_M", 0.2f)),
          max_distance_m(getenv_float("A1_DIST_MAX_M", 8.0f)) {}
};

struct BranchView {
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
    const float left_bound = 0.35f * static_cast<float>(img_w);
    const float right_bound = 0.65f * static_cast<float>(img_w);
    const float width = std::max(1.0f, box[2] - box[0]);

    if (width / std::max(1.0f, static_cast<float>(img_w)) > 0.75f) {
        return "wide";
    }

    const float left_overlap = std::max(0.0f, std::min(box[2], left_bound) - std::max(box[0], 0.0f)) / width;
    const float center_overlap = std::max(0.0f, std::min(box[2], right_bound) - std::max(box[0], left_bound)) / width;
    const float right_overlap = std::max(0.0f, std::min(box[2], static_cast<float>(img_w)) - std::max(box[0], right_bound)) / width;

    if (left_overlap >= 0.20f && center_overlap >= 0.20f && right_overlap >= 0.20f) {
        return "wide";
    }
    if (center_overlap >= 0.50f) {
        return "center";
    }
    if (center_overlap >= 0.25f && left_overlap >= 0.25f) return "left_center";
    if (center_overlap >= 0.25f && right_overlap >= 0.25f) return "center_right";

    const float cx = 0.5f * (box[0] + box[2]);
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

std::string quality_from_box(const std::array<float, 4>& box,
                             int img_w,
                             int img_h,
                             int display_cls,
                             float score)
{
    const float area_ratio = box_area_ratio(box, img_w, img_h);
    const float width_ratio = box_width_ratio(box, img_w);
    const float height_ratio = box_height_ratio(box, img_h);
    const int touch = count_touch_borders(box, img_w, img_h);

    if (obstacle::semantic::IsObstacleClass(display_cls) &&
        score < 0.28f &&
        (width_ratio > 0.88f || area_ratio > 0.70f) &&
        touch >= 2) {
        return "coarse";
    }

    // Wide, low-detail obstacle boxes often come from shelves, screens, rails,
    // or merged background structures. Keep them available as navigation
    // evidence, but mark them coarse so NMS/tracker prefer finer object boxes.
    if (obstacle::semantic::IsObstacleClass(display_cls) &&
        !obstacle::semantic::IsFurnitureLikeSemantic(display_cls) &&
        score < 0.55f &&
        width_ratio > 0.62f &&
        height_ratio < 0.48f &&
        area_ratio < 0.45f) {
        return "coarse";
    }

    if (obstacle::semantic::IsObstacleClass(display_cls) &&
        display_cls != obstacle::semantic::CHAIR_SEAT &&
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

    // A partial target that touches borders or covers most of the image is not
    // a clean size-prior observation; keep it as a nearfield/risk cue instead.
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

// Decodes one DFL side only after the anchor passed the class threshold.
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
                 …3506 tokens truncated…                          static_cast<uint16_t>(roi_[view][1]),
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
    int ret = RunAiPreprocessPipe(pipe_offline_[active_view_], *img_in, *input_tensor);
    if (ret != 0) {
        std::cout << "[YOLOV8GRAY][ERROR] RunAiPreprocessPipe failed, ret=" << ret << std::endl;
        return false;
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

    // Applies geometry and semantic sanity checks after one anchor was decoded.
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

        const bool wide_flat_midframe =
            width_ratio > 0.42f && height_ratio < 0.55f &&
            bottom_ratio < 0.88f && width_ratio > height_ratio * 1.05f;
        // Do not reject ordinary person boxes based on indoor aspect ratio.
        // Only remove extremely weak, wide background responses.
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
                                        item.class_id, item.score);
        const bool unreliable_wide_coarse =
            obstacle::semantic::IsObstacleClass(item.class_id) &&
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

    // Scan class logits in-place. DFL is evaluated only for anchors that pass
    // a class threshold, avoiding more than 190k exponentials on empty frames.
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
                const bool keep_person = person_cls >= 0 && person_cls != best_cls &&
                    person_logit >= threshold_logits[person_cls] &&
                    person_score >= std::max(0.13f, best_score * 0.55f);
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
