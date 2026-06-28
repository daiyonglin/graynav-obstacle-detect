#include "../include/utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

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

// ONNX head outputs are NCHW, but SSNE runtime tensors are exposed as
// image-like width/height buffers. On board the converted .m1model outputs are
// observed through that runtime layout, so read C as the innermost dimension.
constexpr bool kReadModelOutputAsHwc = true;

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

bool is_furniture_raw_class(int cls_id)
{
    return obstacle::semantic::IsFurnitureLikeRawClass(cls_id);
}

float candidate_threshold_for_class(int raw_cls)
{
    return obstacle::semantic::CandidateThreshold(raw_cls);
}

void push_unique_candidate(std::vector<ClassCandidate>* candidates, int raw_cls, float score)
{
    if (raw_cls < 0) {
        return;
    }
    for (size_t i = 0; i < candidates->size(); ++i) {
        if ((*candidates)[i].raw_cls == raw_cls) {
            if (score > (*candidates)[i].score) {
                (*candidates)[i].score = score;
            }
            return;
        }
    }
    ClassCandidate c;
    c.raw_cls = raw_cls;
    c.score = score;
    candidates->push_back(c);
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
    } else if (source == "fused") {
        confidence += 0.10f;
    } else if (source == "size") {
        confidence -= (touch > 0 || area_ratio > 0.35f) ? 0.25f : 0.05f;
    } else if (source == "nearfield") {
        confidence = 0.42f;
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

    if (prefer_ground_distance(raw_class_id)) {
        return true;
    }

    // A tabletop class that covers most of the frame is usually a nearby
    // partial object, not a clean size-prior observation.
    // Rejecting the size prior here prevents huge laptop/tv boxes from
    // collapsing to unrealistically small distances.
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
    if (size_m >= 0.0f && reject_size_distance_for_box(box, img_w, img_h, raw_class_id)) {
        size_m = -1.0f;
    }

    if (ground_m >= 0.0f && size_m >= 0.0f) {
        const float ratio = size_m / std::max(0.01f, ground_m);
        if (ratio < 0.55f || ratio > 1.80f) {
            *source = "ground";
            return ground_m;
        }
    }

    float fused = fuse_distance_m(ground_m, size_m, source);
    const float bottom_ratio = box[3] / std::max(1.0f, static_cast<float>(img_h));
    if (fused < 0.0f && bottom_ratio > 0.92f) {
        *source = "nearfield";
        return 0.60f;
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
    if (kReadModelOutputAsHwc) {
        return b.data[(y * b.w + x) * b.c + c];
    }
    return b.data[c * b.h * b.w + y * b.w + x];
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
    img_shape = *in_img_shape;
    det_shape = *in_det_shape;

    nms_threshold = 0.60f;
    top_k = 800;
    keep_top_k = 80;

    BuildClassNames();

    lb_info_.src_w = img_shape[0];
    lb_info_.src_h = img_shape[1];
    lb_info_.dst_w = det_shape[0];
    lb_info_.dst_h = det_shape[1];

    const float scale_w = static_cast<float>(det_shape[0]) / static_cast<float>(img_shape[0]);
    const float scale_h = static_cast<float>(det_shape[1]) / static_cast<float>(img_shape[1]);
    lb_info_.scale = std::min(scale_w, scale_h);

    const int resize_w = static_cast<int>(std::round(img_shape[0] * lb_info_.scale));
    const int resize_h = static_cast<int>(std::round(img_shape[1] * lb_info_.scale));

    lb_info_.pad_x = (det_shape[0] - resize_w) / 2;
    lb_info_.pad_y = (det_shape[1] - resize_h) / 2;

    const int pad_right  = det_shape[0] - resize_w - lb_info_.pad_x;
    const int pad_bottom = det_shape[1] - resize_h - lb_info_.pad_y;

    std::cout << "[YOLOV8GRAY][INFO] Initialize start" << std::endl;
    std::cout << "[YOLOV8GRAY][INFO] model path: " << model_path << std::endl;
    std::cout << "[YOLOV8GRAY][INFO] src shape: " << img_shape[0] << "x" << img_shape[1] << std::endl;
    std::cout << "[YOLOV8GRAY][INFO] det shape: " << det_shape[0] << "x" << det_shape[1] << std::endl;
    std::cout << "[YOLOV8GRAY][INFO] model class count=" << NUM_CLASSES
              << ", semantic classes=" << obstacle::semantic::NUM_SEMANTIC_CLASSES
              << std::endl;
    std::cout << "[YOLOV8GRAY][INFO] letterbox scale=" << lb_info_.scale
              << ", pad_x=" << lb_info_.pad_x
              << ", pad_y=" << lb_info_.pad_y
              << ", pad_right=" << pad_right
              << ", pad_bottom=" << pad_bottom << std::endl;

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

    pipe_offline = GetAIPreprocessPipe();

    const uint32_t det_w = static_cast<uint32_t>(det_shape[0]);
    const uint32_t det_h = static_cast<uint32_t>(det_shape[1]);
    inputs[0] = create_tensor(det_w, det_h, SSNE_BGR, SSNE_BUF_AI);

    Clear(pipe_offline);

    int ret = 0;

    ret = SetCrop(pipe_offline, 0, 0,
                  static_cast<uint16_t>(img_shape[0]),
                  static_cast<uint16_t>(img_shape[1]));
    std::cout << "[YOLOV8GRAY][INFO] SetCrop ret=" << ret << std::endl;

    ret = SetPadding2(pipe_offline,
                      static_cast<uint16_t>(lb_info_.pad_x),
                      static_cast<uint16_t>(lb_info_.pad_y),
                      static_cast<uint16_t>(pad_right),
                      static_cast<uint16_t>(pad_bottom),
                      114);
    std::cout << "[YOLOV8GRAY][INFO] SetPadding2 ret=" << ret << std::endl;

    ret = SetNormalize(pipe_offline, model_id);
    std::cout << "[YOLOV8GRAY][INFO] SetNormalize ret=" << ret << std::endl;
}

void YOLOV8GRAY::Preprocess(ssne_tensor_t* img_in, ssne_tensor_t* input_tensor)
{
    int ret = RunAiPreprocessPipe(pipe_offline, *img_in, *input_tensor);
    if (ret != 0) {
        std::cout << "[YOLOV8GRAY][ERROR] RunAiPreprocessPipe failed, ret=" << ret << std::endl;
        return;
    }

    static bool dumped = false;
    if (getenv_flag("A1_DUMP_PREPROCESS_ONCE", false) && !dumped) {
        const std::string dump_path = getenv_string("A1_PREPROCESS_DUMP_PATH", "/tmp/yolov8_input.bin");
        int save_ret = save_tensor_buffer(*input_tensor, dump_path.c_str());
        std::cout << "[YOLOV8GRAY][DEBUG] preprocess dump ret=" << save_ret
                  << ", path=" << dump_path << std::endl;
        dumped = true;
    }
}

void YOLOV8GRAY::MapBoxToOriginalImage(std::array<float, 4>& box)
{
    float x1 = (box[0] - static_cast<float>(lb_info_.pad_x)) / lb_info_.scale;
    float y1 = (box[1] - static_cast<float>(lb_info_.pad_y)) / lb_info_.scale;
    float x2 = (box[2] - static_cast<float>(lb_info_.pad_x)) / lb_info_.scale;
    float y2 = (box[3] - static_cast<float>(lb_info_.pad_y)) / lb_info_.scale;

    x1 = clampf(x1, 0.0f, static_cast<float>(lb_info_.src_w - 1));
    y1 = clampf(y1, 0.0f, static_cast<float>(lb_info_.src_h - 1));
    x2 = clampf(x2, 0.0f, static_cast<float>(lb_info_.src_w - 1));
    y2 = clampf(y2, 0.0f, static_cast<float>(lb_info_.src_h - 1));

    box[0] = x1;
    box[1] = y1;
    box[2] = x2;
    box[3] = y2;
}

void YOLOV8GRAY::Postprocess(DetectionResult* result, float conf_threshold)
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

    if (invalid_output || cls_branches.size() != 3 || reg_branches.size() != 3 ||
        !pair_head_branches(&cls_branches, &reg_branches)) {
        std::cout << "[YOLOV8GRAY][ERROR] output branch grouping failed: cls="
                  << cls_branches.size() << ", reg=" << reg_branches.size()
                  << ". Check dtype/shape/order of the converted .m1model." << std::endl;
        result->Clear();
        return;
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
                  << (kReadModelOutputAsHwc ? "HWC" : "CHW")
                  << " + sigmoid + DFL + anchor decode + reverse letterbox" << std::endl;
        once = true;
    }

    int total_points = 0;
    for (size_t s = 0; s < cls_branches.size(); ++s) {
        total_points += cls_branches[s].w * cls_branches[s].h;
    }

    std::vector<float> cls_scores(NUM_CLASSES * total_points, 0.0f);
    std::vector<float> reg_raw(REG_CHANNELS * total_points, 0.0f);
    std::vector<float> anchor_x, anchor_y, stride_vec;
    build_anchor_points(cls_branches, &anchor_x, &anchor_y, &stride_vec);

    int offset = 0;
    for (size_t s = 0; s < cls_branches.size(); ++s) {
        const BranchView& cb = cls_branches[s];
        const BranchView& rb = reg_branches[s];
        const int HW = cb.w * cb.h;

        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int y = 0; y < cb.h; ++y) {
                for (int x = 0; x < cb.w; ++x) {
                    const int local = y * cb.w + x;
                    const int global = offset + local;
                    cls_scores[c * total_points + global] =
                        fast_sigmoid(read_branch_value(cb, c, y, x));
                }
            }
        }

        for (int c = 0; c < REG_CHANNELS; ++c) {
            for (int y = 0; y < rb.h; ++y) {
                for (int x = 0; x < rb.w; ++x) {
                    const int local = y * rb.w + x;
                    const int global = offset + local;
                    reg_raw[c * total_points + global] =
                        read_branch_value(rb, c, y, x);
                }
            }
        }

        offset += HW;
    }

    std::vector<float> dist(4 * total_points, 0.0f);

    for (int side = 0; side < 4; ++side) {
        for (int j = 0; j < total_points; ++j) {
            float max_logit = -FLT_MAX;
            for (int b = 0; b < REG_MAX; ++b) {
                const float v = reg_raw[(side * REG_MAX + b) * total_points + j];
                if (v > max_logit) {
                    max_logit = v;
                }
            }

            float sum_exp = 0.0f;
            float exp_value = 0.0f;
            for (int b = 0; b < REG_MAX; ++b) {
                const float v = reg_raw[(side * REG_MAX + b) * total_points + j];
                const float e = std::exp(v - max_logit);
                sum_exp += e;
                exp_value += e * static_cast<float>(b);
            }

            if (sum_exp > 1e-6f) {
                dist[side * total_points + j] = exp_value / sum_exp;
            }
        }
    }

    result->Clear();
    const DistanceConfig distance_cfg;

    for (int j = 0; j < total_points; ++j) {
        ClassCandidate best_by_semantic[obstacle::semantic::NUM_SEMANTIC_CLASSES];

        for (int raw_cls = 0; raw_cls < NUM_CLASSES; ++raw_cls) {
            if (!is_supported_raw_class(raw_cls)) {
                continue;
            }

            const float s = cls_scores[raw_cls * total_points + j];
            const int sem = obstacle::semantic::SemanticClassFromRaw(raw_cls);
            if (sem >= 0 && sem < obstacle::semantic::NUM_SEMANTIC_CLASSES &&
                s > best_by_semantic[sem].score) {
                best_by_semantic[sem].raw_cls = raw_cls;
                best_by_semantic[sem].score = s;
            }
        }

        std::vector<ClassCandidate> candidates;
        candidates.reserve(obstacle::semantic::NUM_SEMANTIC_CLASSES);
        for (int sem = 0; sem < obstacle::semantic::NUM_SEMANTIC_CLASSES; ++sem) {
            if (best_by_semantic[sem].score >= candidate_threshold_for_class(best_by_semantic[sem].raw_cls)) {
                push_unique_candidate(&candidates,
                                      best_by_semantic[sem].raw_cls,
                                      best_by_semantic[sem].score);
            }
        }
        if (candidates.empty()) {
            continue;
        }

        const float l = dist[0 * total_points + j];
        const float t = dist[1 * total_points + j];
        const float r = dist[2 * total_points + j];
        const float b = dist[3 * total_points + j];

        const float ax = anchor_x[j];
        const float ay = anchor_y[j];
        const float stride = stride_vec[j];

        const float x1_fm = ax - l;
        const float y1_fm = ay - t;
        const float x2_fm = ax + r;
        const float y2_fm = ay + b;

        const float cx = (x1_fm + x2_fm) * 0.5f * stride;
        const float cy = (y1_fm + y2_fm) * 0.5f * stride;
        const float w = (x2_fm - x1_fm) * stride;
        const float h = (y2_fm - y1_fm) * stride;

        float x1 = cx - 0.5f * w;
        float y1 = cy - 0.5f * h;
        float x2 = cx + 0.5f * w;
        float y2 = cy + 0.5f * h;

        x1 = clampf(x1, 0.0f, static_cast<float>(det_shape[0] - 1));
        y1 = clampf(y1, 0.0f, static_cast<float>(det_shape[1] - 1));
        x2 = clampf(x2, 0.0f, static_cast<float>(det_shape[0] - 1));
        y2 = clampf(y2, 0.0f, static_cast<float>(det_shape[1] - 1));

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            const ClassCandidate& cand = candidates[ci];
            const int display_cls = obstacle::semantic::SemanticClassFromRaw(cand.raw_cls);
            const std::string raw_label = ClassIdToLabel(cand.raw_cls);
            const std::string display_label = obstacle::semantic::SemanticLabel(display_cls);

            DetectionItem item;
            item.box = {x1, y1, x2, y2};
            item.score = cand.score;
            item.class_id = display_cls;
            item.raw_class_id = cand.raw_cls;
            item.label = display_label;
            item.semantic_class = display_label;
            item.raw_label = raw_label;
            item.risk_weight = obstacle::semantic::RiskWeight(display_cls);

            MapBoxToOriginalImage(item.box);

            if (item.box[2] <= item.box[0] || item.box[3] <= item.box[1]) {
                continue;
            }

            const float area_ratio = box_area_ratio(item.box, img_shape[0], img_shape[1]);
            const float width_ratio = box_width_ratio(item.box, img_shape[0]);
            const float height_ratio = box_height_ratio(item.box, img_shape[1]);
            const float center_y = box_center_y_ratio(item.box, img_shape[1]);
            const int touch = count_touch_borders(item.box, img_shape[0], img_shape[1]);

            const float bw = item.box[2] - item.box[0];
            const float bh = item.box[3] - item.box[1];

            if (bw < 8.f || bh < 10.f) {
                continue;
            }

            if (area_ratio > 0.98f || (width_ratio > 0.98f && height_ratio > 0.98f)) {
                continue;
            }

            const float final_threshold = candidate_threshold_for_class(item.raw_class_id);
            if (item.score < final_threshold) {
                continue;
            }

            if (item.class_id == DISPLAY_PERSON) {
                if (touch >= 4) {
                    continue;
                }
            } else if (center_y < 0.05f && item.score < 0.24f) {
                continue;
            } else if (item.score < 0.22f &&
                       (area_ratio > 0.85f || (width_ratio > 0.98f && touch >= 3))) {
                continue;
            }

            item.sector = sector_from_box(item.box, img_shape[0]);
            const float ground_distance = estimate_ground_distance_m(item.box, img_shape[1], distance_cfg);
            const float size_distance = estimate_size_distance_m(item.box,
                                                                img_shape[0],
                                                                img_shape[1],
                                                                item.raw_class_id,
                                                                distance_cfg);
            item.distance_m = fuse_distance_m(item.box,
                                              img_shape[0],
                                              img_shape[1],
                                              item.raw_class_id,
                                              ground_distance,
                                              size_distance,
                                              &item.distance_source);
            item.distance_confidence = distance_confidence_for_source(item.box,
                                                                      img_shape[0],
                                                                      img_shape[1],
                                                                      item.distance_source,
                                                                      item.score);
            if (obstacle::semantic::IsSmallObjectSemantic(item.class_id) &&
                item.distance_confidence < 0.48f) {
                item.distance_m = -1.0f;
                item.distance_source = "existence";
                item.distance_confidence = 0.25f;
            }
            item.quality = quality_from_box(item.box,
                                            img_shape[0],
                                            img_shape[1],
                                            item.class_id,
                                            item.score);
            item.risk_level = risk_from_distance(item.distance_m);

            result->items.emplace_back(item);
        }
    }

    result->raw_candidate_count = static_cast<int>(result->items.size());
    if (result->items.empty()) {
        if (debug_post && (postprocess_frame_count == 1 || postprocess_frame_count % debug_interval == 0)) {
            std::cout << "[YOLOV8GRAY][DEBUG] frame=" << postprocess_frame_count
                      << " total_points=" << total_points
                      << " raw_candidates=0 post_nms=0"
                      << " conf_threshold=" << conf_threshold << std::endl;
        }
        return;
    }

    utils::MultiTargetNMS(result, nms_threshold, top_k);
    result->post_nms_count = static_cast<int>(result->items.size());
    result->coarse_drop_count = 0;
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
    }
}

void YOLOV8GRAY::Predict(ssne_tensor_t* img_in,
                         DetectionResult* result,
                         float conf_threshold)
{
    if (result == nullptr) {
        std::cout << "[YOLOV8GRAY][ERROR] result is nullptr" << std::endl;
        return;
    }

    Preprocess(img_in, &inputs[0]);

    if (ssne_inference(model_id, 1, inputs)) {
        std::cout << "[YOLOV8GRAY][ERROR] ssne_inference failed" << std::endl;
        result->Clear();
        return;
    }

    ssne_getoutput(model_id, 6, outputs);

    Postprocess(result, conf_threshold);
}

void YOLOV8GRAY::Release()
{
    release_tensor(inputs[0]);

    for (int i = 0; i < 6; ++i) {
        release_tensor(outputs[i]);
    }

    ReleaseAIPreprocessPipe(pipe_offline);
}
