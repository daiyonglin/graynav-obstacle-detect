#include "../include/surface_segmentation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <sys/stat.h>

namespace obstacle {
namespace {

const int kGrid = 32;
const int kGridCells = kGrid * kGrid;
const int kClasses = SURFACE_CLASS_COUNT;

int64_t monotonic_ms()
{
    return static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool env_hwc()
{
    const char* value = std::getenv("A1_SEG_OUTPUT_LAYOUT");
    if (value == NULL) return true;
    return std::strcmp(value, "CHW") != 0 && std::strcmp(value, "chw") != 0;
}

}  // namespace

SurfaceSegmenter::SurfaceSegmenter()
    : model_id_(0),
      available_(false),
      input_created_(false),
      output_created_(false),
      input_{},
      output_{},
      preprocess_pipe_(NULL),
      image_shape_{720, 1280},
      input_shape_{256, 256},
      roi_{0, 560, 720, 1280}
{
    std::memset(hazard_latched_, 0, sizeof(hazard_latched_));
    std::memset(hazard_clear_count_, 0, sizeof(hazard_clear_count_));
}

bool SurfaceSegmenter::Initialize(const std::string& model_path,
                                  const std::array<int, 2>& image_shape,
                                  const std::array<int, 2>& input_shape)
{
    image_shape_ = image_shape;
    input_shape_ = input_shape;
    const int roi_size = std::min(image_shape_[0], image_shape_[1]);
    const int lower_y = std::max(0, image_shape_[1] - roi_size);
    roi_ = {0, lower_y, roi_size, lower_y + roi_size};

    struct stat file_info;
    if (stat(model_path.c_str(), &file_info) != 0 || file_info.st_size <= 0) {
        std::cout << "[SURFACE][WARN] model missing; detector-only fallback path="
                  << model_path << std::endl;
        return false;
    }
    char* path = const_cast<char*>(model_path.c_str());
    model_id_ = ssne_loadmodel(path, SSNE_DYNAMIC_ALLOC);
    int dtype = -1;
    ssne_get_model_input_dtype(model_id_, &dtype);
    if (dtype < 0) {
        std::cout << "[SURFACE][ERROR] cannot read model input dtype="
                  << dtype << std::endl;
        return false;
    }
    input_ = create_tensor(static_cast<uint32_t>(input_shape_[0]),
                           static_cast<uint32_t>(input_shape_[1]),
                           SSNE_Y_8,
                           SSNE_BUF_AI);
    input_created_ = get_data(input_) != NULL;
    preprocess_pipe_ = GetAIPreprocessPipe();
    if (!input_created_ || preprocess_pipe_ == NULL) {
        std::cout << "[SURFACE][ERROR] failed to allocate input/preprocess resources" << std::endl;
        return false;
    }
    Clear(preprocess_pipe_);
    int ret = SetCrop(preprocess_pipe_,
                      static_cast<uint16_t>(roi_[0]),
                      static_cast<uint16_t>(roi_[1]),
                      static_cast<uint16_t>(roi_[2]),
                      static_cast<uint16_t>(roi_[3]));
    if (ret == 0) ret = SetNormalize(preprocess_pipe_, model_id_);
    if (ret != 0) {
        std::cout << "[SURFACE][ERROR] preprocess configuration failed ret=" << ret << std::endl;
        return false;
    }
    available_ = true;
    std::cout << "[SURFACE][INFO] model=" << model_path
              << " id=" << model_id_
              << " input=1x1x" << input_shape_[1] << "x" << input_shape_[0]
              << " roi=" << roi_[0] << "," << roi_[1] << "," << roi_[2] << "," << roi_[3]
              << " alloc=dynamic" << std::endl;
    std::cout << "[SURFACE][INFO] contract input_count=1 input_dtype=" << dtype
              << " output_count=1 expected_output=1x4x32x32" << std::endl;
    return true;
}

bool SurfaceSegmenter::Preprocess(ssne_tensor_t* image)
{
    return image != NULL && preprocess_pipe_ != NULL &&
           RunAiPreprocessPipe(preprocess_pipe_, *image, input_) == 0;
}

bool SurfaceSegmenter::ReadOutputLogits(std::vector<float>* logits, bool* hwc_layout) const
{
    if (logits == NULL || hwc_layout == NULL || get_data(output_) == NULL) return false;
    const uint32_t width = get_width(output_);
    const uint32_t height = get_height(output_);
    const uint32_t raw_total = get_total_size(output_);
    const uint8_t dtype = get_data_type(output_);
    if (width != kGrid || height != kGrid) {
        std::cout << "[SURFACE][ERROR] output grid mismatch width=" << width
                  << " height=" << height << std::endl;
        return false;
    }
    const size_t expected = static_cast<size_t>(kGridCells * kClasses);
    logits->assign(expected, 0.0f);
    if (dtype == SSNE_FLOAT32 && (raw_total == expected || raw_total == expected * sizeof(float))) {
        const float* data = reinterpret_cast<const float*>(get_data(output_));
        std::copy(data, data + expected, logits->begin());
    } else if ((dtype == SSNE_INT8 || dtype == SSNE_UINT8) && raw_total == expected) {
        if (dtype == SSNE_INT8) {
            const int8_t* data = reinterpret_cast<const int8_t*>(get_data(output_));
            for (size_t i = 0; i < expected; ++i) (*logits)[i] = static_cast<float>(data[i]);
        } else {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(get_data(output_));
            for (size_t i = 0; i < expected; ++i) (*logits)[i] = static_cast<float>(data[i]);
        }
    } else {
        std::cout << "[SURFACE][ERROR] unsupported output dtype=" << static_cast<int>(dtype)
                  << " total=" << raw_total << " expected=" << expected << std::endl;
        return false;
    }
    *hwc_layout = env_hwc();
    return true;
}

void SurfaceSegmenter::MajorityFilter(const std::array<uint8_t, 1024>& input,
                                      std::array<uint8_t, 1024>* output) const
{
    if (output == NULL) return;
    for (int y = 0; y < kGrid; ++y) {
        for (int x = 0; x < kGrid; ++x) {
            int counts[kClasses] = {0, 0, 0, 0};
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = std::max(0, std::min(kGrid - 1, x + dx));
                    const int yy = std::max(0, std::min(kGrid - 1, y + dy));
                    ++counts[input[yy * kGrid + xx]];
                }
            }
            int best = input[y * kGrid + x];
            for (int cls = 0; cls < kClasses; ++cls) {
                if (counts[cls] > counts[best]) best = cls;
            }
            (*output)[y * kGrid + x] = static_cast<uint8_t>(best);
        }
    }
}

bool SurfaceSegmenter::CellInCorridor(int x, int y, int corridor_index)
{
    // 忽略 ROI 顶部四分之一；其余网格按与现有 left/center/right 语义一致的
    // 归一化边界切分，避免依赖未标定的相机内参。
    if (y < 8) return false;
    const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(kGrid);
    if (corridor_index == 0) return nx < 0.40f;
    if (corridor_index == 1) return nx >= 0.40f && nx <= 0.60f;
    return nx > 0.60f;
}

SurfaceSegmenter::CorridorStats SurfaceSegmenter::MeasureCorridor(
    const std::array<uint8_t, 1024>& labels,
    int corridor_index) const
{
    CorridorStats stats;
    for (int y = 0; y < kGrid; ++y) {
        for (int x = 0; x < kGrid; ++x) {
            if (!CellInCorridor(x, y, corridor_index)) continue;
            const int cls = labels[y * kGrid + x];
            ++stats.counts[cls];
            ++stats.total;
            if ((cls == STEP_OR_DROP || cls == POTHOLE) && y > stats.lowest_hazard_y) {
                stats.lowest_hazard_y = y;
            }
        }
    }

    for (int target = BLOCKED_SURFACE; target <= POTHOLE; ++target) {
        std::array<uint8_t, 1024> seen{};
        for (int y = 0; y < kGrid; ++y) {
            for (int x = 0; x < kGrid; ++x) {
                const int start = y * kGrid + x;
                if (seen[start] || labels[start] != target || !CellInCorridor(x, y, corridor_index)) continue;
                int component = 0;
                std::queue<int> pending;
                pending.push(start);
                seen[start] = 1;
                while (!pending.empty()) {
                    const int index = pending.front();
                    pending.pop();
                    ++component;
                    const int cx = index % kGrid;
                    const int cy = index / kGrid;
                    const int nx[4] = {cx - 1, cx + 1, cx, cx};
                    const int ny[4] = {cy, cy, cy - 1, cy + 1};
                    for (int n = 0; n < 4; ++n) {
                        if (nx[n] < 0 || nx[n] >= kGrid || ny[n] < 0 || ny[n] >= kGrid) continue;
                        const int next = ny[n] * kGrid + nx[n];
                        if (!seen[next] && labels[next] == target &&
                            CellInCorridor(nx[n], ny[n], corridor_index)) {
                            seen[next] = 1;
                            pending.push(next);
                        }
                    }
                }
                stats.largest_components[target] = std::max(
                    stats.largest_components[target], component);
            }
        }
    }
    return stats;
}

SurfaceCorridor SurfaceSegmenter::BuildCorridor(const CorridorStats& stats)
{
    SurfaceCorridor corridor;
    const float inv = 1.0f / static_cast<float>(std::max(1, stats.total));
    corridor.ground_ratio = stats.counts[GROUND_CANDIDATE] * inv;
    corridor.blocked_ratio = stats.counts[BLOCKED_SURFACE] * inv;
    corridor.step_ratio = stats.counts[STEP_OR_DROP] * inv;
    corridor.pothole_ratio = stats.counts[POTHOLE] * inv;
    const bool connected_drop =
        (corridor.step_ratio >= 0.03f && stats.largest_components[STEP_OR_DROP] >= 4) ||
        (corridor.pothole_ratio >= 0.03f && stats.largest_components[POTHOLE] >= 4);
    corridor.persistent_hazard = connected_drop;
    corridor.safe_candidate = corridor.ground_ratio >= 0.55f &&
        corridor.blocked_ratio < 0.35f &&
        corridor.step_ratio < 0.03f && corridor.pothole_ratio < 0.03f;
    return corridor;
}

void SurfaceSegmenter::UpdateTemporalState(
    const std::array<SurfaceCorridor, 3>& current,
    std::array<SurfaceCorridor, 3>* stable)
{
    history_.push_back(current);
    while (history_.size() > 3) history_.pop_front();
    *stable = current;
    for (int corridor = 0; corridor < 3; ++corridor) {
        for (int hazard = 0; hazard < 2; ++hazard) {
            int votes = 0;
            for (size_t i = 0; i < history_.size(); ++i) {
                const SurfaceCorridor& item = history_[i][corridor];
                const float ratio = hazard == 0 ? item.step_ratio : item.pothole_ratio;
                if (item.persistent_hazard && ratio >= 0.03f) ++votes;
            }
            const SurfaceCorridor& newest = history_.back()[corridor];
            const float newest_ratio = hazard == 0 ? newest.step_ratio : newest.pothole_ratio;
            const bool current_has_hazard = newest.persistent_hazard && newest_ratio >= 0.03f;
            if (!hazard_latched_[corridor][hazard] && history_.size() >= 2 && votes >= 2) {
                hazard_latched_[corridor][hazard] = true;
                hazard_clear_count_[corridor][hazard] = 0;
            } else if (hazard_latched_[corridor][hazard]) {
                if (current_has_hazard) {
                    hazard_clear_count_[corridor][hazard] = 0;
                } else {
                    ++hazard_clear_count_[corridor][hazard];
                    if (hazard_clear_count_[corridor][hazard] >= 4) {
                        hazard_latched_[corridor][hazard] = false;
                        hazard_clear_count_[corridor][hazard] = 0;
                    }
                }
            }
        }
        (*stable)[corridor].persistent_hazard =
            hazard_latched_[corridor][0] || hazard_latched_[corridor][1];
        if ((*stable)[corridor].persistent_hazard) {
            (*stable)[corridor].safe_candidate = false;
        }
    }
}

bool SurfaceSegmenter::PostprocessLogits(const float* logits,
                                         size_t element_count,
                                         bool hwc_layout,
                                         int64_t timestamp_ms,
                                         SurfaceResult* result)
{
    if (logits == NULL || result == NULL || element_count != kGridCells * kClasses) return false;
    std::array<uint8_t, 1024> raw{};
    for (int y = 0; y < kGrid; ++y) {
        for (int x = 0; x < kGrid; ++x) {
            int best = 0;
            float best_value = -1e30f;
            for (int cls = 0; cls < kClasses; ++cls) {
                const size_t index = hwc_layout
                    ? static_cast<size_t>((y * kGrid + x) * kClasses + cls)
                    : static_cast<size_t>(cls * kGridCells + y * kGrid + x);
                if (logits[index] > best_value) {
                    best_value = logits[index];
                    best = cls;
                }
            }
            raw[y * kGrid + x] = static_cast<uint8_t>(best);
        }
    }
    std::array<uint8_t, 1024> filtered{};
    MajorityFilter(raw, &filtered);
    std::array<CorridorStats, 3> stats = {
        MeasureCorridor(filtered, 0), MeasureCorridor(filtered, 1), MeasureCorridor(filtered, 2)};
    std::array<SurfaceCorridor, 3> current = {
        BuildCorridor(stats[0]), BuildCorridor(stats[1]), BuildCorridor(stats[2])};
    std::array<SurfaceCorridor, 3> stable;
    UpdateTemporalState(current, &stable);

    *result = SurfaceResult();
    result->valid = true;
    result->stale = false;
    result->timestamp_ms = timestamp_ms;
    result->left = stable[0];
    result->center = stable[1];
    result->right = stable[2];
    result->confidence = std::max(result->center.ground_ratio,
        std::max(result->center.blocked_ratio,
                 std::max(result->center.step_ratio, result->center.pothole_ratio)));

    int primary_corridor = -1;
    for (int candidate : {1, 0, 2}) {
        if (stable[candidate].persistent_hazard || stable[candidate].blocked_ratio >= 0.35f) {
            primary_corridor = candidate;
            break;
        }
    }
    if (primary_corridor >= 0) {
        const SurfaceCorridor& primary = stable[primary_corridor];
        if (primary.pothole_ratio >= 0.03f) result->primary_hazard = "pothole";
        else if (primary.step_ratio >= 0.03f) result->primary_hazard = "step_or_drop";
        else result->primary_hazard = "blocked_surface";
        result->primary_sector = primary_corridor == 0 ? "left" :
                                 (primary_corridor == 1 ? "center" : "right");
        result->proximity = stats[primary_corridor].lowest_hazard_y >= 24 ? "near" : "mid";
    } else if (!result->center.safe_candidate) {
        result->primary_hazard = "unknown";
        result->primary_sector = "center";
        result->proximity = "unknown";
    } else {
        result->primary_hazard = "none";
        result->primary_sector = "unknown";
        result->proximity = "unknown";
    }
    return true;
}

bool SurfaceSegmenter::Predict(ssne_tensor_t* image, SurfaceResult* result)
{
    if (!available_ || result == NULL) return false;
    const auto elapsed = [](const std::chrono::steady_clock::time_point& start) {
        return std::chrono::duration_cast<std::chrono::duration<float, std::milli> >(
            std::chrono::steady_clock::now() - start).count();
    };
    auto start = std::chrono::steady_clock::now();
    if (!Preprocess(image)) return false;
    last_timing_.preprocess_ms = elapsed(start);
    start = std::chrono::steady_clock::now();
    if (ssne_inference(model_id_, 1, &input_) != 0) return false;
    last_timing_.inference_ms = elapsed(start);
    start = std::chrono::steady_clock::now();
    if (ssne_getoutput(model_id_, 1, &output_) != 0) return false;
    output_created_ = get_data(output_) != NULL;
    last_timing_.output_ms = elapsed(start);
    std::vector<float> logits;
    bool hwc = true;
    if (!ReadOutputLogits(&logits, &hwc)) return false;
    start = std::chrono::steady_clock::now();
    const bool ok = PostprocessLogits(logits.data(), logits.size(), hwc, monotonic_ms(), result);
    last_timing_.postprocess_ms = elapsed(start);
    return ok;
}

void SurfaceSegmenter::Release()
{
    if (input_created_) {
        release_tensor(input_);
        input_created_ = false;
    }
    if (output_created_) {
        release_tensor(output_);
        output_created_ = false;
    }
    if (preprocess_pipe_ != NULL) {
        ReleaseAIPreprocessPipe(preprocess_pipe_);
        preprocess_pipe_ = NULL;
    }
    available_ = false;
}

}  // namespace obstacle
