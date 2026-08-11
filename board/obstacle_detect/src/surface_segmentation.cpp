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

const int kGrid = SURFACE_GRID_SIZE;
const int kGridCells = SURFACE_GRID_CELLS;
const int kClasses = SURFACE_CLASS_COUNT;
const int kDepthBins = DEPTH_BIN_COUNT;
const float kDepthMinM = 0.30f;
const float kDepthMaxM = 8.0f;
const float kSegOutputScale = 0.0690593868494f;
const float kDepthOutputScale = 0.367095053196f;
const float kUnknownMaxRatio = 0.30f;
const float kDepthGroupMinProbability = 0.45f;
const float kDepthAmbiguityMargin = 0.20f;

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

size_t tensor_elements(const ssne_tensor_t& tensor)
{
    size_t total = static_cast<size_t>(get_total_size(tensor));
    if (get_data_type(tensor) == SSNE_FLOAT32 && total % sizeof(float) == 0) {
        const size_t as_elements = total / sizeof(float);
        if (as_elements == static_cast<size_t>(kGridCells * kClasses) ||
            as_elements == static_cast<size_t>(kGridCells * kDepthBins)) {
            return as_elements;
        }
    }
    return total;
}

float depth_center(int bin)
{
    const float log_min = std::log(kDepthMinM);
    const float step = (std::log(kDepthMaxM) - log_min) / static_cast<float>(kDepthBins);
    return std::exp(log_min + (static_cast<float>(bin) + 0.5f) * step);
}

int depth_group_for_bin(int bin)
{
    const float center = depth_center(bin);
    if (center < semantic::NearDistanceM()) return 0;
    if (center < semantic::WarningDistanceM()) return 1;
    return 2;
}

std::string depth_group_name(int group)
{
    if (group == 0) return "near";
    if (group == 1) return "mid";
    if (group == 2) return "far";
    return "unknown";
}

int depth_severity(const std::string& level)
{
    if (level == "near") return 3;
    if (level == "mid") return 2;
    if (level == "far") return 1;
    return 0;
}

const char* dtype_name(uint8_t dtype)
{
    if (dtype == SSNE_FLOAT32) return "float32";
    if (dtype == SSNE_INT8) return "int8";
    if (dtype == SSNE_UINT8) return "uint8";
    return "unknown";
}

}  // namespace

SurfaceSegmenter::SurfaceSegmenter()
    : model_id_(0),
      available_(false),
      input_created_(false),
      output_created_{false, false},
      input_{},
      outputs_{},
      seg_output_index_(-1),
      depth_output_index_(-1),
      preprocess_pipe_(NULL),
      image_shape_{720, 1280},
      input_shape_{256, 256},
      roi_{0, 560, 720, 1280},
      stable_depth_level_("unknown"),
      output_contract_logged_(false)
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
        std::cout << "[SURFACE_DEPTH][WARN] model missing; detector-only fallback path="
                  << model_path << std::endl;
        return false;
    }
    char* path = const_cast<char*>(model_path.c_str());
    model_id_ = ssne_loadmodel(path, SSNE_DYNAMIC_ALLOC);
    int dtype = -1;
    ssne_get_model_input_dtype(model_id_, &dtype);
    if (dtype < 0) {
        std::cout << "[SURFACE_DEPTH][ERROR] cannot read input dtype=" << dtype << std::endl;
        return false;
    }
    input_ = create_tensor(static_cast<uint32_t>(input_shape_[0]),
                           static_cast<uint32_t>(input_shape_[1]),
                           SSNE_Y_8, SSNE_BUF_AI);
    input_created_ = get_data(input_) != NULL;
    preprocess_pipe_ = GetAIPreprocessPipe();
    if (!input_created_ || preprocess_pipe_ == NULL) return false;
    Clear(preprocess_pipe_);
    int ret = SetCrop(preprocess_pipe_, static_cast<uint16_t>(roi_[0]),
                      static_cast<uint16_t>(roi_[1]), static_cast<uint16_t>(roi_[2]),
                      static_cast<uint16_t>(roi_[3]));
    if (ret == 0) ret = SetNormalize(preprocess_pipe_, model_id_);
    if (ret != 0) {
        std::cout << "[SURFACE_DEPTH][ERROR] preprocess setup ret=" << ret << std::endl;
        return false;
    }
    available_ = true;
    std::cout << "[SURFACE_DEPTH][INFO] model=" << model_path << " id=" << model_id_
              << " input=1x1x256x256 outputs=seg(4x64x64)+depth(16x64x64)"
              << " roi=" << roi_[0] << "," << roi_[1] << "," << roi_[2] << "," << roi_[3]
              << " alloc=dynamic dtype=" << dtype << std::endl;
    return true;
}

bool SurfaceSegmenter::Preprocess(ssne_tensor_t* image)
{
    return image != NULL && preprocess_pipe_ != NULL &&
           RunAiPreprocessPipe(preprocess_pipe_, *image, input_) == 0;
}

bool SurfaceSegmenter::BindOutputs(bool* hwc_layout)
{
    if (hwc_layout == NULL) return false;
    seg_output_index_ = depth_output_index_ = -1;
    for (int index = 0; index < 2; ++index) {
        if (get_data(outputs_[index]) == NULL || get_width(outputs_[index]) != kGrid ||
            get_height(outputs_[index]) != kGrid) {
            continue;
        }
        const size_t elements = tensor_elements(outputs_[index]);
        if (elements == static_cast<size_t>(kGridCells * kClasses)) seg_output_index_ = index;
        if (elements == static_cast<size_t>(kGridCells * kDepthBins)) depth_output_index_ = index;
    }
    *hwc_layout = env_hwc();
    if (seg_output_index_ < 0 || depth_output_index_ < 0 ||
        seg_output_index_ == depth_output_index_) {
        std::cout << "[SURFACE_DEPTH][ERROR] cannot bind two outputs totals="
                  << tensor_elements(outputs_[0]) << "/" << tensor_elements(outputs_[1])
                  << std::endl;
        return false;
    }
    return true;
}

bool SurfaceSegmenter::ReadOutputLogits(const ssne_tensor_t& tensor,
                                        int channels,
                                        float quant_scale,
                                        std::vector<float>* logits) const
{
    if (logits == NULL || get_data(tensor) == NULL) return false;
    const size_t expected = static_cast<size_t>(kGridCells * channels);
    if (tensor_elements(tensor) != expected) return false;
    logits->assign(expected, 0.0f);
    const uint8_t dtype = get_data_type(tensor);
    if (dtype == SSNE_FLOAT32) {
        const float* data = reinterpret_cast<const float*>(get_data(tensor));
        std::copy(data, data + expected, logits->begin());
    } else if (dtype == SSNE_INT8) {
        const int8_t* data = reinterpret_cast<const int8_t*>(get_data(tensor));
        if (!(quant_scale > 0.0f) || !std::isfinite(quant_scale)) return false;
        for (size_t i = 0; i < expected; ++i) {
            (*logits)[i] = static_cast<float>(data[i]) * quant_scale;
        }
    } else {
        // The formal E3 model uses symmetric INT8 outputs.  UINT8 would need a
        // zero point that the public SSNE tensor API does not expose, so fail
        // closed instead of interpreting bytes as logits.
        return false;
    }
    return true;
}

void SurfaceSegmenter::MajorityFilter(
    const std::array<uint8_t, SURFACE_GRID_CELLS>& input,
    std::array<uint8_t, SURFACE_GRID_CELLS>* output) const
{
    if (output == NULL) return;
    for (int y = 0; y < kGrid; ++y) {
        for (int x = 0; x < kGrid; ++x) {
            std::array<int, SURFACE_CLASS_COUNT> counts{};
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = std::max(0, std::min(kGrid - 1, x + dx));
                    const int yy = std::max(0, std::min(kGrid - 1, y + dy));
                    const int cls = input[yy * kGrid + xx];
                    if (cls >= 0 && cls < kClasses) ++counts[cls];
                }
            }
            int best = input[y * kGrid + x] < kClasses
                ? static_cast<int>(input[y * kGrid + x])
                : static_cast<int>(UNKNOWN_OTHER);
            for (int cls = 0; cls < kClasses; ++cls) if (counts[cls] > counts[best]) best = cls;
            (*output)[y * kGrid + x] = static_cast<uint8_t>(best);
        }
    }
}

bool SurfaceSegmenter::CellInCorridor(int x, int y, int corridor_index)
{
    if (y < kGrid / 4) return false;
    const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(kGrid);
    if (corridor_index == 0) return nx < 0.40f;
    if (corridor_index == 1) return nx >= 0.40f && nx <= 0.60f;
    return nx > 0.60f;
}

SurfaceSegmenter::CorridorStats SurfaceSegmenter::MeasureCorridor(
    const std::array<uint8_t, SURFACE_GRID_CELLS>& labels, int corridor_index) const
{
    CorridorStats stats;
    for (int y = 0; y < kGrid; ++y) for (int x = 0; x < kGrid; ++x) {
        if (!CellInCorridor(x, y, corridor_index)) continue;
        const int cls = labels[y * kGrid + x];
        if (cls < 0 || cls >= kClasses) continue;
        ++stats.counts[cls];
        ++stats.total;
        if (cls == STEP_OR_DROP && y > stats.lowest_hazard_y) stats.lowest_hazard_y = y;
    }
    for (int target = BLOCKED_SURFACE; target <= STEP_OR_DROP; ++target) {
        std::array<uint8_t, SURFACE_GRID_CELLS> seen{};
        for (int y = 0; y < kGrid; ++y) for (int x = 0; x < kGrid; ++x) {
            const int start = y * kGrid + x;
            if (seen[start] || labels[start] != target || !CellInCorridor(x, y, corridor_index)) continue;
            int component = 0;
            std::queue<int> pending;
            pending.push(start);
            seen[start] = 1;
            while (!pending.empty()) {
                const int at = pending.front(); pending.pop(); ++component;
                const int cx = at % kGrid, cy = at / kGrid;
                const int nx[4] = {cx - 1, cx + 1, cx, cx};
                const int ny[4] = {cy, cy, cy - 1, cy + 1};
                for (int i = 0; i < 4; ++i) {
                    if (nx[i] < 0 || nx[i] >= kGrid || ny[i] < 0 || ny[i] >= kGrid) continue;
                    const int next = ny[i] * kGrid + nx[i];
                    if (!seen[next] && labels[next] == target && CellInCorridor(nx[i], ny[i], corridor_index)) {
                        seen[next] = 1; pending.push(next);
                    }
                }
            }
            stats.largest_components[target] = std::max(stats.largest_components[target], component);
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
    corridor.unknown_ratio = stats.counts[UNKNOWN_OTHER] * inv;
    corridor.persistent_hazard = corridor.step_ratio >= 0.02f &&
                                 stats.largest_components[STEP_OR_DROP] >= 12;
    corridor.safe_candidate = corridor.ground_ratio >= 0.55f &&
        corridor.blocked_ratio < 0.35f && corridor.step_ratio < 0.02f &&
        corridor.unknown_ratio < kUnknownMaxRatio;
    return corridor;
}

void SurfaceSegmenter::UpdateTemporalState(const std::array<SurfaceCorridor, 3>& current,
                                            std::array<SurfaceCorridor, 3>* stable)
{
    history_.push_back(current);
    while (history_.size() > 3) history_.pop_front();
    *stable = current;
    for (int corridor = 0; corridor < 3; ++corridor) {
        int votes = 0;
        for (const auto& item : history_) if (item[corridor].persistent_hazard) ++votes;
        if (!hazard_latched_[corridor] && history_.size() >= 2 && votes >= 2) {
            hazard_latched_[corridor] = true;
            hazard_clear_count_[corridor] = 0;
        } else if (hazard_latched_[corridor]) {
            if (current[corridor].persistent_hazard) hazard_clear_count_[corridor] = 0;
            else if (++hazard_clear_count_[corridor] >= 4) {
                hazard_latched_[corridor] = false;
                hazard_clear_count_[corridor] = 0;
            }
        }
        (*stable)[corridor].persistent_hazard = hazard_latched_[corridor];
        if (hazard_latched_[corridor]) (*stable)[corridor].safe_candidate = false;
    }
}

float SurfaceSegmenter::Median(std::vector<float>* values)
{
    if (values == NULL || values->empty()) return -1.0f;
    const size_t mid = values->size() / 2;
    std::nth_element(values->begin(), values->begin() + mid, values->end());
    return (*values)[mid];
}

std::string SurfaceSegmenter::DepthLevel(float depth_m, float confidence)
{
    if (depth_m <= 0.0f || confidence < 0.12f) return "unknown";
    if (depth_m < semantic::NearDistanceM()) return "near";
    if (depth_m < semantic::WarningDistanceM()) return "mid";
    return "far";
}

std::string SurfaceSegmenter::StabilizeDepthLevel(const std::string& candidate,
                                                   float confidence,
                                                   float margin)
{
    if (candidate == "unknown") {
        center_depth_level_history_.clear();
        center_depth_history_.clear();
        stable_depth_level_ = "unknown";
        return stable_depth_level_;
    }

    center_depth_level_history_.push_back(candidate);
    while (center_depth_level_history_.size() > 3U) {
        center_depth_level_history_.pop_front();
    }

    if (stable_depth_level_ == "unknown") {
        const bool strong_single_frame = candidate == "near" ||
            (confidence >= 0.65f && margin >= 0.35f);
        const bool repeated = center_depth_level_history_.size() >= 2U &&
            center_depth_level_history_[center_depth_level_history_.size() - 1U] == candidate &&
            center_depth_level_history_[center_depth_level_history_.size() - 2U] == candidate;
        if (strong_single_frame || repeated) stable_depth_level_ = candidate;
        return stable_depth_level_;
    }

    // Moving toward a more dangerous level is immediate.  Moving farther away
    // needs two matching SurfaceDepth frames so the demo does not flicker.
    if (depth_severity(candidate) > depth_severity(stable_depth_level_)) {
        stable_depth_level_ = candidate;
        return stable_depth_level_;
    }
    if (candidate == stable_depth_level_) return stable_depth_level_;
    if (center_depth_level_history_.size() >= 2U &&
        center_depth_level_history_[center_depth_level_history_.size() - 1U] == candidate &&
        center_depth_level_history_[center_depth_level_history_.size() - 2U] == candidate) {
        stable_depth_level_ = candidate;
    }
    return stable_depth_level_;
}

void SurfaceSegmenter::DecodeDepth(
    const float* logits, bool hwc_layout,
    const std::array<uint8_t, SURFACE_GRID_CELLS>& labels, SurfaceResult* result)
{
    std::vector<float> center_depths;
    std::array<std::vector<float>, 3> center_group_probabilities;
    for (int y = 0; y < kGrid; ++y) for (int x = 0; x < kGrid; ++x) {
        float maximum = -1e30f;
        for (int bin = 0; bin < kDepthBins; ++bin) {
            const size_t index = hwc_layout
                ? static_cast<size_t>((y * kGrid + x) * kDepthBins + bin)
                : static_cast<size_t>(bin * kGridCells + y * kGrid + x);
            maximum = std::max(maximum, logits[index]);
        }
        float denom = 0.0f;
        float weighted = 0.0f;
        std::array<float, 3> grouped = {0.0f, 0.0f, 0.0f};
        for (int bin = 0; bin < kDepthBins; ++bin) {
            const size_t index = hwc_layout
                ? static_cast<size_t>((y * kGrid + x) * kDepthBins + bin)
                : static_cast<size_t>(bin * kGridCells + y * kGrid + x);
            const float probability = std::exp(logits[index] - maximum);
            denom += probability;
            weighted += probability * depth_center(bin);
            grouped[depth_group_for_bin(bin)] += probability;
        }
        const int cell = y * kGrid + x;
        result->depth_m[cell] = denom > 0.0f ? weighted / denom : -1.0f;
        if (denom > 0.0f) {
            for (int group = 0; group < 3; ++group) grouped[group] /= denom;
        }
        float top_group = -1.0f;
        float second_group = -1.0f;
        for (int group = 0; group < 3; ++group) {
            if (grouped[group] > top_group) {
                second_group = top_group;
                top_group = grouped[group];
            } else if (grouped[group] > second_group) {
                second_group = grouped[group];
            }
        }
        result->depth_cell_confidence[cell] =
            std::max(0.0f, top_group - std::max(0.0f, second_group));
        // UNKNOWN_OTHER is deliberately excluded from the corridor depth vote.
        // A visually unresolved region must not produce a confident FAR result.
        if (labels[cell] != UNKNOWN_OTHER && CellInCorridor(x, y, 1) &&
            y >= kGrid / 3 && top_group >= 0.34f) {
            center_depths.push_back(result->depth_m[cell]);
            for (int group = 0; group < 3; ++group) {
                center_group_probabilities[group].push_back(grouped[group]);
            }
        }
    }
    result->center_depth_m = Median(&center_depths);
    float group_sum = 0.0f;
    for (int group = 0; group < 3; ++group) {
        result->depth_group_probabilities[group] = Median(&center_group_probabilities[group]);
        group_sum += std::max(0.0f, result->depth_group_probabilities[group]);
    }
    if (group_sum > 0.0f) {
        for (int group = 0; group < 3; ++group) {
            result->depth_group_probabilities[group] /= group_sum;
        }
    }

    int best_group = -1;
    float best_probability = -1.0f;
    float second_probability = -1.0f;
    for (int group = 0; group < 3; ++group) {
        const float probability = result->depth_group_probabilities[group];
        if (probability > best_probability) {
            second_probability = best_probability;
            best_probability = probability;
            best_group = group;
        } else if (probability > second_probability) {
            second_probability = probability;
        }
    }
    result->depth_confidence = std::max(0.0f, best_probability);
    result->depth_margin = std::max(0.0f, best_probability - std::max(0.0f, second_probability));
    std::string candidate = "unknown";
    if (center_depths.size() >= 12U &&
        result->depth_confidence >= kDepthGroupMinProbability &&
        result->depth_margin >= kDepthAmbiguityMargin) {
        candidate = depth_group_name(best_group);
    }
    result->depth_ambiguous = candidate == "unknown";
    result->depth_level = StabilizeDepthLevel(
        candidate, result->depth_confidence, result->depth_margin);
    if (result->depth_level == "unknown") {
        result->depth_source = result->depth_ambiguous
            ? "learned_ambiguous" : "learned_pending";
    } else {
        result->depth_source = "learned_grouped";
    }
    if (result->center_depth_m > 0.0f && !result->depth_ambiguous) {
        center_depth_history_.push_back(result->center_depth_m);
        while (center_depth_history_.size() > 4) center_depth_history_.pop_front();
    }
    result->approaching = center_depth_history_.size() >= 3 &&
        center_depth_history_.front() - center_depth_history_.back() >
            std::max(0.20f, center_depth_history_.front() * 0.15f);
}

bool SurfaceSegmenter::PostprocessLogits(
    const float* seg_logits, size_t seg_count,
    const float* depth_logits, size_t depth_count,
    bool hwc_layout, int64_t timestamp_ms, SurfaceResult* result)
{
    if (seg_logits == NULL || depth_logits == NULL || result == NULL ||
        seg_count != static_cast<size_t>(kGridCells * kClasses) ||
        depth_count != static_cast<size_t>(kGridCells * kDepthBins)) return false;
    std::array<uint8_t, SURFACE_GRID_CELLS> raw{};
    for (int y = 0; y < kGrid; ++y) for (int x = 0; x < kGrid; ++x) {
        int best = 0; float best_value = -1e30f;
        for (int cls = 0; cls < kClasses; ++cls) {
            const size_t index = hwc_layout
                ? static_cast<size_t>((y * kGrid + x) * kClasses + cls)
                : static_cast<size_t>(cls * kGridCells + y * kGrid + x);
            if (seg_logits[index] > best_value) { best_value = seg_logits[index]; best = cls; }
        }
        raw[y * kGrid + x] = static_cast<uint8_t>(best);
    }
    std::array<uint8_t, SURFACE_GRID_CELLS> filtered{};
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
    result->left = stable[0]; result->center = stable[1]; result->right = stable[2];
    result->labels = filtered;
    result->confidence = std::max(result->center.ground_ratio,
        std::max(result->center.blocked_ratio,
            std::max(result->center.step_ratio, result->center.unknown_ratio)));
    DecodeDepth(depth_logits, hwc_layout, filtered, result);
    int primary = -1;
    for (int candidate : {1, 0, 2}) {
        if (stable[candidate].persistent_hazard || stable[candidate].blocked_ratio >= 0.35f) {
            primary = candidate; break;
        }
    }
    if (primary >= 0) {
        result->primary_hazard = stable[primary].persistent_hazard
            ? "step_or_drop" : "blocked_surface";
        result->primary_sector = primary == 0 ? "left" : (primary == 1 ? "center" : "right");
        result->proximity = result->depth_level;
    } else if (!result->center.safe_candidate) {
        result->primary_hazard = result->center.unknown_ratio >= kUnknownMaxRatio
            ? "unknown_other" : "unknown";
        result->primary_sector = "center";
        result->proximity = result->depth_level;
    } else {
        result->primary_hazard = "none";
        result->primary_sector = "unknown";
        result->proximity = result->depth_level;
    }
    if ((result->primary_hazard == "unknown_other" || result->primary_hazard == "unknown") &&
        !result->center.persistent_hazard && result->center.blocked_ratio < 0.35f) {
        // A semantic UNKNOWN region must not be paired with a confident FAR
        // label in Aurora or speech.  The raw probabilities remain in the
        // diagnostic fields for later threshold tuning.
        result->depth_level = "unknown";
        result->depth_ambiguous = true;
        result->depth_source = "surface_unknown";
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
    if (ssne_getoutput(model_id_, 2, outputs_) != 0) return false;
    output_created_[0] = get_data(outputs_[0]) != NULL;
    output_created_[1] = get_data(outputs_[1]) != NULL;
    last_timing_.output_ms = elapsed(start);
    bool hwc = true;
    if (!BindOutputs(&hwc)) return false;
    if (!output_contract_logged_) {
        std::cout << "[SURFACE_DEPTH][CONTRACT] seg_index=" << seg_output_index_
                  << " seg_dtype=" << dtype_name(get_data_type(outputs_[seg_output_index_]))
                  << " seg_elements=" << tensor_elements(outputs_[seg_output_index_])
                  << " seg_scale=" << kSegOutputScale
                  << " depth_index=" << depth_output_index_
                  << " depth_dtype=" << dtype_name(get_data_type(outputs_[depth_output_index_]))
                  << " depth_elements=" << tensor_elements(outputs_[depth_output_index_])
                  << " depth_scale=" << kDepthOutputScale
                  << " layout=" << (hwc ? "HWC" : "CHW")
                  << std::endl;
        output_contract_logged_ = true;
    }
    std::vector<float> seg, depth;
    if (!ReadOutputLogits(outputs_[seg_output_index_], kClasses, kSegOutputScale, &seg) ||
        !ReadOutputLogits(outputs_[depth_output_index_], kDepthBins, kDepthOutputScale, &depth)) {
        std::cout << "[SURFACE_DEPTH][ERROR] unsupported output dtype or quantization contract"
                  << std::endl;
        return false;
    }
    start = std::chrono::steady_clock::now();
    const bool ok = PostprocessLogits(seg.data(), seg.size(), depth.data(), depth.size(),
                                      hwc, monotonic_ms(), result);
    last_timing_.postprocess_ms = elapsed(start);
    return ok;
}

void SurfaceSegmenter::Release()
{
    if (input_created_) { release_tensor(input_); input_created_ = false; }
    for (int i = 0; i < 2; ++i) if (output_created_[i]) {
        release_tensor(outputs_[i]); output_created_[i] = false;
    }
    if (preprocess_pipe_ != NULL) {
        ReleaseAIPreprocessPipe(preprocess_pipe_); preprocess_pipe_ = NULL;
    }
    available_ = false;
    output_contract_logged_ = false;
    history_.clear();
    center_depth_history_.clear();
    center_depth_level_history_.clear();
    stable_depth_level_ = "unknown";
}

}  // namespace obstacle
