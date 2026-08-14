#include "../include/surface_segmentation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <queue>

namespace obstacle {
namespace {

const int kGrid = SURFACE_GRID_SIZE;
const int kGridCells = SURFACE_GRID_CELLS;
const int kClasses = SURFACE_CLASS_COUNT;
const int kDepthBins = DEPTH_BIN_COUNT;
const float kDepthMinM = 0.30f;
const float kDepthMaxM = 8.0f;
const float kUnknownMaxRatio = 0.30f;
const float kDepthGroupMinProbability = 0.45f;
const float kDepthAmbiguityMargin = 0.20f;

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

}  // namespace

SurfaceSegmenter::SurfaceSegmenter()
    : stair_state_(STAIR_NONE),
      stair_suspect_clear_count_(0),
      stair_confirm_clear_count_(0),
      stable_depth_level_("unknown")
{
    std::memset(hazard_latched_, 0, sizeof(hazard_latched_));
    std::memset(hazard_clear_count_, 0, sizeof(hazard_clear_count_));
    std::memset(blocked_latched_, 0, sizeof(blocked_latched_));
    std::memset(blocked_clear_count_, 0, sizeof(blocked_clear_count_));
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
    corridor.step_largest_component = stats.largest_components[STEP_OR_DROP];
    corridor.blocked_largest_component = stats.largest_components[BLOCKED_SURFACE];
    // A full-corridor STEP mask is a known failure mode on dark or repetitive
    // indoor textures.  Segmentation alone may only nominate a bounded,
    // connected step region; a larger region needs edge/depth corroboration.
    corridor.persistent_hazard = corridor.step_ratio >= 0.04f &&
                                 corridor.step_ratio <= 0.35f &&
                                 stats.largest_components[STEP_OR_DROP] >= 12;
    corridor.blocked_persistent = corridor.blocked_ratio >= 0.40f &&
                                  stats.largest_components[BLOCKED_SURFACE] >= 12;
    corridor.safe_candidate = corridor.ground_ratio >= 0.60f &&
        corridor.blocked_ratio < 0.25f && corridor.step_ratio < 0.02f &&
        corridor.unknown_ratio < kUnknownMaxRatio;
    return corridor;
}

void SurfaceSegmenter::UpdateTemporalState(const std::array<SurfaceCorridor, 3>& current,
                                            std::array<SurfaceCorridor, 3>* stable)
{
    history_.push_back(current);
    while (history_.size() > 5) history_.pop_front();
    *stable = current;
    for (int corridor = 0; corridor < 3; ++corridor) {
        int step_votes = 0;
        int blocked_votes = 0;
        int safe_votes = 0;
        const size_t step_begin = history_.size() > 4U ? history_.size() - 4U : 0U;
        for (size_t i = 0; i < history_.size(); ++i) {
            if (i >= step_begin && history_[i][corridor].persistent_hazard) ++step_votes;
            if (i >= step_begin && history_[i][corridor].blocked_persistent) ++blocked_votes;
            if (history_[i][corridor].safe_candidate) ++safe_votes;
        }
        if (!hazard_latched_[corridor] && history_.size() >= 4U && step_votes >= 3) {
            hazard_latched_[corridor] = true;
            hazard_clear_count_[corridor] = 0;
        } else if (hazard_latched_[corridor]) {
            if (current[corridor].persistent_hazard) hazard_clear_count_[corridor] = 0;
            else if (++hazard_clear_count_[corridor] >= 4) {
                hazard_latched_[corridor] = false;
                hazard_clear_count_[corridor] = 0;
            }
        }
        if (!blocked_latched_[corridor] && history_.size() >= 4U && blocked_votes >= 3) {
            blocked_latched_[corridor] = true;
            blocked_clear_count_[corridor] = 0;
        } else if (blocked_latched_[corridor]) {
            if (current[corridor].blocked_persistent) blocked_clear_count_[corridor] = 0;
            else if (++blocked_clear_count_[corridor] >= 4) {
                blocked_latched_[corridor] = false;
                blocked_clear_count_[corridor] = 0;
            }
        }
        (*stable)[corridor].persistent_hazard = hazard_latched_[corridor];
        (*stable)[corridor].blocked_persistent = blocked_latched_[corridor];
        // CLEAR/PATH is deliberately slower than hazard onset.  Three of five
        // lower-ROI observations must agree and no hazard latch may remain.
        (*stable)[corridor].safe_candidate = history_.size() >= 5U && safe_votes >= 3 &&
            !hazard_latched_[corridor] && !blocked_latched_[corridor];
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
        if (stable[candidate].persistent_hazard || stable[candidate].blocked_persistent) {
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
        !result->center.persistent_hazard && !result->center.blocked_persistent) {
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

void SurfaceSegmenter::DecodeStairEdge(const float* scene_logits,
                                       bool hwc_layout,
                                       SurfaceResult* result)
{
    if (scene_logits == NULL || result == NULL) return;
    std::array<float, SURFACE_GRID_SIZE> row_scores{};
    std::array<float, SURFACE_GRID_SIZE> row_spans{};
    int evidence_cells = 0;
    int corridor_cells = 0;
    for (int y = kGrid / 5; y < kGrid; ++y) {
        float row_sum = 0.0f;
        int row_count = 0;
        int consecutive = 0;
        int longest = 0;
        for (int x = 0; x < kGrid; ++x) {
            if (!CellInCorridor(x, y, 1)) continue;
            const size_t index = hwc_layout
                ? static_cast<size_t>((y * kGrid + x) * UNIFIED_SCENE_CHANNELS + STAIR_EDGE_CHANNEL)
                : static_cast<size_t>(STAIR_EDGE_CHANNEL * kGridCells + y * kGrid + x);
            const float logit = scene_logits[index];
            const float probability = 1.0f / (1.0f + std::exp(-std::max(-20.0f, std::min(20.0f, logit))));
            row_sum += probability;
            ++row_count;
            ++corridor_cells;
            if (probability >= 0.50f) {
                ++evidence_cells;
                longest = std::max(longest, ++consecutive);
            } else {
                consecutive = 0;
            }
        }
        row_scores[y] = row_count > 0 ? row_sum / static_cast<float>(row_count) : 0.0f;
        row_spans[y] = row_count > 0
            ? static_cast<float>(longest) / static_cast<float>(row_count) : 0.0f;
    }
    result->stair_edge_score = corridor_cells > 0
        ? static_cast<float>(evidence_cells) / static_cast<float>(corridor_cells) : 0.0f;

    for (int selected = 0; selected < 2; ++selected) {
        int best_row = -1;
        float best_score = 0.08f;
        for (int y = kGrid / 5; y < kGrid; ++y) {
            if (result->stair_edge_rows[0] >= 0 &&
                std::abs(y - result->stair_edge_rows[0]) < 3) continue;
            if (row_scores[y] > best_score) {
                best_score = row_scores[y];
                best_row = y;
            }
        }
        if (best_row >= 0) {
            result->stair_edge_rows[result->stair_edge_count++] = best_row;
            if (selected == 0) {
                result->stair_edge_peak = best_score;
                result->stair_edge_span_ratio = row_spans[best_row];
            }
        }
    }

    float upper_depth = 0.0f;
    float lower_depth = 0.0f;
    int upper_count = 0;
    int lower_count = 0;
    if (result->stair_edge_count > 0) {
        const int row = result->stair_edge_rows[0];
        for (int y = std::max(0, row - 2); y <= std::min(kGrid - 1, row + 2); ++y) {
            for (int x = 0; x < kGrid; ++x) {
                if (!CellInCorridor(x, y, 1)) continue;
                const float value = result->depth_m[y * kGrid + x];
                if (!(value > 0.0f) || !std::isfinite(value)) continue;
                if (y < row) { upper_depth += value; ++upper_count; }
                if (y > row) { lower_depth += value; ++lower_count; }
            }
        }
    }
    const float depth_bins = (upper_count > 0 && lower_count > 0)
        ? std::fabs(std::log((upper_depth / upper_count) / (lower_depth / lower_count))) /
              std::log(kDepthMaxM / kDepthMinM) * kDepthBins
        : 0.0f;
    result->stair_depth_jump_bins = depth_bins;
    const bool semantic_evidence = result->center.step_ratio >= 0.04f &&
        result->center.step_ratio <= 0.35f &&
        result->center.step_largest_component >= 12;
    const bool edge_evidence = result->stair_edge_peak >= 0.55f &&
        result->stair_edge_span_ratio >= 0.45f;
    const bool depth_evidence = depth_bins >= 2.0f;
    const int evidence_count = static_cast<int>(semantic_evidence) +
        static_cast<int>(edge_evidence) + static_cast<int>(depth_evidence);
    const bool suspect_candidate = evidence_count >= 2;
    const bool confirm_candidate = semantic_evidence && edge_evidence && depth_evidence;

    stair_suspect_history_.push_back(suspect_candidate);
    while (stair_suspect_history_.size() > 5U) stair_suspect_history_.pop_front();
    stair_confirm_history_.push_back(confirm_candidate);
    while (stair_confirm_history_.size() > 6U) stair_confirm_history_.pop_front();
    int suspect_hits = 0;
    int confirm_hits = 0;
    for (bool value : stair_suspect_history_) if (value) ++suspect_hits;
    for (bool value : stair_confirm_history_) if (value) ++confirm_hits;

    if (stair_state_ == STAIR_CONFIRMED) {
        if (confirm_candidate) stair_confirm_clear_count_ = 0;
        else if (++stair_confirm_clear_count_ >= 5) {
            stair_state_ = suspect_hits >= 3 ? STAIR_SUSPECTED : STAIR_NONE;
            stair_confirm_clear_count_ = 0;
        }
    } else if (stair_confirm_history_.size() >= 4U && confirm_hits >= 4) {
        stair_state_ = STAIR_CONFIRMED;
        stair_confirm_clear_count_ = 0;
    } else if (stair_state_ == STAIR_SUSPECTED) {
        if (suspect_candidate) stair_suspect_clear_count_ = 0;
        else if (++stair_suspect_clear_count_ >= 3) {
            stair_state_ = STAIR_NONE;
            stair_suspect_clear_count_ = 0;
        }
    } else if (stair_suspect_history_.size() >= 3U && suspect_hits >= 3) {
        stair_state_ = STAIR_SUSPECTED;
        stair_suspect_clear_count_ = 0;
    }

    result->stair_state = stair_state_;
    result->stair_edge_persistent = stair_state_ == STAIR_CONFIRMED;
    result->center.persistent_hazard = result->stair_edge_persistent;
    result->center.safe_candidate = result->center.safe_candidate &&
        stair_state_ == STAIR_NONE;
    if (stair_state_ == STAIR_CONFIRMED) {
        result->center.persistent_hazard = true;
        result->center.safe_candidate = false;
        result->primary_hazard = "step_or_drop";
        result->primary_sector = "center";
        result->proximity = result->depth_level;
    } else if (stair_state_ == STAIR_SUSPECTED) {
        result->primary_hazard = "possible_step";
        result->primary_sector = "center";
        result->proximity = result->depth_level;
    } else if (result->center.blocked_persistent) {
        result->primary_hazard = "blocked_surface";
        result->primary_sector = "center";
    } else if (result->center.safe_candidate) {
        result->primary_hazard = "none";
        result->primary_sector = "unknown";
    } else {
        result->primary_hazard = result->center.unknown_ratio >= kUnknownMaxRatio
            ? "unknown_other" : "unknown";
        result->primary_sector = "center";
    }
}

bool SurfaceSegmenter::PostprocessPackedLogits(const float* scene_logits,
                                               size_t scene_count,
                                               bool hwc_layout,
                                               int64_t timestamp_ms,
                                               SurfaceResult* result)
{
    if (scene_logits == NULL || result == NULL ||
        scene_count != static_cast<size_t>(kGridCells * UNIFIED_SCENE_CHANNELS)) {
        return false;
    }
    std::vector<float> seg(static_cast<size_t>(kGridCells * kClasses));
    std::vector<float> depth(static_cast<size_t>(kGridCells * kDepthBins));
    for (int y = 0; y < kGrid; ++y) {
        for (int x = 0; x < kGrid; ++x) {
            for (int channel = 0; channel < kClasses; ++channel) {
                const size_t source = hwc_layout
                    ? static_cast<size_t>((y * kGrid + x) * UNIFIED_SCENE_CHANNELS + channel)
                    : static_cast<size_t>(channel * kGridCells + y * kGrid + x);
                const size_t target = hwc_layout
                    ? static_cast<size_t>((y * kGrid + x) * kClasses + channel)
                    : static_cast<size_t>(channel * kGridCells + y * kGrid + x);
                seg[target] = scene_logits[source];
            }
            for (int bin = 0; bin < kDepthBins; ++bin) {
                const int channel = kClasses + bin;
                const size_t source = hwc_layout
                    ? static_cast<size_t>((y * kGrid + x) * UNIFIED_SCENE_CHANNELS + channel)
                    : static_cast<size_t>(channel * kGridCells + y * kGrid + x);
                const size_t target = hwc_layout
                    ? static_cast<size_t>((y * kGrid + x) * kDepthBins + bin)
                    : static_cast<size_t>(bin * kGridCells + y * kGrid + x);
                depth[target] = scene_logits[source];
            }
        }
    }
    if (!PostprocessLogits(seg.data(), seg.size(), depth.data(), depth.size(),
                           hwc_layout, timestamp_ms, result)) {
        return false;
    }
    // 统一模型中 segmentation 只提供候选证据，不能独自触发台阶停止。
    result->left.persistent_hazard = false;
    result->center.persistent_hazard = false;
    result->right.persistent_hazard = false;
    DecodeStairEdge(scene_logits, hwc_layout, result);
    return true;
}

}  // namespace obstacle
