#pragma once

#include "common.hpp"

#include <array>
#include <deque>
#include <string>
#include <vector>

namespace obstacle {

/**
 * @brief 单模型 packed scene 输出的 CPU 道路走廊后处理。
 *
 * NPU 模型生命周期由 YOLOV8GRAY 唯一持有。本类不加载模型、不创建
 * SSNE tensor，只解码统一模型的 4+16+1 通道输出并执行时序投票。
 */
class SurfaceSegmenter {
public:
    SurfaceSegmenter();

    /** 离线/板端 golden test 共用的双头 raw logits 后处理入口。 */
    bool PostprocessLogits(const float* seg_logits,
                           size_t seg_count,
                           const float* depth_logits,
                           size_t depth_count,
                           bool hwc_layout,
                           int64_t timestamp_ms,
                           SurfaceResult* result);

    /** Decode one packed 4-seg + 16-depth + 1-stair-edge scene tensor. */
    bool PostprocessPackedLogits(const float* scene_logits,
                                 size_t scene_count,
                                 bool hwc_layout,
                                 int64_t timestamp_ms,
                                 SurfaceResult* result);

private:
    struct CorridorStats {
        std::array<int, SURFACE_CLASS_COUNT> counts;
        std::array<int, SURFACE_CLASS_COUNT> largest_components;
        int total;
        int lowest_hazard_y;
        CorridorStats()
            : counts{},
              largest_components{},
              total(0),
              lowest_hazard_y(-1) {}
    };

    void MajorityFilter(const std::array<uint8_t, SURFACE_GRID_CELLS>& input,
                        std::array<uint8_t, SURFACE_GRID_CELLS>* output) const;
    CorridorStats MeasureCorridor(const std::array<uint8_t, SURFACE_GRID_CELLS>& labels,
                                  int corridor_index) const;
    void UpdateTemporalState(const std::array<SurfaceCorridor, 3>& current,
                             std::array<SurfaceCorridor, 3>* stable);
    static SurfaceCorridor BuildCorridor(const CorridorStats& stats);
    static bool CellInCorridor(int x, int y, int corridor_index);
    static std::string DepthLevel(float depth_m, float confidence);
    static float Median(std::vector<float>* values);
    std::string StabilizeDepthLevel(const std::string& candidate,
                                    float confidence,
                                    float margin);
    void DecodeDepth(const float* logits,
                     bool hwc_layout,
                     const std::array<uint8_t, SURFACE_GRID_CELLS>& labels,
                     SurfaceResult* result);
    void DecodeStairEdge(const float* scene_logits,
                         bool hwc_layout,
                         SurfaceResult* result);

    std::deque<std::array<SurfaceCorridor, 3> > history_;
    bool hazard_latched_[3];
    int hazard_clear_count_[3];
    bool blocked_latched_[3];
    int blocked_clear_count_[3];
    std::deque<float> center_depth_history_;
    std::deque<std::string> center_depth_level_history_;
    std::deque<bool> stair_edge_history_;
    std::string stable_depth_level_;
};

}  // namespace obstacle
