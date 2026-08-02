#pragma once

#include "common.hpp"

#include <array>
#include <deque>
#include <string>
#include <vector>

namespace obstacle {

/**
 * @brief A1 单通道 Fast-SCNN 推理与道路走廊后处理。
 *
 * 模型输入固定为 1x1x256x256 Y8，输出固定为 1x4x32x32 raw logits。
 * Softmax/ArgMax、多数滤波、连通域和三帧投票全部在 CPU 完成。
 */
class SurfaceSegmenter {
public:
    SurfaceSegmenter();

    bool Initialize(const std::string& model_path,
                    const std::array<int, 2>& image_shape,
                    const std::array<int, 2>& input_shape);
    bool Predict(ssne_tensor_t* image, SurfaceResult* result);
    void Release();

    bool Available() const { return available_; }
    uint16_t ModelId() const { return model_id_; }
    SegmenterTiming GetLastTiming() const { return last_timing_; }
    std::array<int, 4> Roi() const { return roi_; }

    /** 离线/板端 golden test 共用的 raw logits 后处理入口。 */
    bool PostprocessLogits(const float* logits,
                           size_t element_count,
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
            : counts{0, 0, 0, 0},
              largest_components{0, 0, 0, 0},
              total(0),
              lowest_hazard_y(-1) {}
    };

    bool Preprocess(ssne_tensor_t* image);
    bool ReadOutputLogits(std::vector<float>* logits, bool* hwc_layout) const;
    void MajorityFilter(const std::array<uint8_t, 1024>& input,
                        std::array<uint8_t, 1024>* output) const;
    CorridorStats MeasureCorridor(const std::array<uint8_t, 1024>& labels,
                                  int corridor_index) const;
    void UpdateTemporalState(const std::array<SurfaceCorridor, 3>& current,
                             std::array<SurfaceCorridor, 3>* stable);
    static SurfaceCorridor BuildCorridor(const CorridorStats& stats);
    static bool CellInCorridor(int x, int y, int corridor_index);

    uint16_t model_id_;
    bool available_;
    bool input_created_;
    bool output_created_;
    ssne_tensor_t input_;
    ssne_tensor_t output_;
    AiPreprocessPipe preprocess_pipe_;
    std::array<int, 2> image_shape_;
    std::array<int, 2> input_shape_;
    std::array<int, 4> roi_;
    SegmenterTiming last_timing_;
    std::deque<std::array<SurfaceCorridor, 3> > history_;
    bool hazard_latched_[3][2];
    int hazard_clear_count_[3][2];
};

}  // namespace obstacle
