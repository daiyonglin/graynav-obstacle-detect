#pragma once

#include "osd-device.hpp"
#include <algorithm>

namespace utils {

    float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b);

    void SortDetectionResult(DetectionResult* result);

    /**
     * @brief 类别感知 NMS
     *
     * 只有 class_id 相同的框才互相抑制
     */
    void NMS(DetectionResult* result, float iou_threshold, int top_k);

    void MultiTargetNMS(DetectionResult* result, float iou_threshold, int top_k);

}  // namespace utils


/**
 * @brief OSD 可视化器
 *
 * 当前版本使用多层 OSD：
 * 1. 顶部危险进度条
 * 2. 低图元避障动作文字
 * 3. 方向/风险短文本
 * 4. 检测框
 */
class VISUALIZER {
public:
    void Initialize(std::array<int, 2>& in_img_shape);
    void Release();

    void DrawTestBox();

    void Draw(const DetectionResult& result);
    void Draw(const DetectionResult& result, const AvoidanceDecision& decision);

private:
    int PickColorByClass(int class_id) const;

private:
    sst::device::osd::OsdDevice osd_device;
    std::string last_action_asset_;
    std::string last_info_asset_;
    bool static_layers_cleaned_ = false;
};
