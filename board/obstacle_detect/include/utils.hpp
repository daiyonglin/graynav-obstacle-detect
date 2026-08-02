#pragma once

#include "osd-device.hpp"

#include <algorithm>

namespace utils {

/** 计算两个 xyxy 检测框的交并比，所有 NMS 与 tracker 关联共用该定义。 */
float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b);

/** 按导航风险、保守距离、质量和置信度对结果排序。 */
void SortDetectionResult(DetectionResult* result);

/** @brief 标准同类 NMS，仅在 class_id 相同时执行重叠抑制。 */
void NMS(DetectionResult* result, float iou_threshold, int top_k);

/**
 * @brief 面向多障碍场景的跨类去重与保护式 NMS。
 *
 * 它在删除同一实体的跨类重复框时，保护横向分离的小目标，并防止低质量
 * wide/coarse 大框吞并内部的 person、chair、bench 等可靠框。
 */
void MultiTargetNMS(DetectionResult* result, float iou_threshold, int top_k);

}  // namespace utils

/**
 * @brief Aurora 多层 OSD 可视化器。
 *
 * 分层绘制检测框、顶部动作、方向/风险信息和三走廊状态。静态文字纹理由
 * 预生成资源显示，动态框使用矢量矩形，避免每帧重载全部图层。
 */
class VISUALIZER {
public:
    void Initialize(std::array<int, 2>& in_img_shape);
    void Release();

    void DrawTestBox();

    void Draw(const DetectionResult& result);
    void Draw(const DetectionResult& result, const AvoidanceDecision& decision);
    void Draw(const DetectionResult& result,
              const AvoidanceDecision& decision,
              const SurfaceResult& surface);

private:
    int PickColorByClass(int class_id) const;

private:
    sst::device::osd::OsdDevice osd_device;
    std::string last_action_asset_;
    std::string last_info_asset_;
    bool static_layers_cleaned_ = false;
    std::array<int, 2> image_shape_ = {720, 1280};
};
