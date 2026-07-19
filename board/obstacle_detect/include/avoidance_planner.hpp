#pragma once

#include "common.hpp"

#include <array>
#include <string>

namespace obstacle {

/**
 * @brief 将稳定目标投影到左、中、右通行走廊并生成保守的导盲动作。
 *
 * 规划器只消费 tracker 输出，不直接读取神经网络结果。它使用安全距离、TTC、
 * 目标横向位置和质量更新三条走廊。单纯左/右侧近障可直接提示向反方向转向，
 * 使窄视场下的侧方障碍能够及时触发动作；只有“中央走廊被阻挡、需要选择绕行
 * 侧”时，才要求双 ROI 的近期观测和候选走廊净空证据。StabilizeAction 负责
 * 风险升级立即生效、风险降低延迟确认和左右反转滞回。
 */
class AvoidancePlanner {
public:
    AvoidancePlanner();

    void Initialize(const std::array<int, 2>& image_shape);
    AvoidanceDecision Update(const DetectionResult& result, int view_id, int64_t timestamp_ms);

public:
    /** 一条地面通行走廊的最近障碍、净空距离、最小 TTC 和双 ROI 观测完备性。 */
    struct Corridor {
        ZoneStatus zone;
        float clearance;
        float min_ttc;
        bool verified;
        Corridor() : clearance(8.0f), min_ttc(-1.0f), verified(false) {}
    };

private:
    bool IsActionHazard(const DetectionItem& item) const;
    void AddToCorridor(Corridor* corridor, const DetectionItem& item) const;
    std::string StabilizeAction(const std::string& desired, int64_t now_ms);

    std::array<int, 2> image_shape_;
    int64_t last_view_ms_[2];
    std::string stable_action_;
    std::string pending_action_;
    int pending_count_;
    int64_t stable_since_ms_;
    int64_t pending_since_ms_;
};

}  // namespace obstacle
