#pragma once

#include "common.hpp"

#include <array>
#include <string>

namespace obstacle {

/**
 * @brief 将稳定目标投影到左、中、右通行走廊并生成保守的导盲动作。
 *
 * 规划器只消费 tracker 输出，不直接读取神经网络结果。它使用安全距离、TTC、
 * 目标横向位置和质量更新三条走廊；只有上下两个 ROI 均在近期完成观测且候选
 * 侧向走廊有足够安全余量时才输出 LEFT/RIGHT。侧方未知时宁可 STOP/SLOW，
 * 不给出未经验证的绕行方向。StabilizeAction 负责风险升级立即生效、风险降低
 * 延迟确认和左右反转滞回。
 */
class AvoidancePlanner {
public:
    AvoidancePlanner();

    void Initialize(const std::array<int, 2>& image_shape);
    AvoidanceDecision Update(const DetectionResult& result, int view_id, int64_t timestamp_ms);

public:
    /** 一条地面通行走廊的最近障碍、净空距离、最小 TTC 和观测完备性。 */
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
