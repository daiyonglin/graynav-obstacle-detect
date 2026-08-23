#pragma once

#include "common.hpp"

#include <array>
#include <string>

namespace obstacle {

/**
 * @brief 将稳定目标投影到左、中、右通行走廊并生成保守的导盲动作。
 *
 * 规划器只消费 tracker 输出，不直接读取神经网络结果。动作由稳定距离和目标
 * 横向位置决定：中央远/中/近目标依次给出 clear/slow/stop；停车后用户轻微
 * 左右试探，使目标进入侧区，规划器随即给出反方向转向。TTC 仅可保留为诊断
 * 信息，绝不参与动作。StabilizeAction 负责风险升级、停车释放和方向滞回。
 */
class AvoidancePlanner {
public:
    AvoidancePlanner();

    void Initialize(const std::array<int, 2>& image_shape);
    AvoidanceDecision Update(const DetectionResult& result, int view_id, int64_t timestamp_ms);

public:
    /** 一条地面通行走廊的最近障碍、动作距离和双 ROI 观测完备性。 */
    struct Corridor {
        ZoneStatus zone;
        float clearance;
        bool verified;
        Corridor() : clearance(8.0f), verified(false) {}
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
