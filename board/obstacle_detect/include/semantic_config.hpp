#pragma once

#include <string>

namespace obstacle {
namespace semantic {

/**
 * @brief 板端避障语义层。
 *
 * 模型保留 ROD25 的 25 个原始类别以维持训练/导出一致性；决策层再把这些类别
 * 映射为 8 类导航语义。这样既能在日志中追溯 raw_label，又能用统一风险权重、
 * 显示名称和阈值处理功能相近的障碍物。
 */
enum SemanticClass {
    PERSON = 0,
    CHAIR_SEAT = 1,
    TABLE_DESK = 2,
    SOFA_BED = 3,
    BAG_SUITCASE = 4,
    SMALL_OBJECT = 5,
    VEHICLE_BICYCLE = 6,
    GENERIC_OBSTACLE = 7,
    NUM_SEMANTIC_CLASSES = 8
};

int ModelClassCount();  // 当前部署模型分类头通道数，必须与 m1model 一致。
bool IsSupportedRawClass(int raw_class_id);
int SemanticClassFromRaw(int raw_class_id);
bool IsObstacleClass(int semantic_class_id);
bool IsFurnitureLikeRawClass(int raw_class_id);
bool IsFurnitureLikeSemantic(int semantic_class_id);
bool IsSmallObjectSemantic(int semantic_class_id);
bool IsVehicleSemantic(int semantic_class_id);
float CandidateThreshold(int raw_class_id);
float RiskWeight(int semantic_class_id);

// 全链路共享的避障阈值。所有值都可由同名 A1_* 环境变量覆盖，确保测距、
// tracker、规划器和串口显示不会各自使用互相矛盾的距离边界。
float UrgentDistanceM();
float NearDistanceM();
float WarningDistanceM();
float StopTtcSeconds();
float SideClearDistanceM();
float TurnClearanceMarginM();

// 窄视场相机的画面分区与地面走廊边界。四项参数均支持 A1_* 环境变量覆盖。
float SectorLeftBoundaryRatio();
float SectorRightBoundaryRatio();
float WideBoxRatio();
float CenterCorridorHalfWidthM();

std::string SemanticLabel(int semantic_class_id);
std::string SemanticShortLabel(int semantic_class_id);
std::string RawLabel(int raw_class_id);

}  // namespace semantic
}  // namespace obstacle
