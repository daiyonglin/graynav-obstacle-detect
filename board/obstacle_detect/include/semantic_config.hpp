#pragma once

#include <string>

namespace obstacle {
namespace semantic {

/**
 * @brief 板端避障语义层。
 *
 * 当前统一模型直接输出固定 Indoor8 顺序；决策层再映射为 8 类导航语义。
 * 旧 ROD25/COCO80 分支仅保留为历史兼容代码，不进入新镜像。
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

/** 当前部署模型分类头通道数，来自 CMake 编译宏，必须与 m1model 一致。 */
int ModelClassCount();

/** 判断 raw class 是否允许进入后处理；Indoor8 的 0..7 全部有效。 */
bool IsSupportedRawClass(int raw_class_id);

/** 把模型原始类别映射到 8 类导航语义，不改变 raw_label。 */
int SemanticClassFromRaw(int raw_class_id);

/** PERSON 之外的导航语义是否属于一般实体障碍。 */
bool IsObstacleClass(int semantic_class_id);

/** 家具、小物体和车辆辅助判定，供阈值、NMS 和规划共享。 */
bool IsFurnitureLikeRawClass(int raw_class_id);
bool IsFurnitureLikeSemantic(int semantic_class_id);
bool IsSmallObjectSemantic(int semantic_class_id);
bool IsVehicleSemantic(int semantic_class_id);

/** 每个 raw class 的召回阈值；与避障距离阈值相互独立。 */
float CandidateThreshold(int raw_class_id);

/** 导航排序权重，不回写或篡改神经网络置信度。 */
float RiskWeight(int semantic_class_id);

// 全链路共享的避障阈值。所有值都可由同名 A1_* 环境变量覆盖，确保测距、
// tracker、规划器和串口显示不会各自使用互相矛盾的距离边界。
float UrgentDistanceM();
float NearDistanceM();
float WarningDistanceM();
float SideClearDistanceM();
float TurnClearanceMarginM();

// 窄视场相机的画面分区与地面走廊边界。四项参数均支持 A1_* 环境变量覆盖。
float SectorLeftBoundaryRatio();
float SectorRightBoundaryRatio();
float WideBoxRatio();
float CenterCorridorHalfWidthM();

/**
 * 墙面/blocked_surface 是否可以直接改变导航动作。
 *
 * 当前室内灰度模型会把部分地面纹理误分为 blocked_surface，因此生产镜像
 * 默认关闭墙面动作，只保留其诊断结果；台阶/落差链路不受该开关影响。
 */
bool WallGuidanceEnabled();

/** 导航语义全名、OSD 简写和模型原始类别名。 */
std::string SemanticLabel(int semantic_class_id);
std::string SemanticShortLabel(int semantic_class_id);
std::string RawLabel(int raw_class_id);

}  // namespace semantic
}  // namespace obstacle
