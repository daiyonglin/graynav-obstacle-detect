#pragma once

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <array>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include "smartsoc/ssne_api.h"
#include "semantic_config.hpp"

/**
 * @brief 板端各模块共享的单目标数据结构。
 *
 * 数据从 YOLO 后处理开始建立，随后依次被语义映射、跟踪器、测距器和
 * 避障规划器补充。`box` 始终使用完整 720x1280 Aurora 画面的坐标系，
 * 因此上、下双 ROI 的结果在离开检测器前必须先反 letterbox、再加 ROI
 * 原点。这样跟踪、测距和 OSD 不需要理解模型输入坐标。
 */
enum DisplayClass {
    DISPLAY_CLASS_PERSON = obstacle::semantic::PERSON,
    DISPLAY_CLASS_OBSTACLE = obstacle::semantic::GENERIC_OBSTACLE
};

struct DetectionItem {
    std::array<float, 4> box;  // [xmin,ymin,xmax,ymax]，完整传感器画面坐标。
    float score;               // sigmoid 后的模型分类置信度。
    int class_id;              // 导航语义类别，定义见 semantic_config.hpp。
    int raw_class_id;          // 当前模型检测头的原始类别编号（统一模型为 Indoor8）。
    std::string label;         // 对外显示的稳定语义名称。
    std::string semantic_class;// 与 label 同域，供 JSON/串口接口使用。
    std::string raw_label;     // 当前模型原始类别名，便于诊断误分类。
    float risk_weight;         // 类别风险权重，参与候选排序和规划。
    std::string sector;        // left/center/right/交叠区/wide。
    float distance_m;          // 融合后的期望距离；负值表示无可靠米级结果。
    float safe_distance_m;     // mean-sigma 保守下界；仅在期望距离缺失时供规划兜底。
    float distance_sigma_m;    // 距离标准差，描述几何与尺寸先验不确定度。
    float lateral_m;           // 相机坐标系横向位置，左负右正。
    std::string distance_source; // ground/size/fused/nearfield/unknown。
    float distance_confidence; // 0~1 的测距可信度。
    std::string risk_level;    // urgent/near/warning/far/unknown。
    std::string quality;       // good/low/coarse；coarse 不输出伪精确距离。
    int track_id;              // 多目标跟踪器分配的稳定编号。
    int age;                   // 轨迹生命周期帧数。
    int missed;                // 当前可见 ROI 内连续未匹配次数。
    float approach_mps;        // 朝向相机的径向速度，非接近时为 0。
    float ttc_s;               // 诊断用碰撞时间；证据不足时为负值，不参与动作。
    int range_measurements;    // 该轨迹累计的可靠测距次数。
    std::string depth_level;   // near/mid/far/unknown，仅用于稳健远近表达。
    float depth_confidence;    // 学习深度与几何证据融合后的 0~1 置信度。
    std::string depth_source;  // geometry/learned/fused/conflict/unknown。
    bool depth_consistent;     // 几何与学习深度的相对差异是否在 40% 内。
    bool approaching;          // tracker 或稠密深度是否显示目标持续接近。

    DetectionItem()
        : box{0.f, 0.f, 0.f, 0.f},
          score(0.f),
          class_id(-1),
          raw_class_id(-1),
          label(""),
          semantic_class(""),
          raw_label(""),
          risk_weight(1.0f),
          sector("unknown"),
          distance_m(-1.0f),
          safe_distance_m(-1.0f),
          distance_sigma_m(-1.0f),
          lateral_m(0.0f),
          distance_source("unknown"),
          distance_confidence(0.0f),
          risk_level("unknown"),
          quality("low"),
          track_id(-1),
          age(0),
          missed(0),
          approach_mps(0.0f),
          ttc_s(-1.0f),
          range_measurements(0),
          depth_level("unknown"),
          depth_confidence(0.0f),
          depth_source("unknown"),
          depth_consistent(false),
          approaching(false) {}
};

/** @brief 左、中、右单条通行走廊的最近风险摘要。 */
struct ZoneStatus {
    std::string dir;
    bool occupied;
    std::string label;
    std::string raw_label;
    std::string semantic_class;
    float risk_weight;
    float distance_m;          // Conservative distance used by the planner.
    float distance_estimate_m; // Expected distance exposed by serial output.
    float safe_distance_m;
    std::string risk_level;
    int object_count;

    ZoneStatus()
        : dir("unknown"),
          occupied(false),
          label(""),
          raw_label(""),
          semantic_class(""),
          risk_weight(1.0f),
          distance_m(-1.0f),
          distance_estimate_m(-1.0f),
          safe_distance_m(-1.0f),
          risk_level("unknown"),
          object_count(0) {}
};

/** @brief 单通道道路分割模型的四类部署契约。 */
enum SurfaceClass {
    GROUND_CANDIDATE = 0,
    BLOCKED_SURFACE = 1,
    STEP_OR_DROP = 2,
    UNKNOWN_OTHER = 3,
    SURFACE_CLASS_COUNT = 4
};

/** @brief 台阶证据状态。疑似只用于减速提示，确认后才允许触发停止。 */
enum StairState {
    STAIR_NONE = 0,
    STAIR_SUSPECTED = 1,
    STAIR_CONFIRMED = 2
};

// The unified model emits scene_logits on its P3 grid: 384 / 8 = 48.
static const int SURFACE_GRID_SIZE = 48;
static const int SURFACE_GRID_CELLS = SURFACE_GRID_SIZE * SURFACE_GRID_SIZE;
static const int DEPTH_BIN_COUNT = 16;
static const int UNIFIED_SCENE_CHANNELS = SURFACE_CLASS_COUNT + DEPTH_BIN_COUNT + 1;
static const int STAIR_EDGE_CHANNEL = UNIFIED_SCENE_CHANNELS - 1;

/** @brief 分割网格投影到单条导航走廊后的面积比例和稳定风险。 */
struct SurfaceCorridor {
    float ground_ratio;
    float blocked_ratio;
    float step_ratio;
    float unknown_ratio;
    int step_largest_component;
    int blocked_largest_component;
    bool safe_candidate;
    bool persistent_hazard;
    bool blocked_persistent;

    SurfaceCorridor()
        : ground_ratio(0.0f),
          blocked_ratio(0.0f),
          step_ratio(0.0f),
          unknown_ratio(0.0f),
          step_largest_component(0),
          blocked_largest_component(0),
          safe_candidate(false),
          persistent_hazard(false),
          blocked_persistent(false) {}
};

/** @brief 一次分割推理经过多数滤波、走廊统计和时序投票后的道路风险。 */
struct SurfaceResult {
    bool valid;
    bool stale;
    bool perception_degraded;
    int64_t timestamp_ms;
    SurfaceCorridor left;
    SurfaceCorridor center;
    SurfaceCorridor right;
    std::string primary_hazard;
    std::string primary_sector;
    std::string proximity;
    float confidence;
    std::string depth_level;
    float depth_confidence;
    float depth_margin;
    bool depth_ambiguous;
    std::array<float, 3> depth_group_probabilities;
    std::string depth_source;
    bool depth_consistent;
    bool approaching;
    float stair_edge_score;
    float stair_edge_peak;
    float stair_edge_span_ratio;
    float stair_depth_jump_bins;
    bool stair_edge_occluded_by_object;
    StairState stair_state;
    std::array<int, 2> stair_edge_rows;
    int stair_edge_count;
    bool stair_edge_persistent;
    bool stair_box_valid;
    std::array<float, 4> stair_box_norm;
    float stair_edge_x1_norm;
    float stair_edge_x2_norm;
    float stair_edge_y_norm;
    float center_depth_m;  // 内部融合使用，不对 Aurora/语音显示米数。
    std::array<uint8_t, SURFACE_GRID_CELLS> labels;
    std::array<float, SURFACE_GRID_CELLS> depth_m;
    std::array<float, SURFACE_GRID_CELLS> depth_cell_confidence;

    SurfaceResult()
        : valid(false),
          stale(true),
          perception_degraded(false),
          timestamp_ms(0),
          primary_hazard("unknown"),
          primary_sector("unknown"),
          proximity("unknown"),
          confidence(0.0f),
          depth_level("unknown"),
          depth_confidence(0.0f),
          depth_margin(0.0f),
          depth_ambiguous(true),
          depth_group_probabilities{0.0f, 0.0f, 0.0f},
          depth_source("unknown"),
          depth_consistent(false),
          approaching(false),
          stair_edge_score(0.0f),
          stair_edge_peak(0.0f),
          stair_edge_span_ratio(0.0f),
          stair_depth_jump_bins(0.0f),
          stair_edge_occluded_by_object(false),
          stair_state(STAIR_NONE),
          stair_edge_rows{-1, -1},
          stair_edge_count(0),
          stair_edge_persistent(false),
          stair_box_valid(false),
          stair_box_norm{0.0f, 0.0f, 0.0f, 0.0f},
          stair_edge_x1_norm(0.0f),
          stair_edge_x2_norm(0.0f),
          stair_edge_y_norm(0.0f),
          center_depth_m(-1.0f),
          labels{},
          depth_m{},
          depth_cell_confidence{} {}
};

/**
 * @brief 避障规划器的最终输出。
 *
 * `action` 只允许 clear/slow/stop/turn_left/turn_right/system_fault，
 * OSD、串口和语音都消费同一个动作，避免不同输出端给出冲突建议。
 */
struct AvoidanceDecision {
    ZoneStatus left;
    ZoneStatus center;
    ZoneStatus right;
    std::string action;
    std::string prompt;
    int nearest_track_id;
    std::string hazard_type;
    std::string hazard_sector;
    std::string perception_source;
    float surface_confidence;
    bool perception_degraded;
    std::string depth_level;
    float depth_confidence;
    float depth_margin;
    bool depth_ambiguous;
    std::string depth_source;
    bool depth_consistent;
    bool approaching;
    std::string cause;
    std::string range;
    std::string object_label;
    std::string scene_label;
    std::string recommended_direction;
    std::string hazard_position;
    std::string primary_class;
    float distance_estimate_m;
    std::string risk;
    float confidence;
    bool ai_ok;

    AvoidanceDecision()
        : action("clear"),
          prompt("clear"),
          nearest_track_id(-1),
          hazard_type("none"),
          hazard_sector("unknown"),
          perception_source("detection"),
          surface_confidence(0.0f),
          perception_degraded(false),
          depth_level("unknown"),
          depth_confidence(0.0f),
          depth_margin(0.0f),
          depth_ambiguous(true),
          depth_source("unknown"),
          depth_consistent(false),
          approaching(false),
          cause("NONE"),
          range("UNKNOWN"),
          object_label("NONE"),
          scene_label("UNKNOWN"),
          recommended_direction("forward"),
          hazard_position("FRONT"),
          primary_class("none"),
          distance_estimate_m(-1.0f),
          risk("UNKNOWN"),
          confidence(0.0f),
          ai_ok(true) {
        left.dir = "left";
        center.dir = "center";
        right.dir = "right";
    }
};

/** Compact, stable per-corridor summary shared by UART and future clients. */
struct GuidanceZone {
    bool occupied;
    std::string object_class;
    float distance_estimate_m;
    float safe_distance_m;
    std::string risk;

    GuidanceZone()
        : occupied(false),
          object_class("clear"),
          distance_estimate_m(-1.0f),
          safe_distance_m(-1.0f),
          risk("SAFE") {}
};

/**
 * @brief 对原始融合决策做过非对称时序稳定后的唯一演示状态。
 *
 * Aurora、可读串口和 SYN6288 都必须由同一个 StableGuidance 生成，避免
 * 距离、方位或目标标签在相邻帧分别抖动成互相矛盾的提示。
 */
struct StableGuidance {
    std::string action;
    std::string cause;
    std::string range;
    std::string sector;
    std::string object_label;
    std::string scene_label;
    std::string recommended_direction;
    std::string hazard_position;
    std::string primary_class;
    float distance_estimate_m;
    std::string risk;
    GuidanceZone left;
    GuidanceZone center;
    GuidanceZone right;
    float confidence;
    bool ai_ok;
    uint64_t timestamp_ms;

    StableGuidance()
        : action("slow"),
          cause("UNKNOWN"),
          range("UNKNOWN"),
          sector("CENTER"),
          object_label("NONE"),
          scene_label("UNKNOWN"),
          recommended_direction("forward"),
          hazard_position("FRONT"),
          primary_class("none"),
          distance_estimate_m(-1.0f),
          risk("UNKNOWN"),
          confidence(0.0f),
          ai_ok(true),
          timestamp_ms(0) {}

    void ApplyTo(AvoidanceDecision* decision) const {
        if (decision == NULL) return;
        decision->action = action;
        decision->cause = cause;
        decision->range = range;
        decision->hazard_sector = sector;
        decision->object_label = object_label;
        decision->scene_label = scene_label;
        decision->recommended_direction = recommended_direction;
        decision->hazard_position = hazard_position;
        decision->primary_class = primary_class;
        decision->distance_estimate_m = distance_estimate_m;
        decision->risk = risk;
        decision->left.occupied = left.occupied;
        decision->left.raw_label = left.object_class;
        decision->left.distance_estimate_m = left.distance_estimate_m;
        decision->left.safe_distance_m = left.safe_distance_m;
        decision->left.distance_m = left.safe_distance_m;
        decision->left.risk_level = left.risk;
        decision->center.occupied = center.occupied;
        decision->center.raw_label = center.object_class;
        decision->center.distance_estimate_m = center.distance_estimate_m;
        decision->center.safe_distance_m = center.safe_distance_m;
        decision->center.distance_m = center.safe_distance_m;
        decision->center.risk_level = center.risk;
        decision->right.occupied = right.occupied;
        decision->right.raw_label = right.object_class;
        decision->right.distance_estimate_m = right.distance_estimate_m;
        decision->right.safe_distance_m = right.safe_distance_m;
        decision->right.distance_m = right.safe_distance_m;
        decision->right.risk_level = right.risk;
        decision->confidence = confidence;
        decision->ai_ok = ai_ok;
        decision->depth_level = range == "NEAR" ? "near" :
            (range == "MID" ? "mid" : (range == "FAR" ? "far" : "unknown"));
    }
};

/**
 * @brief 一次 NPU 推理或一次稳定跟踪输出的目标集合。
 *
 * `raw_candidate_count`、`post_nms_count` 和 `coarse_drop_count` 用于判断
 * 后处理是否异常爆炸；`view_id/roi` 用于双 ROI 跟踪时判断目标当前是否
 * 应当可见，防止奇偶帧切换造成误删或残留框。
 */
struct DetectionResult {
    std::vector<DetectionItem> items;
    int raw_candidate_count;
    int post_nms_count;
    int coarse_drop_count;
    int view_id;
    std::array<int, 4> roi;  // 当前推理 ROI 的全图坐标 [x1,y1,x2,y2]。
    int64_t timestamp_ms;

    DetectionResult()
        : raw_candidate_count(0),
          post_nms_count(0),
          coarse_drop_count(0),
          view_id(0),
          roi{0, 0, 0, 0},
          timestamp_ms(0) {}

    void Clear() {
        items.clear();
        raw_candidate_count = 0;
        post_nms_count = 0;
        coarse_drop_count = 0;
        view_id = 0;
        roi = {0, 0, 0, 0};
        timestamp_ms = 0;
    }

    void Free() {
        std::vector<DetectionItem>().swap(items);
        raw_candidate_count = 0;
        post_nms_count = 0;
        coarse_drop_count = 0;
        view_id = 0;
        roi = {0, 0, 0, 0};
        timestamp_ms = 0;
    }

    size_t Size() const {
        return items.size();
    }
};

/** One invocation of the unified network: detections and packed scene evidence. */
struct UnifiedPerceptionResult {
    DetectionResult detections;
    SurfaceResult surface;
    float stair_edge_score;
    std::array<int, 2> stair_edge_rows;
    std::string depth_level;
    int64_t timestamp_ms;
    bool valid;
    bool degraded;

    UnifiedPerceptionResult()
        : stair_edge_score(0.0f),
          stair_edge_rows{-1, -1},
          depth_level("unknown"),
          timestamp_ms(0),
          valid(false),
          degraded(true) {}
};

/** @brief ROI 等比例缩放到模型输入时的 scale 与 padding。 */
struct LetterboxInfo {
    int src_w;
    int src_h;
    int dst_w;
    int dst_h;
    float scale;
    int pad_x;
    int pad_y;

    LetterboxInfo()
        : src_w(0), src_h(0),
          dst_w(0), dst_h(0),
          scale(1.0f), pad_x(0), pad_y(0) {}
};

/** @brief 单帧检测各阶段耗时，用于 FPS、P95 和端到端延迟统计。 */
struct DetectorTiming {
    float preprocess_ms;
    float inference_ms;
    float output_ms;
    float postprocess_ms;
    DetectorTiming()
        : preprocess_ms(0.0f), inference_ms(0.0f), output_ms(0.0f), postprocess_ms(0.0f) {}
};

/** @brief 单帧道路分割各阶段耗时。 */
struct SegmenterTiming {
    float preprocess_ms;
    float inference_ms;
    float output_ms;
    float postprocess_ms;
    SegmenterTiming()
        : preprocess_ms(0.0f), inference_ms(0.0f), output_ms(0.0f), postprocess_ms(0.0f) {}
};

/**
 * @brief 图像获取模块
 *
 * 职责：
 * 1. 配置 A1 online pipeline，从 SC132GS 获取 Y8 单通道完整画面；
 * 2. 仅负责传感器级裁剪和输出格式，不做模型 ROI、resize 或 normalize；
 * 3. 取帧失败时支持关闭并重新打开 pipeline，供健康管理自动恢复。
 */
class IMAGEPROCESSOR {
public:
    void Initialize(std::array<int, 2>* in_img_shape);
    bool GetImage(ssne_tensor_t* img_sensor);
    bool Restart();
    void Release();

    std::array<int, 2> img_shape;  // [width, height]

private:
    bool ConfigureAndOpen();

    uint8_t format_online;
    bool pipeline_open = false;
};


/**
 * @brief 单模型 Indoor8 检测与 packed scene 感知器
 *
 * 输入：
 *   SC132GS 原始整幅灰度图，当前模型编译为真实 1 通道输入。
 *
 * 内部流程：
 *   1. AI preprocess pipe:
 *      - LOWER、LOWER、UPPER 三帧循环选择重叠 ROI；
 *      - 等比例 letterbox 到 384x384；
 *      - 需要时执行自适应灰度 LUT；
 *      - normalize 参数由 .m1model 的输入量化配置提供。
 *   2. NPU 推理：
 *      - 一个统一 m1model，一次输出 6 个检测 raw branch 和 1 个 scene21 branch。
 *   3. CPU 后处理：
 *      - 自动识别输出分支顺序
 *      - 校验 3 个 8 通道分类头、3 个 64 通道 DFL 回归头和 scene21；
 *      - 依据 runtime layout 以 HWC/CHW 正确读取量化输出；
 *      - top-1 分类、sigmoid、DFL、anchor 解码和多目标 NMS；
 *      - 过滤饱和横框/粗框并映射回完整 Aurora 坐标。
 */
namespace obstacle { class SurfaceSegmenter; }

class YOLOV8GRAY {
public:
    YOLOV8GRAY();
    ~YOLOV8GRAY();
    std::string ModelName() const { return "graynav_unified_indoor8_scene21"; }

    void Initialize(std::string& model_path,
                    std::array<int, 2>* in_img_shape,
                    std::array<int, 2>* in_det_shape);

    bool Predict(ssne_tensor_t* img_in,
                 DetectionResult* result,
                 float conf_threshold = 0.35f,
                 SurfaceResult* surface_result = NULL);

    void Release();

    LetterboxInfo GetLastLetterboxInfo() const { return lb_info_[active_view_]; }
    int ActiveView() const { return active_view_; }
    uint16_t ModelId() const { return model_id; }
    DetectorTiming GetLastTiming() const { return last_timing_; }

public:
    float nms_threshold = 0.60f;
    int top_k = 300;
    int keep_top_k = 40;

    std::array<int, 2> img_shape;   // [w, h]
    std::array<int, 2> det_shape;   // [w, h]
    std::array<int, 2> output_shape; // [w, h], full Aurora/OSD coordinate space

private:
    uint16_t model_id = 0;
    ssne_tensor_t inputs[1] = {};
    ssne_tensor_t outputs[7] = {};

    AiPreprocessPipe pipe_offline_[2] = {nullptr, nullptr};
    LetterboxInfo lb_info_[2];
    std::array<int, 4> roi_[2];
    int active_view_ = 0;
    int predict_count_ = 0;
    bool dual_roi_ = true;
    DetectorTiming last_timing_;

    std::vector<std::string> class_names_;
    obstacle::SurfaceSegmenter* scene_postprocessor_ = NULL;

private:
    /** 从模型类别契约建立原始标签表。 */
    void BuildClassNames();

    /** 选择当前 ROI，并完成 crop、letterbox、normalize 和可选暗光增强。 */
    bool Preprocess(ssne_tensor_t* img_in, ssne_tensor_t* input_tensor);

    /** 校验并解码 6 个检测 raw head，输出映射到全图坐标的检测结果。 */
    bool Postprocess(DetectionResult* result, float conf_threshold);

    /** Decode the packed 4+16+1 scene tensor produced with the detector heads. */
    bool PostprocessScene(SurfaceResult* result, int64_t timestamp_ms);

    /** 将 384 模型坐标去 padding/scale 后加 ROI 原点，恢复为 Aurora 坐标。 */
    void MapBoxToOriginalImage(std::array<float, 4>& box);

    /** 返回 Indoor8 原始类别名；越界编号返回 unknown。 */
    std::string ClassIdToLabel(int class_id) const;
};
