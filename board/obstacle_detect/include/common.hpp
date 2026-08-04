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
    int raw_class_id;          // ROD25 检测头的原始类别编号。
    std::string label;         // 对外显示的稳定语义名称。
    std::string semantic_class;// 与 label 同域，供 JSON/串口接口使用。
    std::string raw_label;     // ROD25 原始类别名，便于诊断误分类。
    float risk_weight;         // 类别风险权重，参与候选排序和规划。
    std::string sector;        // left/center/right/交叠区/wide。
    float distance_m;          // 融合后的期望距离；负值表示无可靠米级结果。
    float safe_distance_m;     // mean-sigma 的保守距离，避障决策优先使用它。
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
    float ttc_s;               // 碰撞时间；证据不足时为负值。
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
    std::string semantic_class;
    float risk_weight;
    float distance_m;
    std::string risk_level;

    ZoneStatus()
        : dir("unknown"),
          occupied(false),
          label(""),
          semantic_class(""),
          risk_weight(1.0f),
          distance_m(-1.0f),
          risk_level("unknown") {}
};

/** @brief 单通道道路分割模型的三类部署契约。 */
enum SurfaceClass {
    GROUND_CANDIDATE = 0,
    BLOCKED_SURFACE = 1,
    STEP_OR_DROP = 2,
    SURFACE_CLASS_COUNT = 3
};

static const int SURFACE_GRID_SIZE = 64;
static const int SURFACE_GRID_CELLS = SURFACE_GRID_SIZE * SURFACE_GRID_SIZE;
static const int DEPTH_BIN_COUNT = 16;

/** @brief 分割网格投影到单条导航走廊后的面积比例和稳定风险。 */
struct SurfaceCorridor {
    float ground_ratio;
    float blocked_ratio;
    float step_ratio;
    bool safe_candidate;
    bool persistent_hazard;

    SurfaceCorridor()
        : ground_ratio(0.0f),
          blocked_ratio(0.0f),
          step_ratio(0.0f),
          safe_candidate(false),
          persistent_hazard(false) {}
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
    std::string depth_source;
    bool depth_consistent;
    bool approaching;
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
          depth_source("unknown"),
          depth_consistent(false),
          approaching(false),
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
    std::string depth_source;
    bool depth_consistent;
    bool approaching;

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
          depth_source("unknown"),
          depth_consistent(false),
          approaching(false) {
        left.dir = "left";
        center.dir = "center";
        right.dir = "right";
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
 * @brief YOLOv8 head6 障碍检测器
 *
 * 输入：
 *   SC132GS 原始整幅灰度图，当前模型编译为真实 1 通道输入。
 *
 * 内部流程：
 *   1. AI preprocess pipe:
 *      - 奇偶帧选择上/下重叠 ROI；
 *      - 等比例 letterbox 到 384x384；
 *      - 需要时执行自适应灰度 LUT；
 *      - normalize 参数由 .m1model 的输入量化配置提供。
 *   2. NPU 推理：
 *      - B3 Gray1-DCE m1model，输出 6 个 raw branch。
 *   3. CPU 后处理：
 *      - 自动识别输出分支顺序
 *      - 校验 3 个 25 通道分类头和 3 个 64 通道 DFL 回归头；
 *      - 依据 runtime layout 以 HWC/CHW 正确读取量化输出；
 *      - top-1 分类、sigmoid、DFL、anchor 解码和多目标 NMS；
 *      - 过滤饱和横框/粗框并映射回完整 Aurora 坐标。
 */
class YOLOV8GRAY {
public:
    std::string ModelName() const { return "yolov8_gray_head6"; }

    void Initialize(std::string& model_path,
                    std::array<int, 2>* in_img_shape,
                    std::array<int, 2>* in_det_shape);

    bool Predict(ssne_tensor_t* img_in,
                 DetectionResult* result,
                 float conf_threshold = 0.35f);

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
    ssne_tensor_t outputs[6] = {};

    AiPreprocessPipe pipe_offline_[2] = {nullptr, nullptr};
    LetterboxInfo lb_info_[2];
    std::array<int, 4> roi_[2];
    int active_view_ = 0;
    int predict_count_ = 0;
    bool dual_roi_ = true;
    DetectorTiming last_timing_;

    std::vector<std::string> class_names_;

private:
    /** 从模型类别契约建立原始标签表。 */
    void BuildClassNames();

    /** 选择当前 ROI，并完成 crop、letterbox、normalize 和可选暗光增强。 */
    bool Preprocess(ssne_tensor_t* img_in, ssne_tensor_t* input_tensor);

    /** 校验并解码 6 个 raw head，输出已经反映射到全图坐标的检测结果。 */
    bool Postprocess(DetectionResult* result, float conf_threshold);

    /** 将 384 模型坐标去 padding/scale 后加 ROI 原点，恢复为 Aurora 坐标。 */
    void MapBoxToOriginalImage(std::array<float, 4>& box);

    /** 返回 ROD25 原始类别名；越界编号返回 unknown。 */
    std::string ClassIdToLabel(int class_id) const;
};
