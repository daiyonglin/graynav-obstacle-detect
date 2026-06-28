#pragma once

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <array>
#include <string>
#include <vector>
#include <algorithm>

#include "smartsoc/ssne_api.h"
#include "semantic_config.hpp"

/**
 * @brief 单个检测框结果
 *
 * box: [xmin, ymin, xmax, ymax]，坐标系统一为“原图坐标”
 * score: 分类置信度
 * class_id: COCO 类别 id
 * label: 人类可读标签
 */
enum DisplayClass {
    DISPLAY_CLASS_PERSON = obstacle::semantic::PERSON,
    DISPLAY_CLASS_OBSTACLE = obstacle::semantic::GENERIC_OBSTACLE
};

struct DetectionItem {
    std::array<float, 4> box;
    float score;
    int class_id;        // Obstacle semantic class id, see semantic_config.hpp.
    int raw_class_id;    // Original model class id.
    std::string label;   // Stable semantic label for user-facing output.
    std::string semantic_class;
    std::string raw_label;
    float risk_weight;
    std::string sector;  // left, center, right, left_center, center_right, wide.
    float distance_m;
    std::string distance_source;  // ground, size, fused, unknown.
    float distance_confidence;
    std::string risk_level;  // near, warning, far, unknown.
    std::string quality;  // good, low, coarse.
    int track_id;
    int age;
    int missed;
    float approach_mps;
    float ttc_s;

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
          distance_source("unknown"),
          distance_confidence(0.0f),
          risk_level("unknown"),
          quality("low"),
          track_id(-1),
          age(0),
          missed(0),
          approach_mps(0.0f),
          ttc_s(-1.0f) {}
};

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

struct AvoidanceDecision {
    ZoneStatus left;
    ZoneStatus center;
    ZoneStatus right;
    std::string action;
    std::string prompt;
    int nearest_track_id;

    AvoidanceDecision()
        : action("clear"),
          prompt("clear"),
          nearest_track_id(-1) {
        left.dir = "left";
        center.dir = "center";
        right.dir = "right";
    }
};

/**
 * @brief 一帧检测结果
 */
struct DetectionResult {
    std::vector<DetectionItem> items;
    int raw_candidate_count;
    int post_nms_count;
    int coarse_drop_count;

    DetectionResult()
        : raw_candidate_count(0),
          post_nms_count(0),
          coarse_drop_count(0) {}

    void Clear() {
        items.clear();
        raw_candidate_count = 0;
        post_nms_count = 0;
        coarse_drop_count = 0;
    }

    void Free() {
        std::vector<DetectionItem>().swap(items);
        raw_candidate_count = 0;
        post_nms_count = 0;
        coarse_drop_count = 0;
    }

    size_t Size() const {
        return items.size();
    }
};

/**
 * @brief letterbox 信息
 */
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

/**
 * @brief 图像获取模块
 *
 * 第一版职责：
 * 1. 从 sensor 获取整幅原始灰度图
 * 2. 不做 ROI 裁剪
 * 3. 不做 resize / normalize
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
 *   原始整幅灰度图
 *
 * 内部流程：
 *   1. AI preprocess pipe:
 *      - 全图 crop（其实就是不裁）
 *      - letterbox 到 384x384
 *      - normalize（由 SetNormalize 从 .m1model 读取）
 *   2. NPU 推理：
 *      - yolov8n_head6.m1model，输出 6 个 raw branch
 *   3. CPU 后处理：
 *      - 自动识别输出分支顺序
 *      - 自动在 CHW/HWC 两种读取方式中选择更合理的一种
 *      - sigmoid + DFL + bbox decode + NMS
 *      - 反 letterbox 映射到原图坐标
 */
class YOLOV8GRAY {
public:
    std::string ModelName() const { return "yolov8_gray_head6"; }

    void Initialize(std::string& model_path,
                    std::array<int, 2>* in_img_shape,
                    std::array<int, 2>* in_det_shape);

    void Predict(ssne_tensor_t* img_in,
                 DetectionResult* result,
                 float conf_threshold = 0.35f);

    void Release();

    LetterboxInfo GetLastLetterboxInfo() const { return lb_info_; }

public:
    float nms_threshold = 0.60f;
    int top_k = 800;
    int keep_top_k = 80;

    std::array<int, 2> img_shape;   // [w, h]
    std::array<int, 2> det_shape;   // [w, h]

private:
    uint16_t model_id = 0;
    ssne_tensor_t inputs[1];
    ssne_tensor_t outputs[6];

    AiPreprocessPipe pipe_offline;
    LetterboxInfo lb_info_;

    std::vector<std::string> class_names_;

private:
    void BuildClassNames();

    void Preprocess(ssne_tensor_t* img_in, ssne_tensor_t* input_tensor);

    void Postprocess(DetectionResult* result, float conf_threshold);

    void MapBoxToOriginalImage(std::array<float, 4>& box);

    std::string ClassIdToLabel(int class_id) const;
};
