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
 * @brief 閸楁洑閲滃Λ鈧ù瀣攱缂佹挻鐏? *
 * box: [xmin, ymin, xmax, ymax]閿涘苯娼楅弽鍥╅兇缂佺喍绔存稉琛♀偓婊冨斧閸ユ儳娼楅弽鍥ｂ偓? * score: 閸掑棛琚純顔讳繆鎼? * class_id: COCO 缁鍩?id
 * label: 娴滆櫣琚崣顖濐嚢閺嶅洨顒? */
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
    float safe_distance_m;
    float distance_sigma_m;
    float lateral_m;
    std::string distance_source;  // ground, size, fused, unknown.
    float distance_confidence;
    std::string risk_level;  // near, warning, far, unknown.
    std::string quality;  // good, low, coarse.
    int track_id;
    int age;
    int missed;
    float approach_mps;
    float ttc_s;
    int range_measurements;

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
          range_measurements(0) {}
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
 * @brief 娑撯偓鐢勵梾濞村绮ㄩ弸? */
struct DetectionResult {
    std::vector<DetectionItem> items;
    int raw_candidate_count;
    int post_nms_count;
    int coarse_drop_count;
    int view_id;
    std::array<int, 4> roi;  // x1, y1, x2, y2 in full-frame coordinates.
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

/**
 * @brief letterbox 娣団剝浼? */
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

/** Per-frame detector stage timings used by competition performance logs. */
struct DetectorTiming {
    float preprocess_ms;
    float inference_ms;
    float output_ms;
    float postprocess_ms;
    DetectorTiming()
        : preprocess_ms(0.0f), inference_ms(0.0f), output_ms(0.0f), postprocess_ms(0.0f) {}
};

/**
 * @brief 閸ユ儳鍎氶懢宄板絿濡€虫健
 *
 * 缁楊兛绔撮悧鍫ｄ捍鐠愶綇绱? * 1. 娴?sensor 閼惧嘲褰囬弫鏉戠畽閸樼喎顫愰悘鏉垮閸? * 2. 娑撳秴浠?ROI 鐟佷礁澹€
 * 3. 娑撳秴浠?resize / normalize
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
 * @brief YOLOv8 head6 闂呮粎顣插Λ鈧ù瀣珤
 *
 * 鏉堟挸鍙嗛敍? *   閸樼喎顫愰弫鏉戠畽閻忔澘瀹抽崶? *
 * 閸愬懘鍎村ù浣衡柤閿? *   1. AI preprocess pipe:
 *      - 閸忋劌娴?crop閿涘牆鍙剧€圭偛姘ㄩ弰顖欑瑝鐟佷緤绱? *      - letterbox 閸?384x384
 *      - normalize閿涘牏鏁?SetNormalize 娴?.m1model 鐠囪褰囬敍? *   2. NPU 閹恒劎鎮婇敍? *      - yolov8n_head6.m1model閿涘矁绶崙?6 娑?raw branch
 *   3. CPU 閸氬骸顦╅悶鍡窗
 *      - 閼奉亜濮╃拠鍡楀焼鏉堟挸鍤崚鍡樻暜妞ゅ搫绨? *      - 閼奉亜濮╅崷?CHW/HWC 娑撱倗顫掔拠璇插絿閺傜懓绱℃稉顓⑩偓澶嬪閺囨潙鎮庨悶鍡欐畱娑撯偓缁? *      - sigmoid + DFL + bbox decode + NMS
 *      - 閸?letterbox 閺勭姴鐨犻崚鏉垮斧閸ユ儳娼楅弽? */
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
    ssne_tensor_t inputs[1];
    ssne_tensor_t outputs[6];

    AiPreprocessPipe pipe_offline_[2] = {nullptr, nullptr};
    LetterboxInfo lb_info_[2];
    std::array<int, 4> roi_[2];
    int active_view_ = 0;
    int predict_count_ = 0;
    bool dual_roi_ = true;
    DetectorTiming last_timing_;

    std::vector<std::string> class_names_;

private:
    void BuildClassNames();

    bool Preprocess(ssne_tensor_t* img_in, ssne_tensor_t* input_tensor);

    bool Postprocess(DetectionResult* result, float conf_threshold);

    void MapBoxToOriginalImage(std::array<float, 4>& box);

    std::string ClassIdToLabel(int class_id) const;
};
