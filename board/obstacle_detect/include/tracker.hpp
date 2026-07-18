#pragma once

#include "avoidance_planner.hpp"
#include "common.hpp"
#include "ranging.hpp"

#include <vector>

namespace obstacle {

/**
 * @brief 双 ROI 多目标跟踪器，是“逐帧检测”与“稳定避障决策”之间的时序层。
 *
 * 检测器输出已经映射到 720x1280 全图坐标。本类在该统一坐标系中完成：
 * 1. 依据 IoU、中心距离、尺寸变化和类别兼容性关联检测与历史轨迹；
 * 2. 对框坐标和类别证据进行时序平滑，抑制框漂移与类别瞬时跳变；
 * 3. 调用 RangingEstimator 获得单帧距离，再以 alpha-beta 状态更新距离和径向速度；
 * 4. 根据当前 ROI 的可见范围管理轨迹寿命，避免上下 ROI 交替时误删目标；
 * 5. 将稳定目标交给 AvoidancePlanner，输出与 OSD、语音共用的唯一决策。
 */
class ObstacleTracker {
public:
    ObstacleTracker();

    void Initialize(const std::array<int, 2>& image_shape);
    void Update(const DetectionResult& raw_result, int frame_id);

    const DetectionResult& StableResult() const { return stable_result_; }
    const AvoidanceDecision& Decision() const { return decision_; }

private:
    /** 单个物体的完整时序状态；DetectionItem 保存对外结果，其余字段只服务于关联和滤波。 */
    struct Track {
        DetectionItem item;
        int id;
        int age;
        int hits;
        int missed;
        int last_frame;
        int64_t last_seen_ms;
        int64_t last_update_ms;
        bool matched_current_frame;
        bool visible_in_current_roi;
        float depth_m;
        float depth_velocity_mps;
        float depth_variance;
        int depth_measurements;
        float pending_far_depth_m;
        int pending_far_depth_count;
        int range_outlier_skips;
        std::vector<float> class_evidence;

        Track();
    };

    /** 全局候选匹配边；按 score 降序选择，保证一个检测只归属一条轨迹。 */
    struct MatchPair {
        int track;
        int detection;
        float score;
    };

    float MatchScore(const Track& track, const DetectionItem& detection) const;
    bool CanStartTrack(const DetectionItem& detection) const;
    bool IsVisibleInRoi(const Track& track, const std::array<int, 4>& roi) const;
    void StartTrack(const DetectionItem& detection, int frame_id, int64_t timestamp_ms);
    void UpdateTrack(Track* track, const DetectionItem& detection,
                     int frame_id, int64_t timestamp_ms);
    void UpdateClassEvidence(Track* track, const DetectionItem& detection);
    void UpdateRangeState(Track* track, const DetectionItem& detection, int64_t timestamp_ms);
    void AgeUnmatchedTracks(const std::vector<int>& matched_tracks,
                            const std::array<int, 4>& roi,
                            int frame_id,
                            int64_t timestamp_ms);
    void RebuildStableResult(const DetectionResult& raw_result, int64_t timestamp_ms);

    std::array<int, 2> image_shape_;
    int next_track_id_;
    std::vector<Track> tracks_;
    DetectionResult stable_result_;
    AvoidanceDecision decision_;
    RangingEstimator ranging_;
    AvoidancePlanner planner_;
};

}  // namespace obstacle
