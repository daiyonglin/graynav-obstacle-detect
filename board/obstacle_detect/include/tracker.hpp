#pragma once

#include "avoidance_planner.hpp"
#include "common.hpp"
#include "ranging.hpp"
#include "surface_fusion.hpp"

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
    void PredictOnly(int frame_id, int64_t timestamp_ms);
    void SetSurfaceResult(const SurfaceResult& surface) { latest_surface_ = surface; }

    const DetectionResult& StableResult() const { return stable_result_; }
    const AvoidanceDecision& Decision() const { return decision_; }
    const SurfaceResult& LatestSurfaceResult() const { return latest_surface_; }

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
        float pending_range_m;
        int pending_range_count;
        int range_outlier_skips;
        std::array<float, 7> inverse_depth_history;
        int inverse_depth_count;
        int inverse_depth_index;
        int last_view_id;
        int pending_class_id;
        int pending_class_count;
        std::vector<float> class_evidence;

        Track();
    };

    /** 全局候选匹配边；按 score 降序选择，保证一个检测只归属一条轨迹。 */
    struct MatchPair {
        int track;
        int detection;
        float score;
    };

    /** 计算轨迹与检测的 IoU、中心、尺度和类别综合关联分数，并拒绝突变。 */
    float MatchScore(const Track& track, const DetectionItem& detection) const;
    /** 判断局部肢体/躯干候选是否可维持一条已经存在的 person 轨迹。 */
    bool IsPersonPartBridge(const Track& track, const DetectionItem& detection) const;
    /** 应用类别专用阈值和几何质量门限，决定候选能否创建新轨迹。 */
    bool CanStartTrack(const DetectionItem& detection) const;
    /** 当前轨迹中心是否位于本帧推理 ROI，用于区分真正丢失与未观测。 */
    bool IsVisibleInRoi(const Track& track, const std::array<int, 4>& roi) const;
    /** 以当前测距和类别证据初始化一条尚未确认的新轨迹。 */
    void StartTrack(const DetectionItem& detection, int frame_id,
                    int64_t timestamp_ms, int view_id);
    /** 更新已匹配轨迹的框、分数、类别、距离、速度和生命周期。 */
    void UpdateTrack(Track* track, const DetectionItem& detection,
                     int frame_id, int64_t timestamp_ms, int view_id);
    /** 对 25 类证据做衰减累积和 1.2 倍切换滞回。 */
    void UpdateClassEvidence(Track* track, const DetectionItem& detection);
    /** 以真实时间差更新距离/径向速度，并计算可靠 TTC。 */
    void UpdateRangeState(Track* track, const DetectionItem& detection, int64_t timestamp_ms);
    /** 只老化当前 ROI 可见却未匹配的轨迹，并删除超时目标。 */
    void AgeUnmatchedTracks(const std::vector<int>& matched_tracks,
                            const std::array<int, 4>& roi,
                            int frame_id,
                            int64_t timestamp_ms);
    /** 发布通过连续命中和可见性检查的稳定目标，供规划和显示使用。 */
    void RebuildStableResult(const DetectionResult& raw_result, int64_t timestamp_ms);

    std::array<int, 2> image_shape_;
    int next_track_id_;
    std::vector<Track> tracks_;
    DetectionResult stable_result_;
    AvoidanceDecision decision_;
    RangingEstimator ranging_;
    DepthRangeFusion depth_fusion_;
    AvoidancePlanner planner_;
    SurfaceResult latest_surface_;
};

}  // namespace obstacle
