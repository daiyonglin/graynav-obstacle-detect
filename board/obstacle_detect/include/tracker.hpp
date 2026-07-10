#pragma once

#include "avoidance_planner.hpp"
#include "common.hpp"
#include "ranging.hpp"

#include <vector>

namespace obstacle {

/**
 * Associates dual-ROI detections in full-frame coordinates, applies
 * motion-adaptive box smoothing and maintains a real-time range state for
 * every obstacle.  The tracker deliberately separates visual confirmation
 * from action planning.
 */
class ObstacleTracker {
public:
    ObstacleTracker();

    void Initialize(const std::array<int, 2>& image_shape);
    void Update(const DetectionResult& raw_result, int frame_id);

    const DetectionResult& StableResult() const { return stable_result_; }
    const AvoidanceDecision& Decision() const { return decision_; }

private:
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
        std::vector<float> class_evidence;

        Track();
    };

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
