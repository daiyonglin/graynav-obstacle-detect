#pragma once

#include "common.hpp"

#include <vector>

namespace obstacle {

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
        float last_distance_m;
        float depth_m;
        float depth_velocity_mps;
        float depth_cov;
        bool has_depth_state;
        int non_person_hits;
        std::vector<int> raw_class_votes;

        Track();
    };

private:
    float MatchScore(const Track& track, const DetectionItem& det) const;
    bool CanStartTrack(const DetectionItem& det) const;
    void StartTrack(const DetectionItem& det, int frame_id);
    void UpdateTrack(Track* track, const DetectionItem& det, int frame_id);
    void AgeUnmatchedTracks(const std::vector<int>& matched_tracks, int frame_id);
    void RebuildStableResult();
    void MergeDisplayObstacles();
    void RebuildDecision();
    void UpdateTrackLabel(Track* track, const DetectionItem& det);
    std::string BestTrackRawLabel(const Track& track, int* raw_class_id, bool allow_person) const;

private:
    std::array<int, 2> image_shape_;
    int next_track_id_;
    std::vector<Track> tracks_;
    DetectionResult stable_result_;
    AvoidanceDecision decision_;
};

}  // namespace obstacle
