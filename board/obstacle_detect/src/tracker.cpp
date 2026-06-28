#include "../include/tracker.hpp"
#include "../include/utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace obstacle {
namespace {

constexpr int kDisplayPerson = DISPLAY_CLASS_PERSON;
constexpr int kRawClasses = 80;
constexpr int kMaxMissedNearFrames = 6;
constexpr int kMaxMissedFarFrames = 4;
constexpr int kMinConfirmedHits = 1;
constexpr int kMaxStableObjects = 8;
constexpr float kMatchMinScore = 0.36f;
constexpr float kHighIouCrossClass = 0.55f;
constexpr float kBoxEmaOld = 0.60f;
constexpr float kScoreEmaOld = 0.55f;
constexpr float kAssumedFps = 15.0f;

float clampf(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

float center_distance_norm(const std::array<float, 4>& a,
                           const std::array<float, 4>& b,
                           int img_w,
                           int img_h)
{
    const float acx = 0.5f * (a[0] + a[2]);
    const float acy = 0.5f * (a[1] + a[3]);
    const float bcx = 0.5f * (b[0] + b[2]);
    const float bcy = 0.5f * (b[1] + b[3]);
    const float dx = (acx - bcx) / std::max(1.0f, static_cast<float>(img_w));
    const float dy = (acy - bcy) / std::max(1.0f, static_cast<float>(img_h));
    return std::sqrt(dx * dx + dy * dy);
}

std::string risk_from_distance(float distance_m)
{
    if (distance_m < 0.0f) return "unknown";
    if (distance_m < 1.0f) return "near";
    if (distance_m < 2.0f) return "warning";
    return "far";
}

bool is_track_near_or_urgent(const DetectionItem& item)
{
    return item.risk_level == "urgent" || item.risk_level == "near" ||
           (item.distance_m >= 0.0f && item.distance_m < 1.0f);
}

int max_missed_for_track(const DetectionItem& item)
{
    return is_track_near_or_urgent(item) ? kMaxMissedNearFrames : kMaxMissedFarFrames;
}

std::string sector_from_box(const std::array<float, 4>& box, int img_w)
{
    const float w = std::max(1.0f, static_cast<float>(img_w));
    const float left_bound = 0.35f * w;
    const float right_bound = 0.65f * w;
    const float box_w = std::max(1.0f, box[2] - box[0]);

    if (box_w / w > 0.75f) {
        return "wide";
    }

    const float left_overlap =
        std::max(0.0f, std::min(box[2], left_bound) - std::max(box[0], 0.0f)) / box_w;
    const float center_overlap =
        std::max(0.0f, std::min(box[2], right_bound) - std::max(box[0], left_bound)) / box_w;
    const float right_overlap =
        std::max(0.0f, std::min(box[2], w) - std::max(box[0], right_bound)) / box_w;

    if (left_overlap >= 0.20f && center_overlap >= 0.20f && right_overlap >= 0.20f) {
        return "wide";
    }
    if (center_overlap >= 0.50f) {
        return "center";
    }
    if (center_overlap >= 0.25f && left_overlap >= 0.25f) return "left_center";
    if (center_overlap >= 0.25f && right_overlap >= 0.25f) return "center_right";

    if (left_overlap >= 0.50f) return "left";
    if (right_overlap >= 0.50f) return "right";

    const float cx = 0.5f * (box[0] + box[2]);
    if (cx < left_bound) return "left";
    if (cx > right_bound) return "right";
    return "center";
}

bool is_center_like_sector(const std::string& sector)
{
    return sector == "center" || sector == "wide" ||
           sector == "left_center" || sector == "center_right";
}

bool has_valid_distance(const DetectionItem& item)
{
    return item.distance_m >= 0.0f;
}

float track_box_width_ratio(const DetectionItem& item, int img_w)
{
    return std::max(0.0f, item.box[2] - item.box[0]) /
           std::max(1.0f, static_cast<float>(img_w));
}

float track_box_area_ratio(const DetectionItem& item, int img_w, int img_h)
{
    const float bw = std::max(0.0f, item.box[2] - item.box[0]);
    const float bh = std::max(0.0f, item.box[3] - item.box[1]);
    return (bw * bh) / std::max(1.0f, static_cast<float>(img_w * img_h));
}

bool box_center_inside(const std::array<float, 4>& inner,
                       const std::array<float, 4>& outer)
{
    const float cx = 0.5f * (inner[0] + inner[2]);
    const float cy = 0.5f * (inner[1] + inner[3]);
    return cx >= outer[0] && cx <= outer[2] && cy >= outer[1] && cy <= outer[3];
}

bool is_better_zone_item(const DetectionItem& a, const DetectionItem& b)
{
    const float a_weighted_score = a.score * std::max(0.40f, a.risk_weight);
    const float b_weighted_score = b.score * std::max(0.40f, b.risk_weight);
    if (a.quality != b.quality) {
        if (a.quality == "good") return true;
        if (b.quality == "good") return false;
        if (a.quality == "coarse") return false;
        if (b.quality == "coarse") return true;
    }
    if (has_valid_distance(a) && has_valid_distance(b)) {
        return a.distance_m < b.distance_m;
    }
    if (has_valid_distance(a) != has_valid_distance(b)) {
        return has_valid_distance(a);
    }
    return a_weighted_score > b_weighted_score;
}

void fill_zone(ZoneStatus* zone, const DetectionItem& item)
{
    zone->occupied = true;
    zone->label = item.label;
    zone->semantic_class = item.semantic_class;
    zone->risk_weight = item.risk_weight;
    zone->distance_m = item.distance_m;
    zone->risk_level = item.risk_level;
}

std::string distance_to_text(float distance_m)
{
    if (distance_m < 0.0f) return "unknown";

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1fm", distance_m);
    return std::string(buf);
}

}  // namespace

ObstacleTracker::Track::Track()
    : id(-1),
      age(0),
      hits(0),
      missed(0),
      last_frame(0),
      last_distance_m(-1.0f),
      depth_m(-1.0f),
      depth_velocity_mps(0.0f),
      depth_cov(1.0f),
      has_depth_state(false),
      non_person_hits(0),
      raw_class_votes(kRawClasses, 0) {}

ObstacleTracker::ObstacleTracker()
    : image_shape_{720, 1280},
      next_track_id_(1) {}

void ObstacleTracker::Initialize(const std::array<int, 2>& image_shape)
{
    image_shape_ = image_shape;
    next_track_id_ = 1;
    tracks_.clear();
    stable_result_.Clear();
    decision_ = AvoidanceDecision();
}

float ObstacleTracker::MatchScore(const Track& track, const DetectionItem& det) const
{
    const float overlap = utils::IoU(track.item.box, det.box);
    const float dist = center_distance_norm(track.item.box, det.box, image_shape_[0], image_shape_[1]);
    const float center_score = clampf(1.0f - dist / 0.35f, 0.0f, 1.0f);

    const bool same_display = track.item.class_id == det.class_id;
    const bool person_bridge = track.item.class_id == kDisplayPerson && overlap > kHighIouCrossClass;
    const bool obstacle_bridge = obstacle::semantic::IsObstacleClass(track.item.class_id) &&
                                 obstacle::semantic::IsObstacleClass(det.class_id) &&
                                 overlap > 0.25f;

    if (!same_display && !person_bridge && !obstacle_bridge) {
        return -1.0f;
    }

    return 0.70f * overlap + 0.30f * center_score;
}

bool ObstacleTracker::CanStartTrack(const DetectionItem& det) const
{
    const float area = track_box_area_ratio(det, image_shape_[0], image_shape_[1]);
    const float width_ratio = track_box_width_ratio(det, image_shape_[0]);

    if (det.class_id == kDisplayPerson) {
        return det.score >= 0.18f && width_ratio < 0.995f && area < 0.98f;
    }

    if (det.score >= 0.20f) {
        return width_ratio < 0.98f && area < 0.90f;
    }

    return false;
}

void ObstacleTracker::StartTrack(const DetectionItem& det, int frame_id)
{
    Track track;
    track.id = next_track_id_++;
    track.item = det;
    track.item.track_id = track.id;
    track.item.age = 1;
    track.item.missed = 0;
    track.age = 1;
    track.hits = 1;
    track.missed = 0;
    track.last_frame = frame_id;
    track.last_distance_m = det.distance_m;
    if (det.distance_m >= 0.0f) {
        track.depth_m = det.distance_m;
        track.depth_velocity_mps = 0.0f;
        track.depth_cov = 0.35f;
        track.has_depth_state = true;
    }
    UpdateTrackLabel(&track, det);
    tracks_.push_back(track);
}

void ObstacleTracker::UpdateTrackLabel(Track* track, const DetectionItem& det)
{
    if (det.raw_class_id >= 0 && det.raw_class_id < static_cast<int>(track->raw_class_votes.size())) {
        track->raw_class_votes[det.raw_class_id] += 1;
    }

    const bool display_person = track->item.class_id == kDisplayPerson;
    int raw_class_id = det.raw_class_id;
    const std::string raw_label = BestTrackRawLabel(*track, &raw_class_id, display_person);
    track->item.raw_class_id = raw_class_id;
    track->item.raw_label = raw_label;

    if (display_person) {
        track->item.label = "person";
        track->item.semantic_class = "person";
        track->item.risk_weight = obstacle::semantic::RiskWeight(kDisplayPerson);
    } else {
        const int semantic_id = obstacle::semantic::SemanticClassFromRaw(raw_class_id);
        track->item.class_id = semantic_id;
        track->item.label = obstacle::semantic::SemanticLabel(semantic_id);
        track->item.semantic_class = track->item.label;
        track->item.risk_weight = obstacle::semantic::RiskWeight(semantic_id);
    }
}

std::string ObstacleTracker::BestTrackRawLabel(const Track& track, int* raw_class_id, bool allow_person) const
{
    int best_id = -1;
    int best_count = 0;
    for (int i = 0; i < static_cast<int>(track.raw_class_votes.size()); ++i) {
        if (!allow_person && i == kDisplayPerson) {
            continue;
        }
        if (track.raw_class_votes[i] > best_count) {
            best_count = track.raw_class_votes[i];
            best_id = i;
        }
    }

    if (best_id >= 0 && best_count > 0) {
        *raw_class_id = best_id;
        return obstacle::semantic::RawLabel(best_id);
    }

    return "unknown";
}

void ObstacleTracker::UpdateTrack(Track* track, const DetectionItem& det, int frame_id)
{
    track->age += 1;
    track->hits += 1;
    track->missed = 0;

    const bool coarse_obstacle = obstacle::semantic::IsObstacleClass(det.class_id) && det.quality == "coarse";
    for (int i = 0; i < 4; ++i) {
        float old_v = track->item.box[i];
        float det_v = det.box[i];
        float ema_old = kBoxEmaOld;
        if (coarse_obstacle) {
            const bool expands_left_or_top = (i == 0 || i == 1) && det_v < old_v;
            const bool expands_right_or_bottom = (i == 2 || i == 3) && det_v > old_v;
            ema_old = (expands_left_or_top || expands_right_or_bottom) ? 0.88f : 0.55f;
        }
        track->item.box[i] = ema_old * old_v + (1.0f - ema_old) * det_v;
    }
    track->item.score = kScoreEmaOld * track->item.score +
                        (1.0f - kScoreEmaOld) * det.score;

    if (det.class_id == kDisplayPerson) {
        track->non_person_hits = 0;
        track->item.class_id = kDisplayPerson;
        track->item.label = "person";
        track->item.semantic_class = "person";
        track->item.risk_weight = obstacle::semantic::RiskWeight(kDisplayPerson);
    } else if (track->item.class_id == kDisplayPerson) {
        track->non_person_hits += 1;
        if (track->non_person_hits >= 5) {
            track->item.class_id = det.class_id;
            track->item.label = det.label;
            track->item.semantic_class = det.semantic_class;
            track->item.risk_weight = det.risk_weight;
        } else {
            track->item.class_id = kDisplayPerson;
            track->item.label = "person";
            track->item.semantic_class = "person";
            track->item.risk_weight = obstacle::semantic::RiskWeight(kDisplayPerson);
        }
    } else {
        track->item.class_id = det.class_id;
        track->item.label = det.label;
        track->item.semantic_class = det.semantic_class;
        track->item.risk_weight = det.risk_weight;
    }

    track->item.sector = sector_from_box(track->item.box, image_shape_[0]);
    track->item.distance_source = det.distance_source;

    track->item.distance_confidence = det.distance_confidence;
    track->item.quality = det.quality;

    const int frame_gap = std::max(1, frame_id - track->last_frame);
    const float dt = static_cast<float>(frame_gap) / kAssumedFps;
    if (det.distance_m >= 0.0f) {
        const float meas_conf = clampf(det.distance_confidence, 0.05f, 1.0f);
        if (!track->has_depth_state || track->depth_m < 0.0f) {
            track->depth_m = det.distance_m;
            track->depth_velocity_mps = 0.0f;
            track->depth_cov = 0.35f;
            track->has_depth_state = true;
            track->item.approach_mps = 0.0f;
        } else {
            const float prev_depth = track->depth_m;
            const float predicted_depth = track->depth_m + track->depth_velocity_mps * dt;
            const float predicted_cov = track->depth_cov + 0.025f + 0.04f * (1.0f - meas_conf);
            const float measurement_noise = 0.06f + 0.70f * (1.0f - meas_conf);
            const float k = predicted_cov / (predicted_cov + measurement_noise);
            const float residual = det.distance_m - predicted_depth;

            track->depth_m = clampf(predicted_depth + k * residual, 0.2f, 8.0f);
            const float measured_velocity = (track->depth_m - prev_depth) / std::max(0.01f, dt);
            track->depth_velocity_mps = 0.70f * track->depth_velocity_mps +
                                        0.30f * measured_velocity;
            track->depth_cov = (1.0f - k) * predicted_cov;
            track->item.approach_mps = std::max(0.0f, -track->depth_velocity_mps);
        }
        track->item.distance_m = track->depth_m;
    } else if (track->has_depth_state && track->missed == 0) {
        track->depth_m = clampf(track->depth_m + track->depth_velocity_mps * dt, 0.2f, 8.0f);
        track->depth_cov += 0.05f;
        track->item.distance_m = track->depth_m;
        track->item.approach_mps = std::max(0.0f, -track->depth_velocity_mps);
    }

    if (track->item.approach_mps > 0.05f && track->item.distance_m >= 0.0f) {
        track->item.ttc_s = track->item.distance_m / track->item.approach_mps;
    } else {
        track->item.ttc_s = -1.0f;
    }

    track->item.risk_level = risk_from_distance(track->item.distance_m);
    if (is_center_like_sector(track->item.sector) &&
        ((track->item.distance_m >= 0.0f && track->item.distance_m < 0.80f) ||
         (track->item.ttc_s > 0.0f && track->item.ttc_s < 1.50f))) {
        track->item.risk_level = "urgent";
    }

    track->item.track_id = track->id;
    track->item.age = track->age;
    track->item.missed = 0;
    track->last_frame = frame_id;
    track->last_distance_m = track->item.distance_m;

    UpdateTrackLabel(track, det);
}

void ObstacleTracker::AgeUnmatchedTracks(const std::vector<int>& matched_tracks, int frame_id)
{
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (std::find(matched_tracks.begin(), matched_tracks.end(), static_cast<int>(i)) != matched_tracks.end()) {
            continue;
        }

        tracks_[i].age += 1;
        tracks_[i].missed += 1;
        tracks_[i].item.age = tracks_[i].age;
        tracks_[i].item.missed = tracks_[i].missed;
        tracks_[i].last_frame = frame_id;
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const Track& t) {
                                     return t.missed > max_missed_for_track(t.item);
                                 }),
                  tracks_.end());
}

void ObstacleTracker::Update(const DetectionResult& raw_result, int frame_id)
{
    stable_result_.raw_candidate_count = raw_result.raw_candidate_count;
    stable_result_.post_nms_count = raw_result.post_nms_count;
    stable_result_.coarse_drop_count = raw_result.coarse_drop_count;

    std::vector<int> matched_tracks;
    std::vector<int> used_detections(raw_result.items.size(), 0);

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        float best_score = kMatchMinScore;
        int best_det = -1;

        for (size_t di = 0; di < raw_result.items.size(); ++di) {
            if (used_detections[di]) {
                continue;
            }

            const float score = MatchScore(tracks_[ti], raw_result.items[di]);
            if (score > best_score) {
                best_score = score;
                best_det = static_cast<int>(di);
            }
        }

        if (best_det >= 0) {
            UpdateTrack(&tracks_[ti], raw_result.items[best_det], frame_id);
            used_detections[best_det] = 1;
            matched_tracks.push_back(static_cast<int>(ti));
        }
    }

    AgeUnmatchedTracks(matched_tracks, frame_id);

    for (size_t di = 0; di < raw_result.items.size(); ++di) {
        if (!used_detections[di] && CanStartTrack(raw_result.items[di])) {
            StartTrack(raw_result.items[di], frame_id);
        }
    }

    RebuildStableResult();
    MergeDisplayObstacles();
    RebuildDecision();
}

void ObstacleTracker::RebuildStableResult()
{
    const int raw_count = stable_result_.raw_candidate_count;
    const int nms_count = stable_result_.post_nms_count;
    const int coarse_drop_count = stable_result_.coarse_drop_count;
    stable_result_.Clear();
    stable_result_.raw_candidate_count = raw_count;
    stable_result_.post_nms_count = nms_count;
    stable_result_.coarse_drop_count = coarse_drop_count;

    for (size_t i = 0; i < tracks_.size(); ++i) {
        const Track& track = tracks_[i];
        if (track.hits < kMinConfirmedHits && track.item.score < 0.60f) {
            continue;
        }

        DetectionItem item = track.item;
        if (track.missed > 0) {
            item.score *= 0.85f;
        }
        stable_result_.items.push_back(item);
    }

    if (stable_result_.items.size() > 1) {
        std::vector<DetectionItem> filtered;
        filtered.reserve(stable_result_.items.size());
        for (size_t i = 0; i < stable_result_.items.size(); ++i) {
            const DetectionItem& candidate = stable_result_.items[i];
            bool suppress = false;
            if (obstacle::semantic::IsObstacleClass(candidate.class_id) && candidate.quality == "coarse") {
                for (size_t j = 0; j < stable_result_.items.size(); ++j) {
                    if (i == j) continue;
                    const DetectionItem& fine = stable_result_.items[j];
                    if (fine.quality != "coarse" &&
                        fine.score >= candidate.score * 0.55f &&
                        box_center_inside(fine.box, candidate.box)) {
                        suppress = true;
                        break;
                    }
                }
            }
            if (!suppress) {
                filtered.push_back(candidate);
            }
        }
        stable_result_.items.swap(filtered);
    }

    utils::SortDetectionResult(&stable_result_);
    if (static_cast<int>(stable_result_.items.size()) > kMaxStableObjects) {
        stable_result_.items.resize(kMaxStableObjects);
    }
}

void ObstacleTracker::MergeDisplayObstacles()
{
    if (stable_result_.items.size() < 2) {
        return;
    }

    std::vector<int> suppressed(stable_result_.items.size(), 0);
    std::vector<DetectionItem> merged;
    merged.reserve(stable_result_.items.size());

    for (size_t i = 0; i < stable_result_.items.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        DetectionItem cur = stable_result_.items[i];
        for (size_t j = i + 1; j < stable_result_.items.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            if (!obstacle::semantic::IsObstacleClass(cur.class_id) ||
                !obstacle::semantic::IsObstacleClass(stable_result_.items[j].class_id) ||
                cur.class_id != stable_result_.items[j].class_id) {
                continue;
            }

            if (utils::IoU(cur.box, stable_result_.items[j].box) > 0.85f) {
                const DetectionItem& other = stable_result_.items[j];
                if (is_better_zone_item(other, cur)) {
                    cur = other;
                }
                suppressed[j] = 1;
            }
        }

        cur.sector = sector_from_box(cur.box, image_shape_[0]);
        merged.push_back(cur);
    }

    stable_result_.items.swap(merged);
    utils::SortDetectionResult(&stable_result_);
    if (static_cast<int>(stable_result_.items.size()) > kMaxStableObjects) {
        stable_result_.items.resize(kMaxStableObjects);
    }
}

void ObstacleTracker::RebuildDecision()
{
    decision_ = AvoidanceDecision();

    int nearest_idx = -1;

    auto update_zone = [](ZoneStatus* zone, const DetectionItem& item) {
        if (!zone->occupied) {
            fill_zone(zone, item);
            return;
        }

        DetectionItem prev;
        prev.label = zone->label;
        prev.distance_m = zone->distance_m;
        prev.risk_level = zone->risk_level;
        prev.score = 0.0f;
        if (is_better_zone_item(item, prev)) {
            fill_zone(zone, item);
        }
    };

    for (size_t i = 0; i < stable_result_.items.size(); ++i) {
        const DetectionItem& item = stable_result_.items[i];

        if (item.sector == "left" || item.sector == "left_center" || item.sector == "wide") {
            update_zone(&decision_.left, item);
        }
        if (is_center_like_sector(item.sector)) {
            update_zone(&decision_.center, item);
        }
        if (item.sector == "right" || item.sector == "center_right" || item.sector == "wide") {
            update_zone(&decision_.right, item);
        }

        if (nearest_idx < 0 || is_better_zone_item(item, stable_result_.items[nearest_idx])) {
            nearest_idx = static_cast<int>(i);
        }
    }

    if (nearest_idx >= 0) {
        decision_.nearest_track_id = stable_result_.items[nearest_idx].track_id;
    }

    const auto is_urgent_or_near = [](const ZoneStatus& zone) {
        return zone.occupied &&
               (zone.risk_level == "urgent" ||
                zone.risk_level == "near" ||
                (zone.distance_m >= 0.0f && zone.distance_m < 1.0f));
    };

    const auto is_warning = [](const ZoneStatus& zone) {
        return zone.occupied &&
               (zone.risk_level == "warning" ||
                (zone.distance_m >= 1.0f && zone.distance_m < 2.0f));
    };

    const auto clearance = [](const ZoneStatus& zone) {
        if (!zone.occupied) return 8.0f;
        if (zone.distance_m < 0.0f) return 2.5f;
        return zone.distance_m;
    };

    const bool left_near = is_urgent_or_near(decision_.left);
    const bool center_near = is_urgent_or_near(decision_.center);
    const bool right_near = is_urgent_or_near(decision_.right);
    const bool all_near = left_near && center_near && right_near;
    bool wide_near = false;
    for (size_t i = 0; i < stable_result_.items.size(); ++i) {
        const DetectionItem& item = stable_result_.items[i];
        if (item.sector == "wide" && is_track_near_or_urgent(item)) {
            wide_near = true;
            break;
        }
    }

    if (all_near || wide_near) {
        const DetectionItem& nearest = stable_result_.items[nearest_idx];
        decision_.action = "stop";
        decision_.prompt = "reason=all_blocked " + nearest.sector + " " +
                           nearest.label + " " + distance_to_text(nearest.distance_m);
    } else if (center_near) {
        const float left_clearance = clearance(decision_.left);
        const float right_clearance = clearance(decision_.right);
        if (left_clearance >= right_clearance) {
            decision_.action = "turn_left";
            decision_.prompt = "reason=center_blocked_left_clear center " +
                               distance_to_text(decision_.center.distance_m);
        } else {
            decision_.action = "turn_right";
            decision_.prompt = "reason=center_blocked_right_clear center " +
                               distance_to_text(decision_.center.distance_m);
        }
    } else if (decision_.center.occupied && is_warning(decision_.center)) {
        decision_.action = "slow";
        decision_.prompt = "reason=center_warning center " +
                           distance_to_text(decision_.center.distance_m);
    } else if (left_near && !center_near) {
        decision_.action = "turn_right";
        decision_.prompt = "reason=left_blocked_center_clear left " +
                           distance_to_text(decision_.left.distance_m);
    } else if (right_near && !center_near) {
        decision_.action = "turn_left";
        decision_.prompt = "reason=right_blocked_center_clear right " +
                           distance_to_text(decision_.right.distance_m);
    } else if (is_warning(decision_.left) || is_warning(decision_.right)) {
        const DetectionItem& nearest = stable_result_.items[nearest_idx];
        decision_.action = "slow";
        decision_.prompt = "reason=side_warning " + nearest.sector + " " +
                           nearest.label + " " + distance_to_text(nearest.distance_m);
    } else if (nearest_idx >= 0) {
        const DetectionItem& nearest = stable_result_.items[nearest_idx];
        decision_.action = "clear";
        decision_.prompt = "reason=far_or_clear nearest " + nearest.sector + " " + nearest.label + " " +
                           distance_to_text(nearest.distance_m);
    } else {
        decision_.action = "clear";
        decision_.prompt = "clear";
    }
}

}  // namespace obstacle
