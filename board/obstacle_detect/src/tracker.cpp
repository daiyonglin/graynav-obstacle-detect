#include "../include/tracker.hpp"

#include "../include/semantic_config.hpp"
#include "../include/utils.hpp"

#include <algorithm>
#include <cmath>

/*
 * 跟踪器处理的所有框均为传感器全图坐标。关联、框平滑、类别证据、距离状态
 * 和轨迹生命期在此集中维护，避免检测、OSD 和避障模块各自保存不一致状态。
 */
namespace obstacle {
namespace {

// 这些阈值控制“何时显示”和“最多保留多少目标”，不改变模型检测阈值。
const int kMinConfirmedHits = 2;
const int kMaxStableObjects = 8;
const float kMatchMinimum = 0.34f;

float clampf(float value, float lo, float hi)
{
    return std::max(lo, std::min(hi, value));
}

float box_width(const std::array<float, 4>& box)
{
    return std::max(1.0f, box[2] - box[0]);
}

float box_height(const std::array<float, 4>& box)
{
    return std::max(1.0f, box[3] - box[1]);
}

float center_distance(const std::array<float, 4>& a,
                      const std::array<float, 4>& b,
                      const std::array<int, 2>& image_shape)
{
    const float dx = (0.5f * (a[0] + a[2]) - 0.5f * (b[0] + b[2])) /
                     std::max(1.0f, static_cast<float>(image_shape[0]));
    const float dy = (0.5f * (a[1] + a[3]) - 0.5f * (b[1] + b[3])) /
                     std::max(1.0f, static_cast<float>(image_shape[1]));
    return std::sqrt(dx * dx + dy * dy);
}

float size_similarity(const std::array<float, 4>& a,
                      const std::array<float, 4>& b)
{
    const float wr = std::min(box_width(a), box_width(b)) /
                     std::max(box_width(a), box_width(b));
    const float hr = std::min(box_height(a), box_height(b)) /
                     std::max(box_height(a), box_height(b));
    return std::sqrt(wr * hr);
}

float containment_ratio(const std::array<float, 4>& a,
                        const std::array<float, 4>& b)
{
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float intersection = std::max(0.0f, x2 - x1) *
                               std::max(0.0f, y2 - y1);
    const float area_a = box_width(a) * box_height(a);
    const float area_b = box_width(b) * box_height(b);
    return intersection / std::max(1.0f, std::min(area_a, area_b));
}

bool implausibly_broad_box(const DetectionItem& item,
                           const std::array<int, 2>& image_shape)
{
    const float wr = box_width(item.box) /
        std::max(1.0f, static_cast<float>(image_shape[0]));
    const float hr = box_height(item.box) /
        std::max(1.0f, static_cast<float>(image_shape[1]));
    return wr > 0.94f || (wr > 0.82f && hr > 0.45f);
}

std::string sector_from_box(const std::array<float, 4>& box, int width)
{
    const float cx = 0.5f * (box[0] + box[2]);
    const float bw = box_width(box);
    const float frame_width = std::max(1.0f, static_cast<float>(width));
    if (bw / frame_width > semantic::WideBoxRatio()) return "wide";
    if (cx < semantic::SectorLeftBoundaryRatio() * frame_width) return "left";
    if (cx > semantic::SectorRightBoundaryRatio() * frame_width) return "right";
    return "center";
}

std::string risk_from_safe(float safe_distance, float ttc)
{
    if (ttc > 0.0f && ttc < semantic::StopTtcSeconds()) return "urgent";
    if (safe_distance < 0.0f) return "unknown";
    if (safe_distance < semantic::UrgentDistanceM()) return "urgent";
    if (safe_distance < semantic::NearDistanceM()) return "near";
    if (safe_distance < semantic::WarningDistanceM()) return "warning";
    return "far";
}

bool better_detection(const DetectionItem& a, const DetectionItem& b)
{
    const int ar = a.risk_level == "urgent" ? 4 :
                   a.risk_level == "near" ? 3 :
                   a.risk_level == "warning" ? 2 : 1;
    const int br = b.risk_level == "urgent" ? 4 :
                   b.risk_level == "near" ? 3 :
                   b.risk_level == "warning" ? 2 : 1;
    if (ar != br) return ar > br;
    const float ad = a.safe_distance_m >= 0.0f ? a.safe_distance_m : 99.0f;
    const float bd = b.safe_distance_m >= 0.0f ? b.safe_distance_m : 99.0f;
    if (std::fabs(ad - bd) > 0.1f) return ad < bd;
    const int aq = a.quality == "good" ? 2 : (a.quality == "low" ? 1 : 0);
    const int bq = b.quality == "good" ? 2 : (b.quality == "low" ? 1 : 0);
    if (aq != bq) return aq > bq;
    return a.score * a.risk_weight > b.score * b.risk_weight;
}

}  // namespace

ObstacleTracker::Track::Track()
    : id(-1),
      age(0),
      hits(0),
      missed(0),
      last_frame(0),
      last_seen_ms(0),
      last_update_ms(0),
      matched_current_frame(false),
      visible_in_current_roi(false),
      depth_m(-1.0f),
      depth_velocity_mps(0.0f),
      depth_variance(1.0f),
      depth_measurements(0),
      pending_range_m(-1.0f),
      pending_range_count(0),
      range_outlier_skips(0),
      inverse_depth_history{0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
      inverse_depth_count(0),
      inverse_depth_index(0),
      last_view_id(-1),
      pending_class_id(-1),
      pending_class_count(0),
      class_evidence(std::max(1, semantic::ModelClassCount()), 0.0f)
{
}

ObstacleTracker::ObstacleTracker()
    : image_shape_{720, 1280}, next_track_id_(1)
{
}

void ObstacleTracker::Initialize(const std::array<int, 2>& image_shape)
{
    image_shape_ = image_shape;
    next_track_id_ = 1;
    tracks_.clear();
    stable_result_.Clear();
    decision_ = AvoidanceDecision();
    ranging_.Initialize(image_shape);
    depth_fusion_.Initialize(image_shape);
    planner_.Initialize(image_shape);
    latest_surface_ = SurfaceResult();
}

float ObstacleTracker::MatchScore(const Track& track,
                                  const DetectionItem& detection) const
{
    // 先用创新门限剔除不可能的跳变，再计算加权关联分数；这一步是抑制漂移的核心。
    const float overlap = utils::IoU(track.item.box, detection.box);
    const float distance = center_distance(track.item.box, detection.box, image_shape_);
    const float shape_score = size_similarity(track.item.box, detection.box);
    const float containment = containment_ratio(track.item.box, detection.box);
    // 平滑前先拒绝位置或尺度突变。否则单帧伪框会把稳定轨迹拉到画面另一处，
    // 在 Aurora 上表现为检测框漂移或凭空横移。
    const bool person_part_bridge = IsPersonPartBridge(track, detection);
    const bool indoor_class_mismatch = semantic::ModelClassCount() == 8 &&
        track.item.raw_class_id >= 0 && detection.raw_class_id >= 0 &&
        track.item.raw_class_id != detection.raw_class_id;
    if (indoor_class_mismatch &&
        (distance >= 0.08f || (overlap < 0.55f && containment < 0.75f))) {
        return 0.0f;
    }
    if (!person_part_bridge && ((overlap < 0.02f && distance > 0.14f) ||
        (distance > 0.10f && overlap < 0.20f) ||
        (shape_score < 0.38f && overlap < 0.30f) ||
        (track.item.quality != "coarse" && detection.quality == "coarse" && overlap < 0.55f))) {
        return 0.0f;
    }
    const float center_score = clampf(1.0f - distance / 0.28f, 0.0f, 1.0f);
    float class_score = track.item.class_id == detection.class_id ? 1.0f : 0.45f;
    if (person_part_bridge) {
        class_score = 0.60f;
    } else if ((track.item.class_id == semantic::PERSON) !=
        (detection.class_id == semantic::PERSON)) {
        class_score = overlap > 0.60f ? 0.35f : 0.05f;
    }
    const float score = 0.52f * overlap + 0.28f * center_score +
                        0.12f * shape_score + 0.08f * class_score;
    // 局部肢体与原全身框的纵向 IoU 可能很低；桥接条件已经验证横向连续性，
    // 因此给予刚超过关联门槛的下限，让其能够维持原 person 轨迹。
    return person_part_bridge ? std::max(score, 0.35f) : score;
}

bool ObstacleTracker::IsPersonPartBridge(const Track& track,
                                         const DetectionItem& detection) const
{
    /*
     * 人体从全身变成腿部/手部可见时，模型可能暂时输出其他实体障碍类别。
     * 只有已有 person 轨迹、候选非 coarse、横向位置连续且空间邻近时才桥接；
     * 该规则不能从零创造人体，只负责维持已有人的局部可见轨迹。
     */
    // Indoor8 has an explicit furniture/people classifier.  Rewriting a real
    // CHAIR/TABLE observation to PERSON creates a permanent PERSON lock, so the
    // historical partial-body bridge is restricted to legacy ROD25 only.
    if (semantic::ModelClassCount() != 25 ||
        track.item.class_id != semantic::PERSON ||
        detection.class_id == semantic::PERSON ||
        detection.quality == "coarse" || detection.score < 0.12f) {
        return false;
    }
    const float tcx = 0.5f * (track.item.box[0] + track.item.box[2]);
    const float tcy = 0.5f * (track.item.box[1] + track.item.box[3]);
    const float dcx = 0.5f * (detection.box[0] + detection.box[2]);
    const float dcy = 0.5f * (detection.box[1] + detection.box[3]);
    const float dx = std::fabs(tcx - dcx) /
        std::max(1.0f, static_cast<float>(image_shape_[0]));
    const float dy = std::fabs(tcy - dcy) /
        std::max(1.0f, static_cast<float>(image_shape_[1]));
    const float horizontal_overlap = std::max(0.0f,
        std::min(track.item.box[2], detection.box[2]) -
        std::max(track.item.box[0], detection.box[0]));
    const float overlap_ratio = horizontal_overlap /
        std::max(1.0f, std::min(box_width(track.item.box), box_width(detection.box)));
    return dx < 0.10f && dy < 0.24f && overlap_ratio > 0.30f;
}

bool ObstacleTracker::CanStartTrack(const DetectionItem& detection) const
{
    if (implausibly_broad_box(detection, image_shape_)) return false;
    if (detection.quality == "coarse") return false;
    if (detection.score >= 0.45f) return true;
    if (detection.class_id == semantic::PERSON) return detection.score >= 0.12f;
    if (semantic::ModelClassCount() == 8) {
        if (semantic::IsFurnitureLikeSemantic(detection.class_id)) {
            return detection.score >= 0.18f;
        }
        return detection.score >= 0.20f;
    }
    if (semantic::ModelClassCount() == 25) {
        // 与室内 ROD25 解码阈值衔接。弱候选仍需连续命中才允许输出，
        // 因此一次量化噪声不会立即生成轨迹并触发避障。
        if (detection.raw_class_id == 17 || detection.raw_class_id == 23) {
            return detection.score >= 0.16f;  // bench / chair
        }
        if (detection.raw_class_id == 9 || detection.raw_class_id == 22) {
            return detection.score >= 0.18f;  // dustbin / electrical box
        }
        if (detection.raw_class_id == 20 || detection.raw_class_id == 24) {
            return detection.score >= 0.20f;  // barrel / rigid rack
        }
    }
    return detection.score >= 0.22f;
}

bool ObstacleTracker::IsVisibleInRoi(const Track& track,
                                     const std::array<int, 4>& roi) const
{
    const float cx = 0.5f * (track.item.box[0] + track.item.box[2]);
    const float cy = 0.5f * (track.item.box[1] + track.item.box[3]);
    return cx >= roi[0] && cx < roi[2] && cy >= roi[1] && cy < roi[3];
}

void ObstacleTracker::UpdateClassEvidence(Track* track,
                                          const DetectionItem& detection)
{
    // 指数衰减使旧类别证据逐渐失效；1.2 倍滞回避免相邻帧在两个类别间来回跳变。
    if (track == NULL) return;
    const bool indoor8 = semantic::ModelClassCount() == 8;
    const float evidence_decay = indoor8 ? 0.85f : 0.95f;
    for (size_t i = 0; i < track->class_evidence.size(); ++i) {
        track->class_evidence[i] *= evidence_decay;
    }
    if (detection.raw_class_id >= 0 &&
        detection.raw_class_id < static_cast<int>(track->class_evidence.size())) {
        track->class_evidence[detection.raw_class_id] += detection.score;
    }

    int best_class = detection.raw_class_id;
    float best_evidence = -1.0f;
    for (int i = 0; i < static_cast<int>(track->class_evidence.size()); ++i) {
        if (track->class_evidence[i] > best_evidence) {
            best_evidence = track->class_evidence[i];
            best_class = i;
        }
    }
    const int current = track->item.raw_class_id;
    const float current_evidence = current >= 0 &&
        current < static_cast<int>(track->class_evidence.size())
        ? track->class_evidence[current] : 0.0f;
    const bool fast_indoor_correction = indoor8 && current >= 0 &&
        detection.raw_class_id >= 0 && detection.raw_class_id != current &&
        detection.score >= 0.45f;
    if (current < 0) {
        track->pending_class_id = -1;
        track->pending_class_count = 0;
    } else if (fast_indoor_correction) {
        if (track->pending_class_id == detection.raw_class_id) {
            ++track->pending_class_count;
        } else {
            track->pending_class_id = detection.raw_class_id;
            track->pending_class_count = 1;
        }
        if (track->pending_class_count < 2) return;
        best_class = detection.raw_class_id;
    } else if (indoor8 && detection.raw_class_id >= 0 &&
               detection.raw_class_id != current) {
        // Count consecutive real top-1 observations from the first mismatch.
        // Waiting to count until the accumulated evidence already wins would
        // turn a three-frame rule into a much longer, class-locking delay.
        if (track->pending_class_id == detection.raw_class_id) {
            ++track->pending_class_count;
        } else {
            track->pending_class_id = detection.raw_class_id;
            track->pending_class_count = 1;
        }
        const float candidate_evidence =
            track->class_evidence[detection.raw_class_id];
        if (track->pending_class_count < 3 ||
            candidate_evidence <= current_evidence * 1.20f) {
            return;
        }
        best_class = detection.raw_class_id;
    } else if (best_class == current) {
        track->pending_class_id = -1;
        track->pending_class_count = 0;
        return;
    } else if (best_evidence > current_evidence * (indoor8 ? 1.20f : 1.50f)) {
        if (track->pending_class_id == best_class) ++track->pending_class_count;
        else {
            track->pending_class_id = best_class;
            track->pending_class_count = 1;
        }
        if (track->pending_class_count < 3) return;
    } else {
        track->pending_class_id = -1;
        track->pending_class_count = 0;
        return;
    }
    const int required_switch_observations = fast_indoor_correction ? 2 : 3;
    if (current < 0 ||
        track->pending_class_count >= required_switch_observations) {
        track->item.raw_class_id = best_class;
        track->item.raw_label = semantic::RawLabel(best_class);
        track->item.class_id = semantic::SemanticClassFromRaw(best_class);
        track->item.label = semantic::SemanticLabel(track->item.class_id);
        track->item.semantic_class = track->item.label;
        track->item.risk_weight = semantic::RiskWeight(track->item.class_id);
        track->pending_class_id = -1;
        track->pending_class_count = 0;
    }
}

void ObstacleTracker::UpdateRangeState(Track* track,
                                       const DetectionItem& detection,
                                       int64_t timestamp_ms)
{
    /*
     * 五点逆深度中值 + 非对称 alpha-beta 滤波。逆深度对框底像素误差近似
     * 线性，适合同时稳定近场和远场；突然接近会被限幅但立即用于风险升级，
     * 突然远离必须连续三次一致才接受，避免单帧框漂移把风险误降为 CLEAR。
     */
    if (track == NULL) return;
    const float dt = track->last_update_ms > 0
        ? clampf((timestamp_ms - track->last_update_ms) / 1000.0f, 0.01f, 0.50f)
        : 0.067f;
    if (detection.distance_m >= 0.0f) {
        float measured_distance = detection.distance_m;
        const float inverse_depth = 1.0f / std::max(0.20f, measured_distance);
        track->inverse_depth_history[track->inverse_depth_index] = inverse_depth;
        track->inverse_depth_index = (track->inverse_depth_index + 1) %
            static_cast<int>(track->inverse_depth_history.size());
        track->inverse_depth_count = std::min(
            track->inverse_depth_count + 1,
            static_cast<int>(track->inverse_depth_history.size()));
        // 所有距离统一使用最近 3~5 次逆深度中值。五点窗口只需三个新观测
        // 即可重捕获，避免人体前后移动后旧距离长期滞留。
        if (track->inverse_depth_count >= 3) {
            std::array<float, 5> sorted = track->inverse_depth_history;
            std::sort(sorted.begin(), sorted.begin() + track->inverse_depth_count);
            const float median_inverse = sorted[track->inverse_depth_count / 2];
            measured_distance = 1.0f / std::max(0.02f, median_inverse);
        }
        const float confidence = clampf(detection.distance_confidence, 0.10f, 0.95f);
        bool measurement_used = true;
        if (track->depth_measurements == 0 || track->depth_m < 0.0f) {
            track->depth_m = measured_distance;
            track->depth_velocity_mps = 0.0f;
            track->depth_variance = std::max(0.04f,
                detection.distance_sigma_m * detection.distance_sigma_m);
            track->pending_range_m = -1.0f;
            track->pending_range_count = 0;
        } else {
            const float predicted = track->depth_m + track->depth_velocity_mps * dt;
            const float residual = measured_distance - predicted;
            const float jump_gate = std::max(0.45f,
                0.25f * std::max(0.5f, predicted));
            const bool suspicious_jump = std::fabs(residual) > jump_gate &&
                detection.distance_source != "nearfield_cap";
            if (suspicious_jump) {
                const bool agrees_with_pending = track->pending_range_count > 0 &&
                    std::fabs(measured_distance - track->pending_range_m) <=
                    std::max(0.30f, 0.18f * track->pending_range_m);
                if (agrees_with_pending) {
                    ++track->pending_range_count;
                    track->pending_range_m = 0.5f *
                        (track->pending_range_m + measured_distance);
                } else {
                    track->pending_range_m = measured_distance;
                    track->pending_range_count = 1;
                }
                if (residual > 0.0f && track->pending_range_count < 3) {
                    // 风险下降必须慢：连续三次远距离一致才允许目标跳远。
                    measurement_used = false;
                    ++track->range_outlier_skips;
                    track->depth_m = clampf(predicted, 0.20f, 8.0f);
                    track->depth_velocity_mps *= 0.80f;
                    track->depth_variance = std::min(4.0f,
                        track->depth_variance + 0.05f + 0.02f * dt);
                } else if (residual < 0.0f && track->pending_range_count < 2) {
                    // 风险上升不能被完全忽略：首个突然接近值先按一个 gate
                    // 向近处移动，第二次一致后再接受完整测量。
                    measured_distance = std::max(measured_distance,
                                                 predicted - jump_gate);
                }
            } else {
                track->pending_range_m = -1.0f;
                track->pending_range_count = 0;
            }

            if (measurement_used) {
                const bool confirmed_jump = track->pending_range_count >=
                    (residual > 0.0f ? 3 : 2);
                const float accepted_distance = confirmed_jump
                    ? track->pending_range_m : measured_distance;
                const float accepted_residual = accepted_distance - predicted;
                // 接近时快速响应，远离时慢速释放；降低 beta 和速度上限，防止
                // 框抖动产生虚假高速/TTC。
                const float alpha = accepted_residual < 0.0f
                    ? 0.30f + 0.20f * confidence
                    : 0.14f + 0.14f * confidence;
                const float beta = accepted_residual < 0.0f
                    ? 0.035f + 0.055f * confidence
                    : 0.015f + 0.035f * confidence;
                track->depth_m = clampf(predicted + alpha * accepted_residual, 0.20f, 8.0f);
                track->depth_velocity_mps = clampf(
                    track->depth_velocity_mps + beta * accepted_residual / dt, -2.0f, 2.0f);
                const float measurement_variance = std::max(0.04f,
                    detection.distance_sigma_m * detection.distance_sigma_m);
                track->depth_variance = (1.0f - alpha) *
                    (track->depth_variance + 0.03f * dt) + alpha * measurement_variance;
                track->pending_range_m = -1.0f;
                track->pending_range_count = 0;
            }
        }
        if (measurement_used) ++track->depth_measurements;
        track->item.distance_m = track->depth_m;
        track->item.distance_sigma_m = std::sqrt(std::max(0.01f, track->depth_variance));
        track->item.safe_distance_m = clampf(
            track->depth_m - track->item.distance_sigma_m, 0.20f, 8.0f);
        // Preserve a detector-side near-field upper bound as planning evidence
        // without replacing the target-specific metric estimate shown on UART.
        if (detection.safe_distance_m > 0.0f) {
            track->item.safe_distance_m = std::min(
                track->item.safe_distance_m, detection.safe_distance_m);
        }
        track->item.distance_confidence = measurement_used
            ? detection.distance_confidence
            : std::max(0.15f, detection.distance_confidence * 0.60f);
        track->item.distance_source = measurement_used
            ? detection.distance_source : "temporal_hold_far_jump";
        track->item.range_measurements = track->depth_measurements;
        track->item.approach_mps = std::max(0.0f, -track->depth_velocity_mps);
        track->item.ttc_s = track->depth_measurements >= 5 &&
            track->item.approach_mps > 0.20f
            ? track->item.safe_distance_m / track->item.approach_mps : -1.0f;
    } else {
        track->item.safe_distance_m = detection.safe_distance_m;
        track->item.distance_sigma_m = detection.distance_sigma_m;
        track->item.distance_confidence = detection.distance_confidence;
        track->item.distance_source = detection.distance_source;
        track->item.ttc_s = -1.0f;
    }
    track->item.lateral_m = detection.lateral_m;
    track->item.risk_level = risk_from_safe(track->item.safe_distance_m,
                                             track->item.ttc_s);
}

void ObstacleTracker::StartTrack(const DetectionItem& detection,
                                 int frame_id,
                                 int64_t timestamp_ms,
                                 int view_id)
{
    Track track;
    track.item = detection;
    track.id = next_track_id_++;
    track.age = 1;
    track.hits = 1;
    track.last_frame = frame_id;
    track.last_seen_ms = timestamp_ms;
    track.last_update_ms = timestamp_ms;
    track.matched_current_frame = true;
    track.last_view_id = view_id;
    track.item.track_id = track.id;
    track.item.age = track.age;
    UpdateClassEvidence(&track, detection);
    UpdateRangeState(&track, detection, timestamp_ms);
    tracks_.push_back(track);
}

void ObstacleTracker::UpdateTrack(Track* track,
                                  const DetectionItem& detection,
                                  int frame_id,
                                  int64_t timestamp_ms,
                                  int view_id)
{
    // 运动越快，旧框权重越小；高置信新检测也会更快拉回真实位置，兼顾稳定与低延迟。
    if (track == NULL) return;
    DetectionItem effective_detection = detection;
    if (IsPersonPartBridge(*track, detection)) {
        const int person_raw_id = semantic::ModelClassCount() == 25 ? 3 : 0;
        effective_detection.raw_class_id = person_raw_id;
        effective_detection.raw_label = semantic::RawLabel(person_raw_id);
        effective_detection.class_id = semantic::PERSON;
        effective_detection.label = semantic::SemanticLabel(semantic::PERSON);
        effective_detection.semantic_class = effective_detection.label;
        effective_detection.risk_weight = semantic::RiskWeight(semantic::PERSON);
        ranging_.Estimate(&effective_detection);
    }
    const float overlap = utils::IoU(track->item.box, detection.box);
    float alpha = overlap >= 0.60f ? 0.25f : (overlap >= 0.30f ? 0.40f : 0.65f);
    if (track->last_view_id >= 0 && track->last_view_id != view_id) {
        alpha = std::min(alpha, 0.25f);
    }
    const float old_cx = 0.5f * (track->item.box[0] + track->item.box[2]);
    const float old_cy = 0.5f * (track->item.box[1] + track->item.box[3]);
    const float old_w = box_width(track->item.box);
    const float old_h = box_height(track->item.box);
    const float det_cx = 0.5f * (detection.box[0] + detection.box[2]);
    const float det_cy = 0.5f * (detection.box[1] + detection.box[3]);
    const float target_w = clampf(box_width(detection.box), old_w * 0.80f, old_w * 1.20f);
    const float target_h = clampf(box_height(detection.box), old_h * 0.80f, old_h * 1.20f);
    const float cx = old_cx + alpha * (det_cx - old_cx);
    const float cy = old_cy + alpha * (det_cy - old_cy);
    const float width = old_w + alpha * (target_w - old_w);
    const float height = old_h + alpha * (target_h - old_h);
    track->item.box = {
        clampf(cx - 0.5f * width, 0.0f, static_cast<float>(image_shape_[0] - 1)),
        clampf(cy - 0.5f * height, 0.0f, static_cast<float>(image_shape_[1] - 1)),
        clampf(cx + 0.5f * width, 0.0f, static_cast<float>(image_shape_[0] - 1)),
        clampf(cy + 0.5f * height, 0.0f, static_cast<float>(image_shape_[1] - 1))
    };
    track->item.score = 0.60f * track->item.score + 0.40f * detection.score;
    track->item.quality = detection.quality;
    track->item.sector = sector_from_box(track->item.box, image_shape_[0]);
    ++track->age;
    ++track->hits;
    track->missed = 0;
    track->last_frame = frame_id;
    track->last_seen_ms = timestamp_ms;
    track->matched_current_frame = true;
    track->last_view_id = view_id;
    UpdateClassEvidence(track, effective_detection);
    UpdateRangeState(track, effective_detection, timestamp_ms);
    track->last_update_ms = timestamp_ms;
    track->item.track_id = track->id;
    track->item.age = track->age;
    track->item.missed = 0;
}

void ObstacleTracker::AgeUnmatchedTracks(const std::vector<int>& matched_tracks,
                                         const std::array<int, 4>& roi,
                                         int frame_id,
                                         int64_t timestamp_ms)
{
    /*
     * 只有轨迹中心落在当前 ROI 内时，本帧未匹配才计为真正丢失；另一 ROI 中的
     * 轨迹仅短暂保留。这是交替 UPPER/LOWER 推理不会让框隔帧消失的关键。
     */
    for (size_t i = 0; i < tracks_.size(); ++i) {
        Track& track = tracks_[i];
        track.matched_current_frame = false;
        track.visible_in_current_roi = IsVisibleInRoi(track, roi);
        if (std::find(matched_tracks.begin(), matched_tracks.end(),
                      static_cast<int>(i)) != matched_tracks.end()) {
            track.matched_current_frame = true;
            continue;
        }
        ++track.age;
        if (track.visible_in_current_roi) ++track.missed;
        track.item.age = track.age;
        track.item.missed = track.missed;
        track.last_frame = frame_id;
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [timestamp_ms](const Track& track) {
            const int64_t age_ms = timestamp_ms - track.last_seen_ms;
            return (track.visible_in_current_roi && track.missed > 3) || age_ms > 900;
        }), tracks_.end());
}

void ObstacleTracker::RebuildStableResult(const DetectionResult& raw_result,
                                           int64_t timestamp_ms)
{
    // 对外只发布已确认且当前可见的轨迹，并按导航优先级截断为有限数量。
    stable_result_.Clear();
    stable_result_.raw_candidate_count = raw_result.raw_candidate_count;
    stable_result_.post_nms_count = raw_result.post_nms_count;
    stable_result_.coarse_drop_count = raw_result.coarse_drop_count;
    stable_result_.view_id = raw_result.view_id;
    stable_result_.roi = raw_result.roi;
    stable_result_.timestamp_ms = timestamp_ms;

    for (size_t i = 0; i < tracks_.size(); ++i) {
        const Track& track = tracks_[i];
        if (implausibly_broad_box(track.item, image_shape_)) continue;
        const bool inactive_view_hold = !track.visible_in_current_roi &&
                                        timestamp_ms - track.last_seen_ms <= 500;
        if (!track.matched_current_frame && !inactive_view_hold) continue;
        // 不绘制仅出现一帧的候选。双 ROI 交替时，同一 ROI 的两次观测间隔很短，
        // 等待第二次命中可过滤大部分由量化 head 瞬时波动产生的幽灵框。
        const int required_hits = track.item.class_id == semantic::PERSON
            ? kMinConfirmedHits : 3;
        if (track.hits < required_hits) continue;
        if (track.item.class_id == semantic::PERSON &&
            track.item.score < 0.11f && track.hits < 3) continue;
        DetectionItem item = track.item;
        if (!track.matched_current_frame) item.score *= 0.90f;
        stable_result_.items.push_back(item);
    }

    std::sort(stable_result_.items.begin(), stable_result_.items.end(), better_detection);
    // Tracks are associated per alternating ROI.  A partial upper-body box and
    // a lower-ROI body box can therefore survive as two track IDs even though
    // they describe the same person.  Reuse the navigation-aware containment
    // suppression before publishing so OSD and the planner see one entity.
    utils::MultiTargetNMS(&stable_result_, 0.45f, kMaxStableObjects);
    if (stable_result_.items.size() > static_cast<size_t>(kMaxStableObjects)) {
        stable_result_.items.resize(kMaxStableObjects);
    }
}

void ObstacleTracker::Update(const DetectionResult& raw_result, int frame_id)
{
    /*
     * 每帧建立全部 track-detection 候选边并全局降序选择，避免按输入顺序贪心
     * 导致多人场景交换 ID。关联完成后依次更新、老化、发布和规划。
     */
    const int64_t timestamp_ms = raw_result.timestamp_ms > 0
        ? raw_result.timestamp_ms : static_cast<int64_t>(frame_id) * 67;
    DetectionResult ranged = raw_result;
    for (size_t i = 0; i < ranged.items.size(); ++i) {
        ranging_.Estimate(&ranged.items[i]);
    }

    std::vector<MatchPair> pairs;
    pairs.reserve(tracks_.size() * ranged.items.size());
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        for (size_t di = 0; di < ranged.items.size(); ++di) {
            const float score = MatchScore(tracks_[ti], ranged.items[di]);
            if (score >= kMatchMinimum) {
                MatchPair pair;
                pair.track = static_cast<int>(ti);
                pair.detection = static_cast<int>(di);
                pair.score = score;
                pairs.push_back(pair);
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const MatchPair& a, const MatchPair& b) {
        return a.score > b.score;
    });

    std::vector<int> used_tracks(tracks_.size(), 0);
    std::vector<int> used_detections(ranged.items.size(), 0);
    std::vector<int> matched_tracks;
    for (size_t i = 0; i < pairs.size(); ++i) {
        const MatchPair& pair = pairs[i];
        if (used_tracks[pair.track] || used_detections[pair.detection]) continue;
        UpdateTrack(&tracks_[pair.track], ranged.items[pair.detection],
                    frame_id, timestamp_ms, raw_result.view_id);
        used_tracks[pair.track] = 1;
        used_detections[pair.detection] = 1;
        matched_tracks.push_back(pair.track);
    }

    AgeUnmatchedTracks(matched_tracks, raw_result.roi, frame_id, timestamp_ms);
    for (size_t di = 0; di < ranged.items.size(); ++di) {
        if (!used_detections[di] && CanStartTrack(ranged.items[di])) {
            StartTrack(ranged.items[di], frame_id, timestamp_ms,
                       raw_result.view_id);
        }
    }

    RebuildStableResult(raw_result, timestamp_ms);
    depth_fusion_.Apply(&stable_result_, &latest_surface_);
    decision_ = planner_.Update(stable_result_, raw_result.view_id, timestamp_ms);
}

void ObstacleTracker::PredictOnly(int frame_id, int64_t timestamp_ms)
{
    /*
     * A failed unified inference is not an empty scene. Keep the last boxes
     * alive briefly for display continuity, but the caller separately forces
     * AI_FAIL and never reuses stale scene/depth evidence for navigation.
     */
    if (timestamp_ms <= 0) timestamp_ms = static_cast<int64_t>(frame_id) * 67;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        Track& track = tracks_[i];
        track.matched_current_frame = false;
        track.visible_in_current_roi = false;
        ++track.age;
        track.item.age = track.age;
        track.last_frame = frame_id;
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [timestamp_ms](const Track& track) {
            return timestamp_ms - track.last_seen_ms > 700;
        }), tracks_.end());

    DetectionResult predicted;
    predicted.raw_candidate_count = stable_result_.raw_candidate_count;
    predicted.post_nms_count = stable_result_.post_nms_count;
    predicted.coarse_drop_count = stable_result_.coarse_drop_count;
    predicted.view_id = stable_result_.view_id;
    predicted.roi = stable_result_.roi;
    predicted.timestamp_ms = timestamp_ms;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        const Track& track = tracks_[i];
        if (track.hits < kMinConfirmedHits ||
            timestamp_ms - track.last_seen_ms > 700 ||
            implausibly_broad_box(track.item, image_shape_)) {
            continue;
        }
        DetectionItem item = track.item;
        item.score *= 0.97f;
        predicted.items.push_back(item);
    }
    std::sort(predicted.items.begin(), predicted.items.end(), better_detection);
    if (predicted.items.size() > static_cast<size_t>(kMaxStableObjects)) {
        predicted.items.resize(kMaxStableObjects);
    }
    stable_result_ = predicted;
    depth_fusion_.Apply(&stable_result_, &latest_surface_);
    decision_ = planner_.Update(stable_result_, predicted.view_id, timestamp_ms);
}

}  // namespace obstacle
