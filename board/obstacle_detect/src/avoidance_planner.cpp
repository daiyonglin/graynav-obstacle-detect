#include "../include/avoidance_planner.hpp"

#include "../include/semantic_config.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

namespace obstacle {
namespace {

/*
 * 避障规划层只消费跟踪稳定且完成测距的目标。它不修改检测框，而是将
 * 地面横向位置划入左/中/右走廊，计算各走廊最近保守距离和最小 TTC，
 * 再输出唯一动作，供 OSD、串口和语音共同使用。
 */

int action_severity(const std::string& action)
{
    if (action == "system_fault") return 5;
    if (action == "stop") return 4;
    if (action == "slow") return 3;
    if (action == "turn_left" || action == "turn_right") return 2;
    return 1;
}

void copy_zone(ZoneStatus* zone, const DetectionItem& item)
{
    zone->occupied = true;
    zone->label = item.label;
    zone->semantic_class = item.semantic_class;
    zone->risk_weight = item.risk_weight;
    zone->distance_m = item.safe_distance_m >= 0.0f ? item.safe_distance_m : item.distance_m;
    zone->risk_level = item.risk_level;
}

bool near_or_urgent(const AvoidancePlanner::Corridor& corridor)
{
    return corridor.zone.occupied &&
           (corridor.zone.risk_level == "urgent" ||
            corridor.zone.risk_level == "near" ||
             corridor.clearance < semantic::NearDistanceM() ||
             (corridor.min_ttc > 0.0f &&
              corridor.min_ttc < semantic::StopTtcSeconds()));
}

bool warning(const AvoidancePlanner::Corridor& corridor)
{
    return corridor.zone.occupied &&
           (corridor.zone.risk_level == "warning" ||
            corridor.clearance < semantic::WarningDistanceM());
}

}  // namespace

AvoidancePlanner::AvoidancePlanner()
    : image_shape_{720, 1280},
      last_view_ms_{0, 0},
      stable_action_("clear"),
      pending_action_("clear"),
      pending_count_(0),
      stable_since_ms_(0),
      pending_since_ms_(0)
{
}

void AvoidancePlanner::Initialize(const std::array<int, 2>& image_shape)
{
    image_shape_ = image_shape;
    last_view_ms_[0] = last_view_ms_[1] = 0;
    stable_action_ = "clear";
    pending_action_ = "clear";
    pending_count_ = 0;
    stable_since_ms_ = pending_since_ms_ = 0;
    std::cout << "[NAV][THRESHOLDS] urgent=" << semantic::UrgentDistanceM()
              << "m near=" << semantic::NearDistanceM()
              << "m warning=" << semantic::WarningDistanceM()
              << "m ttc_stop=" << semantic::StopTtcSeconds()
              << "s side_clear=" << semantic::SideClearDistanceM()
              << "m turn_margin=" << semantic::TurnClearanceMarginM()
              << "m sector=" << semantic::SectorLeftBoundaryRatio()
              << "/" << semantic::SectorRightBoundaryRatio()
              << " wide=" << semantic::WideBoxRatio()
              << " center_half=" << semantic::CenterCorridorHalfWidthM()
              << "m" << std::endl;
}

bool AvoidancePlanner::IsActionHazard(const DetectionItem& item) const
{
    // road 不作为障碍；建筑/标志等场景结构必须有可靠近场几何证据才干预。
    if (item.raw_class_id == 7 || item.raw_label == "road") return false;
    const bool scene_structure = item.raw_class_id == 1 || item.raw_class_id == 5 ||
                                 item.raw_class_id == 6 || item.raw_class_id == 12 ||
                                 item.raw_class_id == 22;
    if (scene_structure) {
        return item.distance_confidence >= 0.35f ||
               item.distance_source == "nearfield_bound" ||
               item.distance_source == "nearfield_cap";
    }
    return true;
}

void AvoidancePlanner::AddToCorridor(Corridor* corridor, const DetectionItem& item) const
{
    // 每条走廊只保留最危险目标摘要，同时累计最小 clearance 与最小 TTC。
    if (corridor == NULL) return;
    const float distance = item.safe_distance_m >= 0.0f
        ? item.safe_distance_m
        : (item.distance_m >= 0.0f ? item.distance_m : 2.5f);
    if (!corridor->zone.occupied || distance < corridor->clearance) {
        copy_zone(&corridor->zone, item);
    }
    corridor->clearance = std::min(corridor->clearance, distance);
    if (item.ttc_s > 0.0f && (corridor->min_ttc < 0.0f || item.ttc_s < corridor->min_ttc)) {
        corridor->min_ttc = item.ttc_s;
    }
}

std::string AvoidancePlanner::StabilizeAction(const std::string& desired, int64_t now_ms)
{
    /*
     * 动作滞回规则：风险升级可快速生效，STOP 立即生效；普通动作需连续
     * 两次确认；左右反转等待 300ms；STOP 解除等待 500ms；CLEAR 需稳定
     * 700ms。该状态机抑制检测抖动造成的语音左右反复切换。
     */
    if (stable_since_ms_ == 0) stable_since_ms_ = now_ms;
    if (desired == stable_action_) {
        pending_action_ = desired;
        pending_count_ = 0;
        pending_since_ms_ = now_ms;
        return stable_action_;
    }

    if (desired != pending_action_) {
        pending_action_ = desired;
        pending_count_ = 1;
        pending_since_ms_ = now_ms;
    } else {
        ++pending_count_;
    }

    const bool escalation = action_severity(desired) > action_severity(stable_action_);
    const bool stop_now = desired == "stop" || desired == "system_fault";
    const bool clear_ready = desired == "clear" && now_ms - pending_since_ms_ >= 700;
    const bool direction_flip =
        (stable_action_ == "turn_left" && desired == "turn_right") ||
        (stable_action_ == "turn_right" && desired == "turn_left");
    const bool direction_ready = !direction_flip || now_ms - pending_since_ms_ >= 300;
    const bool stop_release_ready = stable_action_ != "stop" ||
                                    now_ms - pending_since_ms_ >= 500;
    const bool normal_ready = pending_count_ >= 2 && direction_ready && stop_release_ready;

    if (stop_now || (escalation && desired == "slow") || clear_ready || normal_ready) {
        stable_action_ = desired;
        stable_since_ms_ = now_ms;
        pending_count_ = 0;
    }
    return stable_action_;
}

AvoidanceDecision AvoidancePlanner::Update(const DetectionResult& result,
                                            int view_id,
                                            int64_t timestamp_ms)
{
    /*
     * 双 ROI 的近期观测用于判断中央阻塞时哪一侧确实可通行。单纯侧方障碍
     * 则直接给出反方向转向建议，不再等待另一 ROI 完成安全确认。
     */
    if (view_id >= 0 && view_id < 2) last_view_ms_[view_id] = timestamp_ms;
    const bool both_views_recent =
        last_view_ms_[0] > 0 && last_view_ms_[1] > 0 &&
        timestamp_ms - last_view_ms_[0] < 500 &&
        timestamp_ms - last_view_ms_[1] < 500;

    Corridor left;
    Corridor center;
    Corridor right;
    left.zone.dir = "left";
    center.zone.dir = "center";
    right.zone.dir = "right";
    left.verified = center.verified = right.verified = both_views_recent;

    int nearest_track = -1;
    float nearest = 1e9f;
    const DetectionItem* nearest_item = NULL;
    const DetectionItem* depth_candidate = NULL;
    bool wide_urgent = false;
    bool uncertain_hazard = false;
    for (size_t i = 0; i < result.items.size(); ++i) {
        const DetectionItem& item = result.items[i];
        if (!IsActionHazard(item)) continue;
        if (item.depth_level != "unknown" &&
            (depth_candidate == NULL ||
             item.depth_confidence > depth_candidate->depth_confidence)) {
            depth_candidate = &item;
        }
        const float distance = item.safe_distance_m >= 0.0f
            ? item.safe_distance_m : item.distance_m;
        if (distance >= 0.0f && distance < nearest) {
            nearest = distance;
            nearest_track = item.track_id;
            nearest_item = &item;
        }
        if (item.distance_confidence < 0.25f && item.quality != "good") {
            uncertain_hazard = true;
        }

        const bool wide = item.sector == "wide";
        if (wide) {
            // 粗粒度宽框仅作为“不确定风险”保留，不能单独封死全部走廊，
            // 也不能在缺少近场可靠证据时直接触发 STOP。
            if (item.quality == "coarse" || item.distance_confidence < 0.25f) {
                uncertain_hazard = true;
                continue;
            }
            AddToCorridor(&left, item);
            AddToCorridor(&center, item);
            AddToCorridor(&right, item);
            if (item.quality != "coarse" &&
                (item.risk_level == "urgent" || item.risk_level == "near")) {
                wide_urgent = true;
            }
            continue;
        }

        const float center_half_width = semantic::CenterCorridorHalfWidthM();
        if (item.lateral_m < -center_half_width || item.sector == "left") {
            AddToCorridor(&left, item);
        } else if (item.lateral_m > center_half_width || item.sector == "right") {
            AddToCorridor(&right, item);
        } else {
            AddToCorridor(&center, item);
        }
        if (item.sector == "left_center") AddToCorridor(&center, item);
        if (item.sector == "center_right") AddToCorridor(&center, item);
    }

    const bool left_near = near_or_urgent(left);
    const bool center_near = near_or_urgent(center);
    const bool right_near = near_or_urgent(right);
    const bool left_warning = warning(left);
    const bool center_warning = warning(center);
    const bool right_warning = warning(right);
    std::string desired = "clear";
    std::string reason = "clear";

    const bool center_ttc_urgent = center.min_ttc > 0.0f &&
                                   center.min_ttc < semantic::StopTtcSeconds();
    const bool left_clear = left.verified && !left_near &&
                            left.clearance > semantic::SideClearDistanceM();
    const bool right_clear = right.verified && !right_near &&
                             right.clearance > semantic::SideClearDistanceM();
    if (wide_urgent || center_ttc_urgent || (center_near && !left_clear && !right_clear)) {
        desired = "stop";
        reason = wide_urgent ? "wide_near" : "center_blocked_no_verified_side";
    } else if (center_near || left_near || right_near) {
        // 单纯侧方近障不再等待双 ROI 将另一侧标记为 verified：右侧障碍直接
        // 提示左转，左侧障碍直接提示右转。中央阻塞时仍要求候选走廊已确认安全。
        if (right_near && !center_near && !left_near) {
            desired = "turn_left";
            reason = "right_obstacle_direct_avoid";
        } else if (left_near && !center_near && !right_near) {
            desired = "turn_right";
            reason = "left_obstacle_direct_avoid";
        } else if ((center_near || right_near) && left_clear &&
            left.clearance > right.clearance + semantic::TurnClearanceMarginM()) {
            desired = "turn_left";
            reason = "left_corridor_verified";
        } else if ((center_near || left_near) && right_clear &&
                   right.clearance > left.clearance + semantic::TurnClearanceMarginM()) {
            desired = "turn_right";
            reason = "right_corridor_verified";
        } else {
            desired = center_near ? "stop" : "slow";
            reason = center_near ? "center_near" : "side_near";
        }
    } else if (center_warning || left_warning || right_warning || uncertain_hazard) {
        if (right_warning && !center_warning && !left_warning) {
            desired = "turn_left";
            reason = "right_warning_direct_avoid";
        } else if (left_warning && !center_warning && !right_warning) {
            desired = "turn_right";
            reason = "left_warning_direct_avoid";
        } else {
            desired = "slow";
            reason = uncertain_hazard ? "uncertain_obstacle" : "warning_range";
        }
    }

    AvoidanceDecision decision;
    if (nearest_item == NULL) nearest_item = depth_candidate;
    decision.left = left.zone;
    decision.center = center.zone;
    decision.right = right.zone;
    decision.nearest_track_id = nearest_track >= 0 ? nearest_track :
        (nearest_item != NULL ? nearest_item->track_id : -1);
    if (nearest_item != NULL) {
        decision.depth_level = nearest_item->depth_level;
        decision.depth_confidence = nearest_item->depth_confidence;
        decision.depth_margin = 0.0f;
        decision.depth_ambiguous = nearest_item->depth_level == "unknown";
        decision.depth_source = nearest_item->depth_source;
        decision.depth_consistent = nearest_item->depth_consistent;
        decision.approaching = nearest_item->approaching;
    }
    decision.action = StabilizeAction(desired, timestamp_ms);
    std::ostringstream prompt;
    prompt << "reason=" << reason
           << " desired=" << desired
           << " views=" << (both_views_recent ? "verified" : "warming")
           << " clearances=" << left.clearance << "/" << center.clearance << "/" << right.clearance;
    decision.prompt = prompt.str();
    return decision;
}

}  // namespace obstacle
