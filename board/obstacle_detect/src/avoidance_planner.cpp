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
    zone->raw_label = item.raw_label.empty() ? item.label : item.raw_label;
    zone->semantic_class = item.semantic_class;
    zone->risk_weight = item.risk_weight;
    zone->distance_estimate_m = item.distance_m;
    zone->safe_distance_m = item.safe_distance_m >= 0.0f
        ? item.safe_distance_m : item.distance_m;
    zone->distance_m = zone->safe_distance_m;
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
    const bool legacy_rod25 = semantic::ModelClassCount() == 25;
    if (legacy_rod25 && (item.raw_class_id == 7 || item.raw_label == "road")) {
        return false;
    }
    const bool scene_structure = legacy_rod25 &&
        (item.raw_class_id == 1 || item.raw_class_id == 5 ||
         item.raw_class_id == 6 || item.raw_class_id == 12 ||
         item.raw_class_id == 22);
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
    ++corridor->zone.object_count;
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
    const bool direct_side_escape = stable_action_ == "slow" &&
        (desired == "turn_left" || desired == "turn_right");
    const bool normal_ready = (direct_side_escape || pending_count_ >= 2) &&
                              direction_ready && stop_release_ready;

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
    // A side object may overlap the narrow centre footprint even though its
    // physical centre is clearly on one side.  Track the primary horizontal
    // origin separately so that a large chair on the left produces RIGHT,
    // rather than being mistaken for an unescapable two-corridor blockage.
    bool primary_near[3] = {false, false, false};
    bool primary_warning[3] = {false, false, false};
    bool primary_center_ttc_urgent = false;
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

        // A named detection represents one physical obstacle, so publish it in
        // exactly one primary horizontal zone.  The previous footprint rule
        // copied a large right-side person into LEFT/CENTER/RIGHT and made one
        // target look like a three-obstacle blockade.  Truly frame-spanning
        // detections were handled by the explicit `wide` branch above.
        const float frame_w = std::max(1.0f, static_cast<float>(image_shape_[0]));
        const float bx1 = std::max(0.0f, std::min(1.0f, item.box[0] / frame_w));
        const float bx2 = std::max(bx1, std::min(1.0f, item.box[2] / frame_w));
        const float box_span = std::max(0.01f, bx2 - bx1);
        const float box_center = 0.5f * (bx1 + bx2);
        int primary_zone = 1;
        // Only a genuinely frame-spanning box loses its lateral identity.
        // Partial-person boxes are often 55--75% of the image width; forcing
        // those boxes to CENTER made one person on the right look like three
        // independent corridor obstacles.  The model/ranging stage already
        // owns the stricter wide-box contract, so use the same threshold here.
        if (box_span < semantic::WideBoxRatio()) {
            if (box_center < semantic::SectorLeftBoundaryRatio()) {
                primary_zone = 0;
            } else if (box_center > semantic::SectorRightBoundaryRatio()) {
                primary_zone = 2;
            }
        }
        const bool item_near = item.risk_level == "urgent" ||
            item.risk_level == "near" ||
            (distance >= 0.0f && distance < semantic::NearDistanceM()) ||
            (item.ttc_s > 0.0f && item.ttc_s < semantic::StopTtcSeconds());
        const bool item_warning = item_near || item.risk_level == "warning" ||
            (distance >= 0.0f && distance < semantic::WarningDistanceM());
        primary_near[primary_zone] = primary_near[primary_zone] || item_near;
        primary_warning[primary_zone] = primary_warning[primary_zone] || item_warning;
        if (primary_zone == 1 && item.ttc_s > 0.0f &&
            item.ttc_s < semantic::StopTtcSeconds()) {
            primary_center_ttc_urgent = true;
        }
        Corridor* corridors[3] = {&left, &center, &right};
        AddToCorridor(corridors[primary_zone], item);
    }

    const bool left_near = near_or_urgent(left);
    const bool center_near = near_or_urgent(center);
    const bool right_near = near_or_urgent(right);
    const bool left_warning = warning(left);
    const bool center_warning = warning(center);
    const bool right_warning = warning(right);
    std::string desired = "clear";
    std::string reason = "clear";

    const bool left_clear = left.verified && !left_near &&
                            left.clearance > semantic::SideClearDistanceM();
    const bool right_clear = right.verified && !right_near &&
                             right.clearance > semantic::SideClearDistanceM();
    const int near_count = static_cast<int>(left_near) +
        static_cast<int>(center_near) + static_cast<int>(right_near);
    const int warning_count = static_cast<int>(left_warning) +
        static_cast<int>(center_warning) + static_cast<int>(right_warning);
    // Footprint occupancy answers "which paths does this box touch?" whereas
    // primary evidence answers "where is the physical obstacle?".  Do not
    // cancel a right-side turn merely because the same large box also touches
    // the centre/left corridor masks.
    const bool left_primary_only_near = primary_near[0] &&
        !primary_near[1] && !primary_near[2];
    const bool right_primary_only_near = primary_near[2] &&
        !primary_near[0] && !primary_near[1];
    const bool left_primary_only_warning = primary_warning[0] &&
        !primary_warning[1] && !primary_warning[2];
    const bool right_primary_only_warning = primary_warning[2] &&
        !primary_warning[0] && !primary_warning[1];
    if (wide_urgent || primary_center_ttc_urgent) {
        desired = "stop";
        reason = wide_urgent ? "wide_near" : "center_primary_ttc_urgent";
    } else if (left_primary_only_near) {
        desired = "turn_right";
        reason = "left_primary_near_avoid_right";
    } else if (right_primary_only_near) {
        desired = "turn_left";
        reason = "right_primary_near_avoid_left";
    } else if (near_count == 3) {
        desired = "stop";
        reason = "three_corridors_distinct_near";
    } else if (near_count == 2) {
        if (!left_near && !left.zone.occupied) {
            desired = "turn_left";
            reason = "two_corridors_open_left";
        } else if (!right_near && !right.zone.occupied) {
            desired = "turn_right";
            reason = "two_corridors_open_right";
        } else if (!center_near) {
            desired = "slow";
            reason = "side_corridors_near_center_open";
        } else {
            desired = "stop";
            reason = "two_corridors_no_verified_escape";
        }
    } else if (center_near || left_near || right_near) {
        // 单纯侧方近障不再等待双 ROI 将另一侧标记为 verified：右侧障碍直接
        // 提示左转，左侧障碍直接提示右转。中央阻塞时仍要求候选走廊已确认安全。
        if (right_near && !center_near && !left_near) {
            desired = "turn_left";
            reason = "right_obstacle_direct_avoid";
        } else if (left_near && !center_near && !right_near) {
            desired = "turn_right";
            reason = "left_obstacle_direct_avoid";
        } else if (center_near && left_clear && !right_clear) {
            desired = "turn_left";
            reason = "only_left_corridor_verified";
        } else if (center_near && right_clear && !left_clear) {
            desired = "turn_right";
            reason = "only_right_corridor_verified";
        } else if (center_near && left_clear && right_clear &&
                   left.clearance > right.clearance + semantic::TurnClearanceMarginM()) {
            desired = "turn_left";
            reason = "left_corridor_safer";
        } else if (center_near && left_clear && right_clear &&
                   right.clearance > left.clearance + semantic::TurnClearanceMarginM()) {
            desired = "turn_right";
            reason = "right_corridor_safer";
        } else {
            desired = center_near ? "stop" : "slow";
            reason = center_near ? "center_near_no_clear_margin" : "side_near";
        }
    } else if (left_primary_only_warning) {
        desired = "turn_right";
        reason = "left_primary_warning_avoid_right";
    } else if (right_primary_only_warning) {
        desired = "turn_left";
        reason = "right_primary_warning_avoid_left";
    } else if (warning_count >= 2) {
        if (!left_warning && !left.zone.occupied) {
            desired = "turn_left";
            reason = "two_warnings_open_left";
        } else if (!right_warning && !right.zone.occupied) {
            desired = "turn_right";
            reason = "two_warnings_open_right";
        } else {
            desired = "slow";
            reason = "multiple_warning_slow";
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
        decision.primary_class = nearest_item->raw_label.empty()
            ? nearest_item->label : nearest_item->raw_label;
        decision.distance_estimate_m = nearest_item->distance_m;
    }
    const bool active_left = left_near || left_warning;
    const bool active_center = center_near || center_warning;
    const bool active_right = right_near || right_warning;
    const bool primary_active[3] = {
        primary_near[0] || primary_warning[0],
        primary_near[1] || primary_warning[1],
        primary_near[2] || primary_warning[2]
    };
    const int primary_active_count = static_cast<int>(primary_active[0]) +
        static_cast<int>(primary_active[1]) + static_cast<int>(primary_active[2]);
    const int active_count = static_cast<int>(active_left) +
        static_cast<int>(active_center) + static_cast<int>(active_right);
    if (wide_urgent) {
        decision.hazard_sector = "blocked";
    } else if (primary_active_count > 0) {
        decision.hazard_sector = primary_active_count >= 3 ? "blocked" :
            (primary_active_count == 2 ? "multi" :
             (primary_active[0] ? "left" :
              (primary_active[2] ? "right" : "center")));
    } else {
        decision.hazard_sector = active_count >= 3 ? "blocked" :
            (active_count == 2 ? "multi" :
             (active_left ? "left" : (active_right ? "right" : "center")));
    }
    decision.action = StabilizeAction(desired, timestamp_ms);
    decision.recommended_direction = decision.action == "turn_left" ? "left" :
        (decision.action == "turn_right" ? "right" :
         (decision.action == "stop" ? "hold" : "forward"));
    decision.hazard_position = decision.hazard_sector == "left" ? "LEFT" :
        (decision.hazard_sector == "right" ? "RIGHT" :
         (decision.hazard_sector == "multi" ? "MULTI" :
          (decision.hazard_sector == "blocked" ? "BLOCKED" : "FRONT")));
    decision.risk = decision.action == "stop" ? "URGENT" :
        (decision.action == "clear" ? "SAFE" :
         (uncertain_hazard ? "UNKNOWN" : "WARNING"));
    std::ostringstream prompt;
    prompt << "reason=" << reason
           << " desired=" << desired
           << " views=" << (both_views_recent ? "verified" : "warming")
           << " clearances=" << left.clearance << "/" << center.clearance << "/" << right.clearance;
    decision.prompt = prompt.str();
    return decision;
}

}  // namespace obstacle
