#include "../include/utils.hpp"
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace {

float navigation_rank(const DetectionItem& item)
{
    float rank = item.score * 0.35f + item.risk_weight * 0.35f;
    if (item.distance_m >= 0.0f) {
        rank += std::max(0.0f, 3.0f - item.distance_m) * 0.18f;
    }
    if (item.risk_level == "urgent") rank += 0.50f;
    else if (item.risk_level == "near") rank += 0.35f;
    else if (item.risk_level == "warning") rank += 0.20f;
    if (item.quality == "good") rank += 0.12f;
    if (item.quality == "coarse") rank -= 0.20f;
    return rank;
}

}  // namespace

namespace utils {

float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b)
{
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);

    const float inter_w = std::max(0.0f, x2 - x1);
    const float inter_h = std::max(0.0f, y2 - y1);
    const float inter_area = inter_w * inter_h;

    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);

    const float union_area = area_a + area_b - inter_area;
    if (union_area <= 1e-6f) {
        return 0.0f;
    }

    return inter_area / union_area;
}

void SortDetectionResult(DetectionResult* result)
{
    std::sort(result->items.begin(), result->items.end(),
              [](const DetectionItem& a, const DetectionItem& b) {
                  const float ar = navigation_rank(a);
                  const float br = navigation_rank(b);
                  if (std::fabs(ar - br) > 0.03f) {
                      return ar > br;
                  }
                  if (a.quality != b.quality) {
                      if (a.quality == "good") return true;
                      if (b.quality == "good") return false;
                      if (a.quality == "coarse") return false;
                      if (b.quality == "coarse") return true;
                  }
                  const bool a_has_dist = a.distance_m >= 0.0f;
                  const bool b_has_dist = b.distance_m >= 0.0f;
                  if (a_has_dist && b_has_dist && std::fabs(a.distance_m - b.distance_m) > 0.05f) {
                      return a.distance_m < b.distance_m;
                  }
                  if (a_has_dist != b_has_dist) {
                      return a_has_dist;
                  }
                  return a.score > b.score;
              });
}

void NMS(DetectionResult* result, float iou_threshold, int top_k)
{
    auto area = [](const DetectionItem& item) {
        return std::max(0.0f, item.box[2] - item.box[0]) *
               std::max(0.0f, item.box[3] - item.box[1]);
    };

    std::sort(result->items.begin(), result->items.end(),
              [](const DetectionItem& a, const DetectionItem& b) {
                  return a.score > b.score;
              });

    if ((int)result->items.size() > top_k) {
        result->items.resize(top_k);
    }

    std::vector<int> suppressed(result->items.size(), 0);
    std::vector<DetectionItem> kept;
    kept.reserve(result->items.size());

    for (size_t i = 0; i < result->items.size(); ++i) {
        if (suppressed[i]) continue;

        const DetectionItem& cur = result->items[i];
        kept.push_back(cur);

        for (size_t j = i + 1; j < result->items.size(); ++j) {
            if (suppressed[j]) continue;

            const DetectionItem& other = result->items[j];

            // 閻滄澘婀?class_id 瀹歌尙绮￠弰顖涙付缂佸牊妯夌粈铏硅閸掝偓绱?            // 0 -> person
            // 1 -> obstacle
            if (cur.raw_class_id != other.raw_class_id) continue;

            const float overlap = IoU(cur.box, other.box);
            if (overlap > iou_threshold) {
                const float cur_area = area(cur);
                const float other_area = area(other);
                const bool other_much_smaller = other_area < cur_area * 0.60f;
                const bool cur_much_smaller = cur_area < other_area * 0.60f;

                // In obstacle scenes a coarse high-score box often covers
                // several smaller objects. Keep a smaller candidate when its
                // score is close enough, then let display-level filtering
                // decide which one should drive navigation.
                if (other_much_smaller && other.score >= cur.score * 0.65f) {
                    continue;
                }
                if (cur_much_smaller && cur.score >= other.score * 0.65f) {
                    suppressed[j] = 1;
                    continue;
                }
                suppressed[j] = 1;
            }
        }
    }

    result->items.swap(kept);
}

namespace {

bool is_furniture_raw_class_for_multi_nms(int cls_id)
{
    return obstacle::semantic::IsFurnitureLikeRawClass(cls_id);
}

float box_area_for_multi_nms(const DetectionItem& item)
{
    return std::max(0.0f, item.box[2] - item.box[0]) *
           std::max(0.0f, item.box[3] - item.box[1]);
}

float box_width_ratio_for_multi_nms(const DetectionItem& item)
{
    constexpr float kFrameW = 720.0f;
    return std::max(0.0f, item.box[2] - item.box[0]) / kFrameW;
}

bool is_broad_coarse_obstacle_for_multi_nms(const DetectionItem& item)
{
    return obstacle::semantic::IsObstacleClass(item.class_id) &&
           item.quality == "coarse" &&
           box_width_ratio_for_multi_nms(item) > 0.58f;
}

bool is_broad_obstacle_for_multi_nms(const DetectionItem& item)
{
    return obstacle::semantic::IsObstacleClass(item.class_id) &&
           (item.sector == "wide" || box_width_ratio_for_multi_nms(item) > 0.58f);
}

float nms_priority_for_multi_target(const DetectionItem& item)
{
    float priority = item.score;
    if (item.quality == "good") priority += 0.10f;
    if (item.quality == "coarse") priority -= 0.25f;
    if (item.class_id == DISPLAY_CLASS_PERSON) priority += 0.08f;
    if (obstacle::semantic::IsFurnitureLikeSemantic(item.class_id)) priority += 0.05f;
    if (is_broad_obstacle_for_multi_nms(item)) priority -= 0.15f;
    if (item.distance_source == "existence") priority -= 0.08f;
    return priority;
}

float center_x_for_multi_nms(const DetectionItem& item)
{
    return 0.5f * (item.box[0] + item.box[2]);
}

std::string sector_for_multi_nms(const DetectionItem& item)
{
    const float w = 720.0f;
    const float left_bound = 0.35f * w;
    const float right_bound = 0.65f * w;
    const float box_w = std::max(1.0f, item.box[2] - item.box[0]);
    const float left_overlap =
        std::max(0.0f, std::min(item.box[2], left_bound) - std::max(item.box[0], 0.0f)) / box_w;
    const float center_overlap =
        std::max(0.0f, std::min(item.box[2], right_bound) - std::max(item.box[0], left_bound)) / box_w;
    const float right_overlap =
        std::max(0.0f, std::min(item.box[2], w) - std::max(item.box[0], right_bound)) / box_w;

    if (left_overlap >= 0.20f && center_overlap >= 0.20f && right_overlap >= 0.20f) return "wide";
    if (center_overlap >= 0.50f) return "center";
    if (center_overlap >= 0.25f && left_overlap >= 0.25f) return "left_center";
    if (center_overlap >= 0.25f && right_overlap >= 0.25f) return "center_right";
    return (left_overlap >= right_overlap) ? "left" : "right";
}

bool should_suppress_for_multi_nms(const DetectionItem& cur,
                                   const DetectionItem& other,
                                   float overlap,
                                   float iou_threshold)
{
    if (overlap <= 0.0f) {
        return false;
    }

    const float cur_area = box_area_for_multi_nms(cur);
    const float other_area = box_area_for_multi_nms(other);
    const float min_area = std::max(1.0f, std::min(cur_area, other_area));
    const float max_area = std::max(1.0f, std::max(cur_area, other_area));
    const bool protected_raw =
        cur.class_id == DISPLAY_CLASS_PERSON || other.class_id == DISPLAY_CLASS_PERSON ||
        obstacle::semantic::IsFurnitureLikeSemantic(cur.class_id) ||
        obstacle::semantic::IsFurnitureLikeSemantic(other.class_id) ||
        is_furniture_raw_class_for_multi_nms(cur.raw_class_id) ||
        is_furniture_raw_class_for_multi_nms(other.raw_class_id);
    const bool other_much_smaller = other_area < cur_area * 0.70f;
    const bool cur_much_smaller = cur_area < other_area * 0.70f;
    const float center_dx = std::fabs(center_x_for_multi_nms(cur) - center_x_for_multi_nms(other)) / 720.0f;
    const std::string cur_sector = sector_for_multi_nms(cur);
    const std::string other_sector = sector_for_multi_nms(other);
    const bool spatially_separated =
        center_dx > 0.12f ||
        (cur_sector != other_sector && cur_sector != "wide" && other_sector != "wide");

    if (spatially_separated && protected_raw) {
        return false;
    }
    if (is_broad_obstacle_for_multi_nms(cur) &&
        other_much_smaller &&
        overlap > 0.03f &&
        other.score >= cur.score * 0.25f) {
        return false;
    }
    if (is_broad_obstacle_for_multi_nms(other) &&
        cur_much_smaller &&
        overlap > 0.03f &&
        cur.score >= other.score * 0.25f) {
        return true;
    }
    if (other_much_smaller && other.score >= cur.score * (protected_raw ? 0.45f : 0.60f)) {
        return false;
    }
    if (cur_much_smaller && cur.score >= other.score * (protected_raw ? 0.45f : 0.60f)) {
        return true;
    }

    const bool same_raw = cur.raw_class_id == other.raw_class_id;
    const bool duplicate_same_anchor =
        overlap > 0.88f &&
        max_area / min_area < 1.30f &&
        center_dx < 0.035f;
    if (!same_raw && !duplicate_same_anchor) {
        return false;
    }

    const float threshold = (!same_raw || protected_raw)
        ? std::max(0.72f, iou_threshold) : iou_threshold;
    return overlap > threshold;
}

}  // namespace

void MultiTargetNMS(DetectionResult* result, float iou_threshold, int top_k)
{
    std::sort(result->items.begin(), result->items.end(),
              [](const DetectionItem& a, const DetectionItem& b) {
                  const float ap = nms_priority_for_multi_target(a);
                  const float bp = nms_priority_for_multi_target(b);
                  if (std::fabs(ap - bp) > 0.02f) {
                      return ap > bp;
                  }
                  return a.score > b.score;
              });

    if ((int)result->items.size() > top_k) {
        result->items.resize(top_k);
    }

    std::vector<int> suppressed(result->items.size(), 0);
    std::vector<DetectionItem> kept;
    kept.reserve(result->items.size());

    for (size_t i = 0; i < result->items.size(); ++i) {
        if (suppressed[i]) continue;

        const DetectionItem& cur = result->items[i];
        kept.push_back(cur);

        for (size_t j = i + 1; j < result->items.size(); ++j) {
            if (suppressed[j]) continue;

            const DetectionItem& other = result->items[j];
            const float overlap = IoU(cur.box, other.box);
            if (should_suppress_for_multi_nms(cur, other, overlap, iou_threshold)) {
                suppressed[j] = 1;
            }
        }
    }

    result->items.swap(kept);
}

}  // namespace utils


void VISUALIZER::Initialize(std::array<int, 2>& in_img_shape)
{
    osd_device.Initialize(in_img_shape[0], in_img_shape[1]);
    last_action_asset_.clear();
    last_info_asset_.clear();
    static_layers_cleaned_ = false;
}

void VISUALIZER::Release()
{
    osd_device.Release();
}

void VISUALIZER::DrawTestBox()
{
    std::vector<sst::device::osd::OsdQuadRangle> quads;

    sst::device::osd::OsdQuadRangle q;
    q.box = {100.f, 100.f, 300.f, 300.f};
    q.border = 3;
    q.layer_id = 0;
    q.type = fdevice::TYPE_HOLLOW;
    q.alpha = fdevice::TYPE_ALPHA75;
    q.color = 0;

    quads.emplace_back(q);
    osd_device.Draw(quads);
}

int VISUALIZER::PickColorByClass(int class_id) const
{
    (void)class_id;
    // Use one OSD LUT entry for all detection boxes. Aurora renders the LUT
    // colors over a grayscale preview, so class-dependent colors looked like
    // inconsistent black/gray boxes during field tests.
    return 2;
}

namespace {

float nearest_distance_from_decision(const AvoidanceDecision& decision)
{
    float nearest = 1e9f;
    const ZoneStatus* zones[3] = {&decision.left, &decision.center, &decision.right};
    for (int i = 0; i < 3; ++i) {
        if (zones[i]->occupied && zones[i]->distance_m >= 0.0f) {
            nearest = std::min(nearest, zones[i]->distance_m);
        }
    }
    return nearest < 1e8f ? nearest : -1.0f;
}

std::string nearest_dir_from_decision(const AvoidanceDecision& decision)
{
    float nearest = 1e9f;
    std::string dir = "center";
    const ZoneStatus* zones[3] = {&decision.left, &decision.center, &decision.right};
    for (int i = 0; i < 3; ++i) {
        if (zones[i]->occupied && zones[i]->distance_m >= 0.0f && zones[i]->distance_m < nearest) {
            nearest = zones[i]->distance_m;
            dir = zones[i]->dir;
        }
    }
    return dir;
}

int find_primary_index(const DetectionResult& result)
{
    if (result.items.empty()) {
        return -1;
    }

    int best = 0;
    for (size_t i = 1; i < result.items.size(); ++i) {
        const DetectionItem& cur = result.items[i];
        const DetectionItem& prev = result.items[best];
        const bool cur_has_dist = cur.distance_m >= 0.0f;
        const bool prev_has_dist = prev.distance_m >= 0.0f;
        const float cur_rank = navigation_rank(cur);
        const float prev_rank = navigation_rank(prev);
        if (cur_rank > prev_rank + 0.03f) {
            best = static_cast<int>(i);
            continue;
        }
        if (cur.quality != prev.quality) {
            if (cur.quality == "good" ||
                (prev.quality == "coarse" && cur.quality != "coarse")) {
                best = static_cast<int>(i);
            }
            continue;
        }
        if (cur_has_dist && prev_has_dist && cur.distance_m + 0.05f < prev.distance_m) {
            best = static_cast<int>(i);
        } else if (cur_has_dist && !prev_has_dist) {
            best = static_cast<int>(i);
        } else if (cur_has_dist == prev_has_dist && cur.score > prev.score + 0.05f) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

void push_quad(std::vector<sst::device::osd::OsdQuadRangle>* quads,
               float x1,
               float y1,
               float x2,
               float y2,
               int border,
               fdevice::QUADRANGLETYPE type,
               fdevice::ALPHATYPE alpha,
               int color = 2)
{
    sst::device::osd::OsdQuadRangle q;
    q.box = {x1, y1, x2, y2};
    q.border = border;
    q.layer_id = 0;
    q.type = type;
    q.alpha = alpha;
    q.color = color;
    quads->emplace_back(q);
}

void push_hollow(std::vector<sst::device::osd::OsdQuadRangle>* quads,
                 float x1,
                 float y1,
                 float x2,
                 float y2,
                 int border)
{
    push_quad(quads, x1, y1, x2, y2, border, fdevice::TYPE_HOLLOW, fdevice::TYPE_ALPHA75);
}

void push_solid(std::vector<sst::device::osd::OsdQuadRangle>* quads,
                float x1,
                float y1,
                float x2,
                float y2)
{
    push_quad(quads, x1, y1, x2, y2, 1, fdevice::TYPE_SOLID, fdevice::TYPE_ALPHA100);
}

int action_level(const std::string& action)
{
    if (action == "stop") return 3;
    if (action == "system_fault") return 3;
    if (action == "slow" || action == "turn_left" || action == "turn_right") return 2;
    return 0;
}

int distance_level(float distance_m)
{
    if (distance_m < 0.0f) return 0;
    if (distance_m < 1.0f) return 3;
    if (distance_m < 2.0f) return 2;
    return 1;
}

class HudGlyphRenderer {
public:
    static void DrawWord(std::vector<sst::device::osd::OsdQuadRangle>* quads,
                         float x,
                         float y,
                         float scale,
                         const std::string& word) {
        const float glyph_w = 42.0f * scale;
        const float gap = 10.0f * scale;
        float cursor = x;
        for (size_t i = 0; i < word.size(); ++i) {
            const char ch = static_cast<char>(std::toupper(static_cast<unsigned char>(word[i])));
            if (ch == ' ') {
                cursor += gap * 1.5f;
                continue;
            }
            DrawChar(quads, cursor, y, scale, ch);
            cursor += glyph_w + gap;
        }
    }

private:
    static void Segment(std::vector<sst::device::osd::OsdQuadRangle>* quads,
                        float x,
                        float y,
                        float scale,
                        int seg) {
        const float w = 42.0f * scale;
        const float h = 62.0f * scale;
        const float t = 8.0f * scale;
        const float mid_y = y + h * 0.5f - t * 0.5f;
        const float cx = x + w * 0.5f - t * 0.5f;

        switch (seg) {
            case 0: push_solid(quads, x + t, y, x + w - t, y + t); break;                         // top
            case 1: push_solid(quads, x, y + t, x + t, y + h * 0.5f); break;                       // upper-left
            case 2: push_solid(quads, x + w - t, y + t, x + w, y + h * 0.5f); break;               // upper-right
            case 3: push_solid(quads, x + t, mid_y, x + w - t, mid_y + t); break;                  // middle
            case 4: push_solid(quads, x, y + h * 0.5f, x + t, y + h - t); break;                   // lower-left
            case 5: push_solid(quads, x + w - t, y + h * 0.5f, x + w, y + h - t); break;           // lower-right
            case 6: push_solid(quads, x + t, y + h - t, x + w - t, y + h); break;                  // bottom
            case 7: push_solid(quads, cx, y + t, cx + t, y + h - t); break;                        // center vertical
            case 8: push_solid(quads, x + w * 0.24f, y + h * 0.58f, x + w * 0.76f, y + h * 0.72f); break; // W notch
            default: break;
        }
    }

    static void DrawSegments(std::vector<sst::device::osd::OsdQuadRangle>* quads,
                             float x,
                             float y,
                             float scale,
                             const int* segs,
                             int count) {
        for (int i = 0; i < count; ++i) {
            Segment(quads, x, y, scale, segs[i]);
        }
    }

    static void DrawChar(std::vector<sst::device::osd::OsdQuadRangle>* quads,
                         float x,
                         float y,
                         float scale,
                         char ch) {
        switch (ch) {
            case 'A': { const int s[] = {0, 1, 2, 3, 4, 5}; DrawSegments(quads, x, y, scale, s, 6); break; }
            case 'B': { const int s[] = {0, 1, 3, 4, 5, 6}; DrawSegments(quads, x, y, scale, s, 6); break; }
            case 'C': { const int s[] = {0, 1, 4, 6}; DrawSegments(quads, x, y, scale, s, 4); break; }
            case 'D': { const int s[] = {0, 1, 2, 4, 5, 6}; DrawSegments(quads, x, y, scale, s, 6); break; }
            case 'E': { const int s[] = {0, 1, 3, 4, 6}; DrawSegments(quads, x, y, scale, s, 5); break; }
            case 'F': { const int s[] = {0, 1, 3, 4}; DrawSegments(quads, x, y, scale, s, 4); break; }
            case 'G': { const int s[] = {0, 1, 3, 4, 5, 6}; DrawSegments(quads, x, y, scale, s, 6); break; }
            case 'H': { const int s[] = {1, 2, 3, 4, 5}; DrawSegments(quads, x, y, scale, s, 5); break; }
            case 'I': { const int s[] = {0, 6, 7}; DrawSegments(quads, x, y, scale, s, 3); break; }
            case 'L': { const int s[] = {1, 4, 6}; DrawSegments(quads, x, y, scale, s, 3); break; }
            case 'M': { const int s[] = {1, 2, 4, 5, 7}; DrawSegments(quads, x, y, scale, s, 5); break; }
            case 'N': { const int s[] = {1, 2, 4, 5, 7}; DrawSegments(quads, x, y, scale, s, 5); break; }
            case 'O': { const int s[] = {0, 1, 2, 4, 5, 6}; DrawSegments(quads, x, y, scale, s, 6); break; }
            case 'P': { const int s[] = {0, 1, 2, 3, 4}; DrawSegments(quads, x, y, scale, s, 5); break; }
            case 'R': { const int s[] = {0, 1, 2, 3, 4, 5}; DrawSegments(quads, x, y, scale, s, 6); break; }
            case 'S': { const int s[] = {0, 1, 3, 5, 6}; DrawSegments(quads, x, y, scale, s, 5); break; }
            case 'T': { const int s[] = {0, 7}; DrawSegments(quads, x, y, scale, s, 2); break; }
            case 'U': { const int s[] = {1, 2, 4, 5, 6}; DrawSegments(quads, x, y, scale, s, 5); break; }
            case 'V': { const int s[] = {1, 2, 4, 5, 6}; DrawSegments(quads, x, y, scale, s, 5); break; }
            case 'W': { const int s[] = {1, 2, 4, 5, 6, 8}; DrawSegments(quads, x, y, scale, s, 6); break; }
            default: break;
        }
    }
};

std::string action_text(const std::string& action)
{
    if (action == "stop") return "STOP";
    if (action == "system_fault") return "STOP";
    if (action == "slow") return "SLOW";
    if (action == "turn_left") return "LEFT";
    if (action == "turn_right") return "RIGHT";
    return "CLEAR";
}

int zone_level(const ZoneStatus& zone)
{
    if (!zone.occupied) return 0;
    if (zone.risk_level == "urgent" || zone.risk_level == "near" ||
        (zone.distance_m >= 0.0f && zone.distance_m < 1.0f)) {
        return 3;
    }
    if (zone.risk_level == "warning" ||
        (zone.distance_m >= 0.0f && zone.distance_m < 2.0f)) {
        return 2;
    }
    return 1;
}

void draw_zone_meter(std::vector<sst::device::osd::OsdQuadRangle>* quads,
                     float x,
                     float y,
                     int level)
{
    const float w = 168.0f;
    const float h = 28.0f;
    push_hollow(quads, x, y, x + w, y + h, 3);
    if (level >= 1) push_solid(quads, x + 6.0f, y + 6.0f, x + 56.0f, y + h - 6.0f);
    if (level >= 2) push_solid(quads, x + 60.0f, y + 6.0f, x + 110.0f, y + h - 6.0f);
    if (level >= 3) push_solid(quads, x + 114.0f, y + 6.0f, x + w - 6.0f, y + h - 6.0f);
}

std::string dir_text(const std::string& dir)
{
    if (dir == "left") return "L";
    if (dir == "center") return "C";
    if (dir == "right") return "R";
    if (dir == "wide") return "WIDE";
    if (dir == "left_center") return "LC";
    if (dir == "center_right") return "CR";
    return "C";
}

std::string risk_text(float distance_m)
{
    if (distance_m < 0.0f) return "UNK";
    if (distance_m < 1.0f) return "NEAR";
    if (distance_m < 2.0f) return "WARN";
    return "FAR";
}

std::string hud_asset_path(const std::string& name)
{
    return "/app_demo/app_assets/osd/" + name + ".ssbmp";
}

std::string nearest_risk_text(const AvoidanceDecision& decision)
{
    const float nearest = nearest_distance_from_decision(decision);
    return risk_text(nearest);
}

}  // namespace

void VISUALIZER::Draw(const DetectionResult& result)
{
    AvoidanceDecision decision;
    Draw(result, decision);
}

void VISUALIZER::Draw(const DetectionResult& result, const AvoidanceDecision& decision)
{
    std::vector<sst::device::osd::OsdQuadRangle> box_quads;
    box_quads.reserve(result.items.size());

    // OSD layers:
    // layer 0 = disabled legacy risk bar, layer 1 = action .ssbmp,
    // layer 2 = dir/risk .ssbmp, layer 3 = kept empty,
    // layer 4 = detection boxes.
    if (!static_layers_cleaned_) {
        osd_device.CleanLayer(0);
        osd_device.CleanLayer(3);
        static_layers_cleaned_ = true;
    }

    const size_t max_display_boxes = std::min<size_t>(result.items.size(), 6);
    for (size_t i = 0; i < max_display_boxes; ++i) {
        const auto& item = result.items[i];

        sst::device::osd::OsdQuadRangle q;
        q.box = item.box;
        q.border = 4;
        q.layer_id = 0;
        q.type = fdevice::TYPE_HOLLOW;
        q.alpha = fdevice::TYPE_ALPHA75;
        q.color = PickColorByClass(item.class_id);

        box_quads.emplace_back(q);
    }

    const std::string action_name = action_text(decision.action);
    int primary_idx = find_primary_index(result);
    std::string dir_name = "C";
    std::string risk_name = "UNK";
    if (primary_idx >= 0) {
        dir_name = dir_text(result.items[primary_idx].sector);
        risk_name = risk_text(result.items[primary_idx].distance_m);
    }
    const std::string action_asset = hud_asset_path(action_name);
    const std::string info_asset = hud_asset_path(dir_name + "_" + risk_name);
    if (action_asset != last_action_asset_) {
        if (osd_device.DrawTexture(action_asset, 24, 36, 1)) {
            last_action_asset_ = action_asset;
        }
    }
    if (info_asset != last_info_asset_) {
        if (osd_device.DrawTexture(info_asset, 24, 128, 2)) {
            last_info_asset_ = info_asset;
        }
    }
    osd_device.Draw(box_quads, 4);
}
