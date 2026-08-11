#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <poll.h>
#include <unistd.h>

#include "include/tracker.hpp"
#include "include/utils.hpp"

#ifndef A1_ENABLE_SURFACE_SEG
#define A1_ENABLE_SURFACE_SEG 0
#endif

#if A1_ENABLE_SURFACE_SEG
#include "include/surface_fusion.hpp"
#include "include/surface_segmentation.hpp"
#endif

#ifndef A1_ENABLE_VOICE
#define A1_ENABLE_VOICE 0
#endif

#ifndef A1_MODEL_FILENAME
#define A1_MODEL_FILENAME "yolov8n80_graycopy_head6.m1model"
#endif

#ifndef A1_SEG_MODEL_FILENAME
#define A1_SEG_MODEL_FILENAME "graynav_surface_depth_gray1_int8.m1model"
#endif

#if A1_ENABLE_VOICE
#include "include/voice_notifier.hpp"
#endif

/*
 * obstacle_detect 主程序只负责编排模块，不重复实现算法：
 * IMAGEPROCESSOR 取全图 -> YOLOV8GRAY 单帧检测 -> ObstacleTracker 时序稳定/测距/
 * 规划 -> VISUALIZER 和 VoiceNotifier 并行输出。SystemHealth 可在任一环节覆盖正常
 * 决策为 system_fault，从而统一触发保护停下、OSD 告警和“异常”语音。
 */
constexpr bool kOutputJsonLines = false;
constexpr bool kOutputHumanSummary = true;
constexpr bool kOutputSerialDiagnostics = false;
constexpr int kOutputIntervalFrames = 10;

std::atomic<bool> g_exit_flag(false);

bool env_flag_enabled(const char* name, bool default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T';
}

int env_int_value(const char* name, int default_value, int min_value, int max_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return default_value;
    }
    if (parsed < min_value) {
        return min_value;
    }
    if (parsed > max_value) {
        return max_value;
    }
    return static_cast<int>(parsed);
}

float env_float_value(const char* name, float default_value, float min_value, float max_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return default_value;
    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value) return default_value;
    return std::max(min_value, std::min(parsed, max_value));
}

std::string env_string_value(const char* name, const std::string& default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return std::string(value);
}

struct FrameStats {
    float fps;
    float fps_avg;
    float frame_ms;
    float p95_ms;
    float jitter_pct;
    FrameStats() : fps(0.0f), fps_avg(0.0f), frame_ms(0.0f), p95_ms(0.0f), jitter_pct(0.0f) {}
};

struct LightStats {
    float mean;
    float stddev;
    float dark_ratio;
    float bright_ratio;
    float dynamic_range;
    float edge_ratio;
    float center_mean;
    float center_stddev;
    float center_dynamic_range;
    float center_edge_ratio;
    int cover_score;
    bool cover_candidate;
    std::string state;
    uint32_t sample_hash;

    LightStats()
        : mean(0.0f), stddev(0.0f), dark_ratio(0.0f), bright_ratio(0.0f),
          dynamic_range(0.0f), edge_ratio(0.0f), center_mean(0.0f),
          center_stddev(0.0f), center_dynamic_range(0.0f), center_edge_ratio(0.0f),
          cover_score(0), cover_candidate(false),
          state("unknown"), sample_hash(2166136261u) {}
};

/**
 * @brief 三类异常的统一锁存与恢复状态机。
 *
 * camera/data：连续取帧失败、遮挡/过曝/近乎纯色、画面冻结；
 * inference：NPU 调用或 head 校验连续失败；
 * resource：低 FPS、高 P95、低内存或候选爆炸。
 * 故障一旦锁存，必须连续一段可配置的健康帧才恢复，避免遮挡边缘短暂露光时
 * 误报 CLEAR。遮挡触发和恢复帧数分别由 A1_COVER_TRIGGER_FRAMES、
 * A1_COVER_RECOVERY_FRAMES 控制。
 */
struct SystemHealth {
    int capture_failures;
    int inference_failures;
    int data_fault_frames;
    int resource_fault_frames;
    int candidate_burst_frames;
    int frozen_frames;
    int low_memory_frames;
    int resource_checks;
    int healthy_recovery_frames;
    int cover_trigger_frames;
    int cover_recovery_frames;
    int last_cover_score;
    bool last_cover_candidate;
    bool fault_latched;
    int memory_available_kb;
    uint32_t last_image_hash;
    std::string state;
    std::string reason;

    SystemHealth()
        : capture_failures(0),
          inference_failures(0),
          data_fault_frames(0),
          resource_fault_frames(0),
          candidate_burst_frames(0),
          frozen_frames(0),
          low_memory_frames(0),
          resource_checks(0),
          healthy_recovery_frames(0),
          cover_trigger_frames(env_int_value("A1_COVER_TRIGGER_FRAMES", 3, 2, 30)),
          cover_recovery_frames(env_int_value("A1_COVER_RECOVERY_FRAMES", 18, 8, 90)),
          last_cover_score(0),
          last_cover_candidate(false),
          fault_latched(false),
          memory_available_kb(-1),
          last_image_hash(0),
          state("ok"),
          reason("ok") {}

    void UpdateData(const LightStats& light)
    {
        // cover_candidate 综合全图和中心区域的纹理、梯度及动态范围。它可以识别
        // 手掌贴近镜头时“非纯黑但大面积失焦”的遮挡，而不仅依赖暗像素比例。
        last_cover_score = light.cover_score;
        last_cover_candidate = light.cover_candidate;
        // 计数封顶，避免长时间遮挡后需要同样长时间才能恢复；故障锁存仍由
        // cover_recovery_frames 保证解除遮挡后经过稳定健康观测才恢复导航。
        data_fault_frames = light.cover_candidate
            ? std::min(cover_trigger_frames + 4, data_fault_frames + 1)
            : std::max(0, data_fault_frames - 2);
        if (last_image_hash != 0 && light.sample_hash == last_image_hash) {
            ++frozen_frames;
        } else {
            frozen_frames = 0;
        }
        last_image_hash = light.sample_hash;
    }

    void UpdateResource(const FrameStats& frame_stats, const DetectionResult& result)
    {
        const bool slow = (frame_stats.fps_avg > 0.1f && frame_stats.fps_avg < 3.0f) ||
                          frame_stats.p95_ms > 600.0f;
        resource_fault_frames = slow ? resource_fault_frames + 1 : 0;
        candidate_burst_frames = result.raw_candidate_count > 1400 ? candidate_burst_frames + 1 : 0;
        ++resource_checks;
        if (resource_checks % 30 == 0) {
            std::ifstream meminfo("/proc/meminfo");
            std::string key;
            int value = -1;
            std::string unit;
            while (meminfo >> key >> value >> unit) {
                if (key == "MemAvailable:") {
                    memory_available_kb = value;
                    break;
                }
            }
            low_memory_frames = memory_available_kb >= 0 && memory_available_kb < 8192
                ? low_memory_frames + 1 : 0;
        }
    }

    bool FaultActive() const
    {
        return fault_latched ||
               capture_failures >= 3 ||
               inference_failures >= 2 ||
               data_fault_frames >= cover_trigger_frames ||
               frozen_frames >= 15 ||
               resource_fault_frames >= 20 ||
               low_memory_frames >= 3 ||
               candidate_burst_frames >= 5;
    }

    AvoidanceDecision SafeDecision() const
    {
        // system_fault 是跨模块约定的最高优先级动作，语音层会映射为“异常”。
        AvoidanceDecision decision;
        decision.action = "system_fault";
        decision.prompt = "reason=system_health " + reason;
        return decision;
    }

    void RefreshState()
    {
        const bool raw_fault = capture_failures >= 3 || inference_failures >= 2 ||
                               data_fault_frames >= cover_trigger_frames || frozen_frames >= 15 ||
                               resource_fault_frames >= 20 || low_memory_frames >= 3 ||
                               candidate_burst_frames >= 5;
        if (raw_fault) {
            fault_latched = true;
            healthy_recovery_frames = 0;
        } else if (fault_latched) {
            const bool recovery_clean = capture_failures == 0 && inference_failures == 0 &&
                                        data_fault_frames == 0 && frozen_frames == 0 &&
                                        resource_fault_frames == 0 && low_memory_frames == 0 &&
                                        candidate_burst_frames == 0;
            healthy_recovery_frames = recovery_clean ? healthy_recovery_frames + 1 : 0;
            // 手移开镜头的瞬间可能只露出一道亮边，使单帧统计暂时恢复正常。
            // 因此必须连续获得足够多健康帧后，才解除故障并重新允许正常导航播报。
            if (healthy_recovery_frames >= cover_recovery_frames) {
                fault_latched = false;
                healthy_recovery_frames = 0;
            }
        }

        if (capture_failures >= 3) {
            state = "sensor";
            reason = "capture_failed";
        } else if (inference_failures >= 2) {
            state = "ai";
            reason = "inference_failed";
        } else if (frozen_frames >= 15) {
            state = "sensor";
            reason = "frozen_frame";
        } else if (data_fault_frames >= cover_trigger_frames) {
            state = "sensor";
            reason = "bad_image";
        } else if (resource_fault_frames >= 20) {
            state = "resource";
            reason = "low_fps";
        } else if (low_memory_frames >= 3) {
            state = "resource";
            reason = "low_memory";
        } else if (candidate_burst_frames >= 5) {
            state = "resource";
            reason = "candidate_burst";
        } else if (!fault_latched) {
            state = "ok";
            reason = "ok";
        }
    }
};

/** 维护最近 120 个帧周期，输出平均 FPS、P95 延迟和帧间波动。 */
class RuntimeMeter {
public:
    RuntimeMeter() : initialized_(false), fps_avg_(0.0f) { intervals_ms_.reserve(120); }

    FrameStats Tick()
    {
        const auto now = std::chrono::steady_clock::now();
        FrameStats stats;
        if (!initialized_) {
            last_ = now;
            initialized_ = true;
            return stats;
        }

        const std::chrono::duration<float> dt = now - last_;
        last_ = now;
        if (dt.count() > 1e-4f) {
            stats.fps = 1.0f / dt.count();
            stats.frame_ms = dt.count() * 1000.0f;
            fps_avg_ = (fps_avg_ <= 0.0f) ? stats.fps : (0.85f * fps_avg_ + 0.15f * stats.fps);
            stats.fps_avg = fps_avg_;
            intervals_ms_.push_back(stats.frame_ms);
            if (intervals_ms_.size() > 120) intervals_ms_.erase(intervals_ms_.begin());
            std::vector<float> ordered = intervals_ms_;
            std::sort(ordered.begin(), ordered.end());
            if (!ordered.empty()) {
                const size_t p95_index = static_cast<size_t>(0.95f * (ordered.size() - 1));
                stats.p95_ms = ordered[p95_index];
                const float mean_ms = 1000.0f / std::max(0.1f, fps_avg_);
                stats.jitter_pct = 100.0f * std::fabs(stats.p95_ms - mean_ms) /
                                   std::max(1.0f, mean_ms);
            }
        }
        return stats;
    }

private:
    bool initialized_;
    float fps_avg_;
    std::vector<float> intervals_ms_;
    std::chrono::steady_clock::time_point last_;
};

LightStats analyze_light_stats(ssne_tensor_t* img)
{
    LightStats stats;
    if (img == nullptr) {
        return stats;
    }

    const int w = static_cast<int>(get_width(*img));
    const int h = static_cast<int>(get_height(*img));
    const uint8_t* data = reinterpret_cast<const uint8_t*>(get_data(*img));
    const uint32_t total_size = get_total_size(*img);
    if (w <= 0 || h <= 0 || data == nullptr || total_size < static_cast<uint32_t>(w * h)) {
        return stats;
    }

    const int step = 8;
    int count = 0;
    int dark = 0;
    int bright = 0;
    int edge_count = 0;
    int edge_samples = 0;
    int center_count = 0;
    int center_edge_count = 0;
    int center_edge_samples = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double center_sum = 0.0;
    double center_sum_sq = 0.0;
    std::array<int, 256> histogram = {};
    std::array<int, 256> center_histogram = {};
    const int center_x0 = static_cast<int>(0.18f * w);
    const int center_x1 = static_cast<int>(0.82f * w);
    const int center_y0 = static_cast<int>(0.15f * h);
    const int center_y1 = static_cast<int>(0.85f * h);

    for (int y = 0; y < h; y += step) {
        for (int x = 0; x < w; x += step) {
            const int v = data[y * w + x];
            stats.sample_hash ^= static_cast<uint32_t>(v + ((x & 0xFF) << 8) + ((y & 0xFF) << 16));
            stats.sample_hash *= 16777619u;
            sum += v;
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
            histogram[static_cast<size_t>(v)]++;
            if (v < 35) dark++;
            if (v > 220) bright++;
            int gradient = 0;
            if (x + step < w) gradient += std::abs(v - static_cast<int>(data[y * w + x + step]));
            if (y + step < h) gradient += std::abs(v - static_cast<int>(data[(y + step) * w + x]));
            if (x + step < w || y + step < h) {
                edge_count += gradient >= 28 ? 1 : 0;
                edge_samples++;
            }
            if (x >= center_x0 && x < center_x1 && y >= center_y0 && y < center_y1) {
                center_sum += v;
                center_sum_sq += static_cast<double>(v) * static_cast<double>(v);
                center_histogram[static_cast<size_t>(v)]++;
                center_count++;
                if (x + step < w || y + step < h) {
                    center_edge_count += gradient >= 24 ? 1 : 0;
                    center_edge_samples++;
                }
            }
            count++;
        }
    }

    if (count <= 0) {
        return stats;
    }

    stats.mean = static_cast<float>(sum / count);
    const double variance = std::max(0.0, sum_sq / count - static_cast<double>(stats.mean) * stats.mean);
    stats.stddev = static_cast<float>(std::sqrt(variance));
    stats.dark_ratio = static_cast<float>(dark) / static_cast<float>(count);
    stats.bright_ratio = static_cast<float>(bright) / static_cast<float>(count);
    stats.edge_ratio = edge_samples > 0
        ? static_cast<float>(edge_count) / static_cast<float>(edge_samples) : 0.0f;

    const auto percentile = [](const std::array<int, 256>& hist, int total, float q) {
        if (total <= 0) return 0;
        const int target = std::max(1, static_cast<int>(std::ceil(q * total)));
        int accumulated = 0;
        for (int i = 0; i < 256; ++i) {
            accumulated += hist[static_cast<size_t>(i)];
            if (accumulated >= target) return i;
        }
        return 255;
    };
    stats.dynamic_range = static_cast<float>(
        percentile(histogram, count, 0.95f) - percentile(histogram, count, 0.05f));

    if (center_count > 0) {
        stats.center_mean = static_cast<float>(center_sum / center_count);
        const double center_variance = std::max(
            0.0, center_sum_sq / center_count -
            static_cast<double>(stats.center_mean) * stats.center_mean);
        stats.center_stddev = static_cast<float>(std::sqrt(center_variance));
        stats.center_dynamic_range = static_cast<float>(
            percentile(center_histogram, center_count, 0.95f) -
            percentile(center_histogram, center_count, 0.05f));
        stats.center_edge_ratio = center_edge_samples > 0
            ? static_cast<float>(center_edge_count) /
              static_cast<float>(center_edge_samples) : 0.0f;
    }

    // 多证据遮挡评分：全图低纹理、中心低纹理和灰度分布压缩分别计分；
    // 大面积暗/亮饱和额外计分。这样手掌、衣物等非纯黑遮挡也能被识别。
    int cover_score = 0;
    cover_score += stats.stddev < 10.0f ? 2 : (stats.stddev < 18.0f ? 1 : 0);
    cover_score += stats.dynamic_range < 35.0f ? 2 : (stats.dynamic_range < 60.0f ? 1 : 0);
    cover_score += stats.edge_ratio < 0.015f ? 2 : (stats.edge_ratio < 0.035f ? 1 : 0);
    cover_score += stats.center_stddev < 14.0f ? 2 : (stats.center_stddev < 22.0f ? 1 : 0);
    cover_score += stats.center_dynamic_range < 42.0f ? 2 :
                   (stats.center_dynamic_range < 70.0f ? 1 : 0);
    cover_score += stats.center_edge_ratio < 0.020f ? 2 :
                   (stats.center_edge_ratio < 0.045f ? 1 : 0);
    if (stats.dark_ratio > 0.65f || stats.bright_ratio > 0.65f) cover_score += 2;

    const bool hard_dark_cover = stats.mean < 55.0f && stats.dark_ratio > 0.65f &&
                                 stats.stddev < 20.0f;
    const bool hard_bright_cover = stats.mean > 215.0f && stats.bright_ratio > 0.70f &&
                                   stats.stddev < 20.0f;
    const bool hard_flat_frame = stats.stddev > 0.0f && stats.stddev < 5.0f;
    const bool center_occluded = stats.center_stddev < 14.0f &&
                                 stats.center_dynamic_range < 48.0f &&
                                 stats.center_edge_ratio < 0.025f;
    static const int score_threshold =
        env_int_value("A1_COVER_SCORE_THRESHOLD", 5, 3, 12);
    stats.cover_score = cover_score;
    stats.cover_candidate = hard_dark_cover || hard_bright_cover || hard_flat_frame ||
                            center_occluded || cover_score >= score_threshold;

    if (stats.cover_candidate) {
        stats.state = "covered";
    } else if (stats.stddev < 18.0f) {
        stats.state = "low_contrast";
    } else if (stats.mean < 55.0f || stats.dark_ratio > 0.45f) {
        stats.state = "dark";
    } else if (stats.mean > 195.0f || stats.bright_ratio > 0.35f) {
        stats.state = "bright";
    } else {
        stats.state = "normal";
    }
    return stats;
}

void keyboard_listener()
{
    std::cout << "[INFO] keyboard listener started, input q to exit." << std::endl;
    struct pollfd pfd;
    pfd.fd = STDIN_FILENO;
    pfd.events = POLLIN;
    pfd.revents = 0;
    while (!g_exit_flag.load()) {
        const int ready = poll(&pfd, 1, 200);
        if (ready > 0 && (pfd.revents & POLLIN)) {
            char input[16] = {0};
            const ssize_t count = read(STDIN_FILENO, input, sizeof(input));
            if (count > 0 && (input[0] == 'q' || input[0] == 'Q')) {
                g_exit_flag.store(true);
                std::cout << "[INFO] exit command received." << std::endl;
                break;
            }
        }
        if (ready < 0) {
            usleep(200000);
        }
    }
}

bool check_exit_flag()
{
    return g_exit_flag.load();
}

std::string json_escape(const std::string& s)
{
    std::ostringstream oss;
    for (size_t i = 0; i < s.size(); ++i) {
        const char ch = s[i];
        switch (ch) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << ch; break;
        }
    }
    return oss.str();
}

void append_distance(std::ostringstream& oss, float distance_m)
{
    if (distance_m < 0.0f) {
        oss << "-1";
        return;
    }

    oss << std::fixed << std::setprecision(2) << distance_m;
    oss.unsetf(std::ios::floatfield);
}

void append_score(std::ostringstream& oss, float score)
{
    oss << std::fixed << std::setprecision(3) << score;
    oss.unsetf(std::ios::floatfield);
}

void append_float_or_null(std::ostringstream& oss, float value)
{
    if (value < 0.0f) {
        oss << "null";
        return;
    }

    oss << std::fixed << std::setprecision(2) << value;
    oss.unsetf(std::ios::floatfield);
}

void append_zone(std::ostringstream& oss, const char* key, const ZoneStatus& zone)
{
    oss << "\"" << key << "\":";
    if (!zone.occupied) {
        oss << "{\"occupied\":false,\"dist_m\":-1,\"risk\":\"unknown\",\"label\":\"\"}";
        return;
    }

    oss << "{\"occupied\":true";
    oss << ",\"label\":\"" << json_escape(zone.label) << "\"";
    oss << ",\"semantic_class\":\"" << json_escape(zone.semantic_class) << "\"";
    oss << ",\"dist_m\":";
    append_distance(oss, zone.distance_m);
    oss << ",\"risk\":\"" << json_escape(zone.risk_level) << "\"}";
}

void print_json_packet(int frame_id,
                       const DetectionResult& result,
                       const AvoidanceDecision& decision,
                       const SurfaceResult& surface)
{
    int nearest_idx = -1;
    float nearest_dist = 1e9f;

    for (size_t i = 0; i < result.items.size(); ++i) {
        const DetectionItem& item = result.items[i];
        if (item.distance_m >= 0.0f && item.distance_m < nearest_dist) {
            nearest_dist = item.distance_m;
            nearest_idx = static_cast<int>(i);
        }
    }

    std::ostringstream oss;
    oss << "{\"type\":\"obstacle\",\"frame\":" << frame_id << ",\"objects\":[";

    for (size_t i = 0; i < result.items.size(); ++i) {
        const DetectionItem& item = result.items[i];
        if (i > 0) {
            oss << ",";
        }

        oss << "{\"dir\":\"" << json_escape(item.sector) << "\"";
        oss << ",\"track\":" << item.track_id;
        oss << ",\"label\":\"" << json_escape(item.label) << "\"";
        oss << ",\"semantic_class\":\"" << json_escape(item.semantic_class) << "\"";
        oss << ",\"raw_label\":\"" << json_escape(item.raw_label) << "\"";
        oss << ",\"raw_cls\":" << item.raw_class_id;
        oss << ",\"conf\":";
        append_score(oss, item.score);
        oss << ",\"dist_m\":";
        append_distance(oss, item.distance_m);
        oss << ",\"safe_dist_m\":";
        append_distance(oss, item.safe_distance_m);
        oss << ",\"lateral_m\":" << std::fixed << std::setprecision(2) << item.lateral_m;
        oss << ",\"dist_src\":\"" << json_escape(item.distance_source) << "\"";
        oss << ",\"risk\":\"" << json_escape(item.risk_level) << "\"";
        oss << ",\"approach_mps\":";
        append_float_or_null(oss, item.approach_mps);
        oss << ",\"ttc_s\":";
        append_float_or_null(oss, item.ttc_s);
        oss << ",\"depth_level\":\"" << json_escape(item.depth_level) << "\"";
        oss << ",\"depth_confidence\":" << std::fixed << std::setprecision(3)
            << item.depth_confidence;
        oss << ",\"depth_source\":\"" << json_escape(item.depth_source) << "\"";
        oss << ",\"depth_consistent\":" << (item.depth_consistent ? "true" : "false");
        oss << ",\"approaching\":" << (item.approaching ? "true" : "false");
        oss << ",\"box\":["
            << static_cast<int>(std::round(item.box[0])) << ","
            << static_cast<int>(std::round(item.box[1])) << ","
            << static_cast<int>(std::round(item.box[2])) << ","
            << static_cast<int>(std::round(item.box[3])) << "]}";
    }

    oss << "],\"nearest\":";
    if (nearest_idx >= 0) {
        const DetectionItem& item = result.items[nearest_idx];
        oss << "{\"dir\":\"" << json_escape(item.sector) << "\"";
        oss << ",\"label\":\"" << json_escape(item.label) << "\"";
        oss << ",\"semantic_class\":\"" << json_escape(item.semantic_class) << "\"";
        oss << ",\"track\":" << item.track_id;
        oss << ",\"dist_m\":";
        append_distance(oss, item.distance_m);
        oss << ",\"risk\":\"" << json_escape(item.risk_level) << "\"}";
    } else {
        oss << "null";
    }
    oss << ",\"zones\":{";
    append_zone(oss, "left", decision.left);
    oss << ",";
    append_zone(oss, "center", decision.center);
    oss << ",";
    append_zone(oss, "right", decision.right);
    oss << "}";
    oss << ",\"nav\":{\"action\":\"" << json_escape(decision.action) << "\"";
    oss << ",\"sector\":\"" << (nearest_idx >= 0 ? json_escape(result.items[nearest_idx].sector) : "clear") << "\"";
    oss << ",\"prompt\":\"" << json_escape(decision.prompt) << "\"";
    oss << ",\"nearest_track\":" << decision.nearest_track_id;
    oss << ",\"hazard_type\":\"" << json_escape(decision.hazard_type) << "\"";
    oss << ",\"hazard_sector\":\"" << json_escape(decision.hazard_sector) << "\"";
    oss << ",\"perception_source\":\"" << json_escape(decision.perception_source) << "\"";
    oss << ",\"surface_confidence\":" << std::fixed << std::setprecision(3)
        << decision.surface_confidence;
    oss << ",\"perception_degraded\":" << (decision.perception_degraded ? "true" : "false");
    oss << ",\"depth_level\":\"" << json_escape(decision.depth_level) << "\"";
    oss << ",\"depth_confidence\":" << decision.depth_confidence;
    oss << ",\"depth_margin\":" << decision.depth_margin;
    oss << ",\"depth_ambiguous\":" << (decision.depth_ambiguous ? "true" : "false");
    oss << ",\"depth_source\":\"" << json_escape(decision.depth_source) << "\"";
    oss << ",\"depth_consistent\":" << (decision.depth_consistent ? "true" : "false");
    oss << ",\"approaching\":" << (decision.approaching ? "true" : "false") << "}";
    const auto append_surface = [&oss](const char* name, const SurfaceCorridor& corridor) {
        oss << "\"" << name << "\":{";
        oss << "\"ground\":" << std::fixed << std::setprecision(3) << corridor.ground_ratio;
        oss << ",\"blocked\":" << corridor.blocked_ratio;
        oss << ",\"step\":" << corridor.step_ratio;
        oss << ",\"unknown\":" << corridor.unknown_ratio;
        oss << ",\"safe_candidate\":" << (corridor.safe_candidate ? "true" : "false");
        oss << ",\"persistent_hazard\":" << (corridor.persistent_hazard ? "true" : "false") << "}";
    };
    oss << ",\"surface\":{\"valid\":" << (surface.valid ? "true" : "false")
        << ",\"stale\":" << (surface.stale ? "true" : "false")
        << ",\"timestamp_ms\":" << surface.timestamp_ms
        << ",\"proximity\":\"" << json_escape(surface.proximity) << "\""
        << ",\"hazard\":\"" << json_escape(surface.primary_hazard) << "\""
        << ",\"sector\":\"" << json_escape(surface.primary_sector) << "\""
        << ",\"depth_level\":\"" << json_escape(surface.depth_level) << "\""
        << ",\"depth_confidence\":" << surface.depth_confidence
        << ",\"depth_margin\":" << surface.depth_margin
        << ",\"depth_ambiguous\":" << (surface.depth_ambiguous ? "true" : "false")
        << ",\"depth_group_probabilities\":["
        << surface.depth_group_probabilities[0] << ","
        << surface.depth_group_probabilities[1] << ","
        << surface.depth_group_probabilities[2] << "]"
        << ",\"depth_source\":\"" << json_escape(surface.depth_source) << "\""
        << ",\"approaching\":" << (surface.approaching ? "true" : "false") << ",";
    append_surface("left", surface.left);
    oss << ",";
    append_surface("center", surface.center);
    oss << ",";
    append_surface("right", surface.right);
    oss << "}";
    oss << "}";

    std::cout << oss.str() << std::endl;
}

std::string to_upper_text(const std::string& text)
{
    std::string out = text;
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[i])));
    }
    return out;
}

std::string action_display_text(const std::string& action)
{
    if (action == "turn_left") return "LEFT";
    if (action == "turn_right") return "RIGHT";
    return to_upper_text(action);
}

// 将规划器动作拆成“速度建议”和“方向建议”，使演示串口无需解释内部枚举即可阅读。
void action_guidance_text(const std::string& action,
                          std::string& speed,
                          std::string& direction)
{
    if (action == "stop" || action == "system_fault") {
        speed = "STOP";
        direction = "HOLD";
    } else if (action == "slow") {
        speed = "SLOW";
        direction = "STRAIGHT";
    } else if (action == "turn_left") {
        speed = "SLOW";
        direction = "LEFT";
    } else if (action == "turn_right") {
        speed = "SLOW";
        direction = "RIGHT";
    } else {
        speed = "NORMAL";
        direction = "STRAIGHT";
    }
}

// 将内部健康状态映射为稳定的三类公开异常名称，直接对应赛题异常处理评分项。
std::string fault_type_text(const SystemHealth& health)
{
    if (health.state == "sensor") return "CAMERA_DATA";
    if (health.state == "ai") return "INFERENCE";
    if (health.state == "resource") return "RESOURCE";
    return "SYSTEM";
}

std::string fault_reason_text(const std::string& reason)
{
    if (reason == "capture_failed") return "FRAME_CAPTURE_FAILED";
    if (reason == "frozen_frame") return "FRAME_FROZEN";
    if (reason == "bad_image") return "LENS_BLOCKED_OR_INVALID_IMAGE";
    if (reason == "inference_failed") return "MODEL_INFERENCE_FAILED";
    if (reason == "low_fps") return "PROCESSING_TIMEOUT";
    if (reason == "low_memory") return "LOW_MEMORY";
    if (reason == "candidate_burst") return "ABNORMAL_DETECTION_OUTPUT";
    return to_upper_text(reason);
}

// 输出一条简洁故障记录；详细计数仅在 A1_OUTPUT_SERIAL_DIAG=1 时打印，
// 正常演示界面只保留故障类型、保护动作、语音和恢复策略等关键信息。
void print_fault_packet(int frame_id,
                        const SystemHealth& health,
                        bool output_serial_diagnostics)
{
    std::ostringstream oss;
    oss << "[FAULT] frame=" << frame_id
        << " status=ACTIVE"
        << " type=" << fault_type_text(health)
        << " reason=" << fault_reason_text(health.reason)
        << " protection=STOP"
        << " voice=ABNORMAL"
        << " recovery=AUTO_MONITORING";
    if (output_serial_diagnostics) {
        oss << " capture_fail=" << health.capture_failures
            << " infer_fail=" << health.inference_failures
            << " bad_image=" << health.data_fault_frames
            << " cover_score=" << health.last_cover_score
            << " cover_candidate=" << (health.last_cover_candidate ? 1 : 0)
            << " frozen=" << health.frozen_frames
            << " slow=" << health.resource_fault_frames
            << " low_mem=" << health.low_memory_frames
            << " candidate_burst=" << health.candidate_burst_frames;
    }
    std::cout << oss.str() << std::endl;
}

int find_nearest_index(const DetectionResult& result)
{
    int nearest_idx = -1;
    float nearest_dist = 1e9f;
    for (size_t i = 0; i < result.items.size(); ++i) {
        const DetectionItem& item = result.items[i];
        if (item.distance_m >= 0.0f && item.distance_m < nearest_dist) {
            nearest_dist = item.distance_m;
            nearest_idx = static_cast<int>(i);
        }
    }

    if (nearest_idx < 0 && !result.items.empty()) {
        nearest_idx = 0;
    }
    return nearest_idx;
}

void print_human_packet(int frame_id,
                        const DetectionResult& result,
                        const AvoidanceDecision& decision,
                        const SurfaceResult& surface,
                        const FrameStats& frame_stats,
                        const LightStats& light_stats,
                        int displayed_count,
                        bool output_serial_diagnostics)
{
    auto append_diagnostics = [&](std::ostringstream& oss, const DetectionItem* item) {
        if (!output_serial_diagnostics) {
            return;
        }
        oss << " light=" << light_stats.state
            << " view=" << result.view_id
            << " fps=" << std::fixed << std::setprecision(1) << frame_stats.fps_avg
            << " p95_ms=" << frame_stats.p95_ms
            << " jitter_pct=" << frame_stats.jitter_pct
            << " raw_candidates=" << result.raw_candidate_count
            << " post_nms=" << result.post_nms_count
            << " displayed=" << displayed_count
            << " tracks=" << result.Size()
            << " coarse_drop=" << result.coarse_drop_count
            << " depth=" << to_upper_text(decision.depth_level)
            << " depth_conf=" << std::setprecision(2) << decision.depth_confidence
            << " depth_margin=" << decision.depth_margin
            << " depth_ambiguous=" << (decision.depth_ambiguous ? 1 : 0)
            << " surface_center="
            << std::setprecision(2) << surface.center.ground_ratio << "/"
            << surface.center.blocked_ratio << "/"
            << surface.center.step_ratio << "/"
            << surface.center.unknown_ratio;
        if (item != nullptr) {
            oss << " conf=" << std::fixed << std::setprecision(2) << item->score
                << " src=" << item->distance_source
                << " q=" << item->quality
                << " raw_label=" << item->raw_label
                << " box="
                << static_cast<int>(std::round(item->box[0])) << ","
                << static_cast<int>(std::round(item->box[1])) << ","
                << static_cast<int>(std::round(item->box[2])) << ","
                << static_cast<int>(std::round(item->box[3]));
        }
    };
    std::string speed;
    std::string direction;
    action_guidance_text(decision.action, speed, direction);
    const int nearest_idx = find_nearest_index(result);
    if (nearest_idx < 0) {
        const char* decision_risk = "CLEAR";
        if (decision.action == "system_fault") {
            decision_risk = "FAULT";
        } else if (decision.action == "stop") {
            decision_risk = "EMERGENCY";
        } else if (decision.hazard_type != "none" ||
                   decision.action != "clear") {
            decision_risk = "WARNING";
        } else if (decision.perception_degraded) {
            decision_risk = "UNKNOWN";
        }
        std::ostringstream oss;
        oss << "[NAV] frame=" << frame_id
            << " speed=" << speed
            << " direction=" << direction
            << " obstacle=NONE"
            << " distance=--"
            << " risk=" << decision_risk
            << " hazard=" << to_upper_text(decision.hazard_type)
            << " hazard_sector=" << to_upper_text(decision.hazard_sector)
            << " perception=" << to_upper_text(decision.perception_source)
            << " depth=" << to_upper_text(decision.depth_level)
            << " degraded=" << (decision.perception_degraded ? 1 : 0);
        append_diagnostics(oss, nullptr);
        std::cout << oss.str() << std::endl;
        return;
    }

    const DetectionItem& item = result.items[nearest_idx];

    std::ostringstream oss;
    oss << "[NAV] frame=" << frame_id
        << " speed=" << speed
        << " direction=" << direction
        << " obstacle=" << item.label
        << " sector=" << to_upper_text(item.sector)
        << " distance=";
    if (item.distance_m >= 0.0f) {
        oss << std::fixed << std::setprecision(2) << item.distance_m << "m";
    } else {
        oss << "unknown";
    }
    oss << " risk=" << to_upper_text(item.risk_level);
    oss << " hazard=" << to_upper_text(decision.hazard_type)
        << " hazard_sector=" << to_upper_text(decision.hazard_sector)
        << " perception=" << to_upper_text(decision.perception_source)
        << " depth=" << to_upper_text(decision.depth_level)
        << " degraded=" << (decision.perception_degraded ? 1 : 0);
    if (result.items.size() > 1) {
        oss << " objects=" << result.items.size();
    }
    append_diagnostics(oss, &item);

    std::cout << oss.str() << std::endl;
}

int main()
{
    /*
     * 启动阶段从环境变量读取可调参数并依次初始化 SSNE、采集、模型、跟踪、OSD、
     * 语音。所有对象在主循环外构造，避免逐帧重复申请硬件和模型资源。
     */
    int full_width = env_int_value("A1_FULL_FRAME_WIDTH", 720, 1, 4096);
    int full_height = env_int_value("A1_FULL_FRAME_HEIGHT", 1280, 1, 4096);
    int capture_width = env_int_value("A1_CAPTURE_WIDTH", 720, 1, full_width);
    int capture_height = env_int_value("A1_CAPTURE_HEIGHT", full_height, 1, full_height);

    std::array<int, 2> det_shape = {384, 384};
    std::string path_det = env_string_value(
        "A1_MODEL_PATH",
        std::string("/app_demo/app_assets/models/") + A1_MODEL_FILENAME);
    const std::array<int, 2> seg_shape = {256, 256};
    const std::string path_seg = env_string_value(
        "A1_SEG_MODEL_PATH",
        std::string("/app_demo/app_assets/models/") + A1_SEG_MODEL_FILENAME);

    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }

    std::array<int, 2> full_shape = {full_width, full_height};
    std::array<int, 2> capture_shape = {capture_width, capture_height};

    YOLOV8GRAY detector;
    detector.Initialize(path_det, &capture_shape, &det_shape);

    SurfaceResult surface_result;
    bool surface_available = false;
    bool surface_degraded = true;
    int surface_failures = 0;
    bool surface_degraded_announced = false;
#if A1_ENABLE_SURFACE_SEG
    obstacle::SurfaceSegmenter surface_segmenter;
    obstacle::SurfaceDecisionFusion surface_fusion;
    surface_available = surface_segmenter.Initialize(path_seg, full_shape, seg_shape);
    if (surface_available && detector.ModelId() == surface_segmenter.ModelId()) {
        std::cout << "[SURFACE][ERROR] detector and segmenter returned the same model_id="
                  << detector.ModelId() << "; disabling surface perception" << std::endl;
        surface_available = false;
    }
    const bool surface_model_present = std::ifstream(path_seg.c_str(), std::ios::binary).good();
    if (!surface_available && surface_model_present) {
        std::cout << "[SURFACE][WARN] retrying dual-model load with both models dynamic" << std::endl;
        surface_segmenter.Release();
        detector.Release();
        ssne_release();
        if (ssne_initial()) {
            std::cout << "[SURFACE][ERROR] SSNE reinitialization failed; detector startup aborted"
                      << std::endl;
            return -1;
        }
        setenv("A1_YOLO_DYNAMIC_ALLOC", "1", 1);
        detector.Initialize(path_det, &capture_shape, &det_shape);
        surface_available = surface_segmenter.Initialize(path_seg, full_shape, seg_shape);
        if (!surface_available || detector.ModelId() == surface_segmenter.ModelId()) {
            surface_available = false;
            std::cout << "[SURFACE][ERROR] dual dynamic load failed; use the predefined "
                      << "Fast-SCNN-0.75 profile (target <=0.9 MiB). Detector-only fallback active."
                      << std::endl;
        }
    }
    surface_degraded = !surface_available;
    surface_result.perception_degraded = surface_degraded;
#endif

    IMAGEPROCESSOR processor;
    processor.Initialize(&capture_shape);

    DetectionResult* det_result = new DetectionResult;
    DetectionResult empty_result;
    RuntimeMeter runtime_meter;
    FrameStats frame_stats;
    LightStats light_stats;
    SystemHealth system_health;
    obstacle::ObstacleTracker tracker;
    tracker.Initialize(full_shape);

    VISUALIZER visualizer;
    visualizer.Initialize(full_shape);

#if A1_ENABLE_VOICE
    obstacle::VoiceNotifier voice_notifier;
    voice_notifier.InitializeFromEnv();
#endif

    std::cout << "sleep for 0.2 second!" << std::endl;
    usleep(200000);

    ssne_tensor_t img_sensor = {};

    std::thread listener_thread(keyboard_listener);

    int frame_id = 0;
    int capture_failures = 0;
    int exit_code = 0;
    const bool output_json_lines = env_flag_enabled("A1_OUTPUT_JSON", kOutputJsonLines);
    const bool output_human_summary = env_flag_enabled("A1_OUTPUT_HUMAN", kOutputHumanSummary);
    const bool output_serial_diagnostics = env_flag_enabled("A1_OUTPUT_SERIAL_DIAG", kOutputSerialDiagnostics);
    const bool capture_auto_restart = env_flag_enabled("A1_CAPTURE_AUTO_RESTART", true);
    const int output_interval_frames = env_int_value("A1_OUTPUT_INTERVAL_FRAMES",
                                                     kOutputIntervalFrames,
                                                     1,
                                                     300);
    const int osd_interval_frames = env_int_value("A1_OSD_INTERVAL_FRAMES", 2, 1, 10);
    const int perf_interval_frames = env_int_value("A1_PERF_INTERVAL_FRAMES", 60, 10, 600);
    const int surface_period = env_int_value("A1_SURFACE_PERIOD", 4, 2, 30);
    const int surface_slot = env_int_value("A1_SURFACE_SLOT", 3, 0, surface_period - 1);
    const int surface_stale_ms = env_int_value("A1_SURFACE_STALE_MS", 1500, 200, 10000);
    const int sensor_fps = env_int_value("A1_SENSOR_FPS", 90, 1, 240);
    const std::string test_fault_type = env_string_value("A1_TEST_FAULT_TYPE", "none");
    const int test_fault_start = env_int_value("A1_TEST_FAULT_START_FRAME", 120, 1, 1000000);
    const int test_fault_duration = env_int_value("A1_TEST_FAULT_DURATION_FRAMES", 180, 1, 1000000);
    std::string last_osd_action;
    bool last_fault_active = false;
    std::string last_fault_reason;
    std::string last_fault_type = "SYSTEM";

    std::cout << "====================================================" << std::endl;
    std::cout << "[INFO] obstacle_detect demo started." << std::endl;
    std::cout << "[INFO] full image shape    = [" << full_shape[0] << ", " << full_shape[1] << "]" << std::endl;
    std::cout << "[INFO] capture image shape = [" << capture_shape[0] << ", " << capture_shape[1] << "]" << std::endl;
    std::cout << "[INFO] det input shape = [" << det_shape[0] << ", " << det_shape[1] << "]" << std::endl;
    std::cout << "[INFO] model path      = " << path_det << std::endl;
    std::cout << "[INFO] surface model   = " << path_seg << std::endl;
    std::cout << "[INFO] surface status  = " << (surface_available ? "ready" : "detector-only")
              << " schedule=" << surface_period << ":" << surface_slot << std::endl;
    std::cout << "[INFO] output json     = " << (output_json_lines ? "on" : "off") << std::endl;
    std::cout << "[INFO] output human    = " << (output_human_summary ? "on" : "off") << std::endl;
    std::cout << "[INFO] output diag     = " << (output_serial_diagnostics ? "on" : "off") << std::endl;
    std::cout << "[INFO] output interval = " << output_interval_frames << " frames" << std::endl;
    std::cout << "[INFO] OSD interval    = " << osd_interval_frames << " frames" << std::endl;
    std::cout << "[INFO] capture restart = " << (capture_auto_restart ? "on" : "off") << std::endl;
    std::cout << "[INFO] cover detector  = score>="
              << env_int_value("A1_COVER_SCORE_THRESHOLD", 5, 3, 12)
              << " trigger=" << system_health.cover_trigger_frames
              << " recovery=" << system_health.cover_recovery_frames
              << " frames" << std::endl;
    if (test_fault_type != "none") {
        std::cout << "[TEST] fault injection type=" << test_fault_type
                  << " start_frame=" << test_fault_start
                  << " duration_frames=" << test_fault_duration << std::endl;
    }
    std::cout << "====================================================" << std::endl;

    while (!check_exit_flag()) {
        /*
         * 一次循环对应一帧：采集和数据健康检查 -> 模型推理 -> tracker 内部测距与
         * 规划 -> 异常决策覆盖 -> OSD/串口/语音输出 -> 性能统计。语音 Update 只写
         * 最新动作邮箱，实际 UART 发送在独立线程，因此不会阻塞 NPU 主循环。
         */
        const std::chrono::steady_clock::time_point loop_start = std::chrono::steady_clock::now();
        frame_id++;
        frame_stats = runtime_meter.Tick();

        const std::chrono::steady_clock::time_point capture_start = std::chrono::steady_clock::now();
        if (!processor.GetImage(&img_sensor)) {
            capture_failures++;
            if (output_serial_diagnostics &&
                (capture_failures == 1 || capture_failures % 30 == 0)) {
                std::cout << "[WARN] skip frame " << frame_id
                          << " because image capture failed, consecutive="
                          << capture_failures << std::endl;
            }
            if (capture_auto_restart &&
                (capture_failures == 10 || (capture_failures > 10 && capture_failures % 120 == 0))) {
                processor.Restart();
            }
            system_health.capture_failures = capture_failures;
            system_health.RefreshState();
            if (capture_failures >= 300) {
                std::cout << "[HEALTH][FATAL] capture recovery exhausted; request supervisor restart" << std::endl;
                exit_code = 10;
                g_exit_flag.store(true);
                break;
            }
            if (system_health.FaultActive()) {
                const AvoidanceDecision safe_decision = system_health.SafeDecision();
#if A1_ENABLE_VOICE
                if (voice_notifier.WantsOsd()) {
                    visualizer.Draw(empty_result, safe_decision, SurfaceResult());
                }
                voice_notifier.Update(frame_id, empty_result, safe_decision);
#else
                visualizer.Draw(empty_result, safe_decision, SurfaceResult());
#endif
                if (output_human_summary && frame_id % output_interval_frames == 0) {
                    print_fault_packet(frame_id, system_health, output_serial_diagnostics);
                }
                last_fault_active = true;
                last_fault_reason = system_health.reason;
                last_fault_type = fault_type_text(system_health);
            }
            usleep(capture_failures <= 10 ? 120000 : 30000);
            continue;
        }
        const float capture_ms = std::chrono::duration_cast<std::chrono::duration<float, std::milli> >(
            std::chrono::steady_clock::now() - capture_start).count();
        capture_failures = 0;
        system_health.capture_failures = 0;

        light_stats = analyze_light_stats(&img_sensor);
        system_health.UpdateData(light_stats);
        if (output_serial_diagnostics &&
            ((frame_id == 1 || frame_id % perf_interval_frames == 0) ||
             system_health.data_fault_frames == 1 ||
             system_health.data_fault_frames == system_health.cover_trigger_frames ||
             (system_health.data_fault_frames > system_health.cover_trigger_frames &&
              system_health.data_fault_frames % 30 == 0))) {
            std::cout << "[HEALTH][DATA] frame=" << frame_id
                      << " state=" << light_stats.state
                      << " mean=" << light_stats.mean
                      << " std=" << light_stats.stddev
                      << " dark=" << light_stats.dark_ratio
                      << " bright=" << light_stats.bright_ratio
                      << " range=" << light_stats.dynamic_range
                      << " edge=" << light_stats.edge_ratio
                      << " center_std=" << light_stats.center_stddev
                      << " center_range=" << light_stats.center_dynamic_range
                      << " center_edge=" << light_stats.center_edge_ratio
                      << " cover_score=" << light_stats.cover_score
                      << " cover_candidate=" << (light_stats.cover_candidate ? 1 : 0)
                      << " bad_frames=" << system_health.data_fault_frames
                      << " frozen_frames=" << system_health.frozen_frames
                      << std::endl;
        }
        const int64_t now_ms = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        bool ran_surface = false;
        bool inference_ok = true;
#if A1_ENABLE_SURFACE_SEG
        ran_surface = surface_available && !surface_degraded &&
                      frame_id % surface_period == surface_slot;
        if (ran_surface) {
            inference_ok = surface_segmenter.Predict(&img_sensor, &surface_result);
            if (inference_ok) {
                surface_failures = 0;
                surface_result.perception_degraded = false;
            } else {
                ++surface_failures;
                if (surface_failures >= 3) {
                    surface_degraded = true;
                    surface_result.perception_degraded = true;
                    if (!surface_degraded_announced) {
                        std::cout << "[SURFACE][DEGRADED] three consecutive inference failures; "
                                  << "detector-only fallback remains active" << std::endl;
                        surface_degraded_announced = true;
                    }
                }
            }
        } else {
            inference_ok = detector.Predict(&img_sensor, det_result, 0.20f);
            system_health.inference_failures = inference_ok ? 0 : system_health.inference_failures + 1;
        }
#else
        inference_ok = detector.Predict(&img_sensor, det_result, 0.20f);
        system_health.inference_failures = inference_ok ? 0 : system_health.inference_failures + 1;
#endif
        system_health.UpdateResource(frame_stats, *det_result);

        /*
         * 可复现异常注入钩子，默认关闭。它只修改健康证据，不破坏模型文件或耗尽
         * 真实资源，因此能安全验证 OSD、语音保护和自动恢复闭环。
         */
        const bool test_fault_active = test_fault_type != "none" &&
                                       frame_id >= test_fault_start &&
                                       frame_id < test_fault_start + test_fault_duration;
        if (test_fault_active) {
            if (test_fault_type == "camera") {
                system_health.data_fault_frames = std::max(
                    system_health.data_fault_frames, system_health.cover_trigger_frames);
            } else if (test_fault_type == "inference") {
                system_health.inference_failures = std::max(system_health.inference_failures, 2);
            } else if (test_fault_type == "resource") {
                system_health.resource_fault_frames = std::max(system_health.resource_fault_frames, 20);
            }
        }
        system_health.RefreshState();
        if (system_health.inference_failures >= 30) {
            std::cout << "[HEALTH][FATAL] inference recovery exhausted; request supervisor restart" << std::endl;
            exit_code = 20;
            g_exit_flag.store(true);
            break;
        }
        SurfaceResult surface_snapshot = surface_result;
        surface_snapshot.perception_degraded = surface_degraded;
        if (surface_snapshot.valid && now_ms - surface_snapshot.timestamp_ms > surface_stale_ms) {
            surface_snapshot.stale = true;
        }
        tracker.SetSurfaceResult(surface_snapshot);
        const std::chrono::steady_clock::time_point tracker_start = std::chrono::steady_clock::now();
        if (ran_surface) {
            tracker.PredictOnly(frame_id, now_ms);
        } else {
            tracker.Update(*det_result, frame_id);
        }
        const float tracker_ms = std::chrono::duration_cast<std::chrono::duration<float, std::milli> >(
            std::chrono::steady_clock::now() - tracker_start).count();

        const DetectionResult& stable_result = tracker.StableResult();
        const AvoidanceDecision& tracker_decision = tracker.Decision();
        surface_snapshot = tracker.LatestSurfaceResult();
        AvoidanceDecision health_decision = tracker_decision;
#if A1_ENABLE_SURFACE_SEG
        health_decision = surface_fusion.Fuse(tracker_decision, surface_snapshot, now_ms);
#else
        health_decision.perception_degraded = true;
        health_decision.perception_source = "detection_only";
#endif
        const bool fault_active = system_health.FaultActive();
        if (fault_active) {
            health_decision = system_health.SafeDecision();
            if (output_human_summary &&
                (!last_fault_active || last_fault_reason != system_health.reason ||
                 frame_id % output_interval_frames == 0)) {
                print_fault_packet(frame_id, system_health, output_serial_diagnostics);
            }
        } else if (last_fault_active && output_human_summary) {
            std::cout << "[FAULT] frame=" << frame_id
                      << " status=RECOVERED"
                      << " type=" << last_fault_type
                      << " protection=RELEASED"
                      << " navigation=RESUMED" << std::endl;
        }
        last_fault_active = fault_active;
        last_fault_reason = fault_active ? system_health.reason : "ok";
        if (fault_active) {
            last_fault_type = fault_type_text(system_health);
        }
#if A1_ENABLE_VOICE
        const bool refresh_osd = fault_active ||
                                 health_decision.action != last_osd_action ||
                                 frame_id % osd_interval_frames == 0;
        if (voice_notifier.WantsOsd() && refresh_osd) {
            visualizer.Draw(fault_active ? empty_result : stable_result,
                            health_decision,
                            surface_snapshot);
            last_osd_action = health_decision.action;
        }
        voice_notifier.Update(frame_id,
                              fault_active ? empty_result : stable_result,
                              health_decision);
#else
        if (health_decision.action != last_osd_action || frame_id % osd_interval_frames == 0) {
            visualizer.Draw(fault_active ? empty_result : stable_result,
                            health_decision,
                            surface_snapshot);
            last_osd_action = health_decision.action;
        }
#endif

        if (output_serial_diagnostics && frame_id % perf_interval_frames == 0) {
            const DetectorTiming detector_timing = detector.GetLastTiming();
#if A1_ENABLE_SURFACE_SEG
            const SegmenterTiming segmenter_timing = surface_segmenter.GetLastTiming();
#else
            const SegmenterTiming segmenter_timing;
#endif
            const float loop_ms = std::chrono::duration_cast<std::chrono::duration<float, std::milli> >(
                std::chrono::steady_clock::now() - loop_start).count();
            std::cout << "[PERF] frame=" << frame_id
                      << " npu_slot=" << (ran_surface ? "surface" : "detection")
                      << " view=" << det_result->view_id
                      << " capture_ms=" << capture_ms
                      << " preprocess_ms=" << detector_timing.preprocess_ms
                      << " inference_ms=" << detector_timing.inference_ms
                      << " output_ms=" << detector_timing.output_ms
                      << " decode_nms_ms=" << detector_timing.postprocess_ms
                      << " seg_preprocess_ms=" << segmenter_timing.preprocess_ms
                      << " seg_inference_ms=" << segmenter_timing.inference_ms
                      << " seg_output_ms=" << segmenter_timing.output_ms
                      << " seg_postprocess_ms=" << segmenter_timing.postprocess_ms
                      << " track_range_plan_ms=" << tracker_ms
                      << " loop_ms=" << loop_ms
                      << " fps_avg=" << frame_stats.fps_avg
                      << " fps_ratio=" << frame_stats.fps_avg / std::max(1.0f, static_cast<float>(sensor_fps))
                      << " frame_p95_ms=" << frame_stats.p95_ms
                      << " jitter_pct=" << frame_stats.jitter_pct
                      << std::endl;
        }

        if (output_json_lines && frame_id % output_interval_frames == 0) {
            print_json_packet(frame_id, stable_result, health_decision, surface_snapshot);
        }
        if (output_human_summary && !fault_active && frame_id % output_interval_frames == 0) {
            print_human_packet(frame_id,
                               stable_result,
                               health_decision,
                               surface_snapshot,
                               frame_stats,
                               light_stats,
                               static_cast<int>(std::min<size_t>(det_result->Size(), 6)),
                               output_serial_diagnostics);
        }
    }

    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    delete det_result;
#if A1_ENABLE_VOICE
    voice_notifier.Release();
#endif
#if A1_ENABLE_SURFACE_SEG
    surface_segmenter.Release();
#endif
    detector.Release();
    processor.Release();
    visualizer.Release();

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return exit_code;
}
