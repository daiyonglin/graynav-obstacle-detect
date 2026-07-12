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

#ifndef A1_ENABLE_VOICE
#define A1_ENABLE_VOICE 0
#endif

#ifndef A1_MODEL_FILENAME
#define A1_MODEL_FILENAME "yolov8n80_graycopy_head6.m1model"
#endif

#if A1_ENABLE_VOICE
#include "include/voice_notifier.hpp"
#endif

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
    std::string state;
    uint32_t sample_hash;

    LightStats()
        : mean(0.0f), stddev(0.0f), dark_ratio(0.0f), bright_ratio(0.0f),
          state("unknown"), sample_hash(2166136261u) {}
};

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
          fault_latched(false),
          memory_available_kb(-1),
          last_image_hash(0),
          state("ok"),
          reason("ok") {}

    void UpdateData(const LightStats& light)
    {
        // A covered sensor can retain a small bright edge, so requiring both
        // 97% dark pixels and very low variance misses real covers. Use strong
        // saturation/mean evidence while keeping ordinary dim scenes degraded
        // rather than failed.
        const bool dark_cover = light.mean < 38.0f && light.dark_ratio > 0.82f &&
                                light.stddev < 10.0f;
        const bool bright_cover = light.mean > 230.0f && light.bright_ratio > 0.88f;
        const bool flat_frame = light.stddev > 0.0f && light.stddev < 2.5f;
        const bool bad = dark_cover || bright_cover || flat_frame;
        // Use a leaky accumulator so a single bright edge in an otherwise
        // covered image cannot immediately erase the sensor-fault evidence.
        data_fault_frames = bad ? data_fault_frames + 1 : std::max(0, data_fault_frames - 1);
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
               data_fault_frames >= 8 ||
               frozen_frames >= 15 ||
               resource_fault_frames >= 20 ||
               low_memory_frames >= 3 ||
               candidate_burst_frames >= 5;
    }

    AvoidanceDecision SafeDecision() const
    {
        AvoidanceDecision decision;
        decision.action = "system_fault";
        decision.prompt = "reason=system_health " + reason;
        return decision;
    }

    void RefreshState()
    {
        const bool raw_fault = capture_failures >= 3 || inference_failures >= 2 ||
                               data_fault_frames >= 8 || frozen_frames >= 15 ||
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
            // A covered lens can briefly expose a bright edge and produce a
            // few nominal frames. Require sustained healthy imagery before
            // allowing CLEAR/navigation speech again.
            if (healthy_recovery_frames >= 30) {
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
        } else if (data_fault_frames >= 8) {
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
    double sum = 0.0;
    double sum_sq = 0.0;

    for (int y = 0; y < h; y += step) {
        for (int x = 0; x < w; x += step) {
            const int v = data[y * w + x];
            stats.sample_hash ^= static_cast<uint32_t>(v + ((x & 0xFF) << 8) + ((y & 0xFF) << 16));
            stats.sample_hash *= 16777619u;
            sum += v;
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
            if (v < 35) dark++;
            if (v > 220) bright++;
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

    if (stats.stddev < 18.0f) {
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
                       const AvoidanceDecision& decision)
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
    oss << ",\"nearest_track\":" << decision.nearest_track_id << "}";
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
            << " coarse_drop=" << result.coarse_drop_count;
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
    const int nearest_idx = find_nearest_index(result);
    if (nearest_idx < 0) {
        std::ostringstream oss;
        oss << "[F" << frame_id << "] "
            << action_display_text(decision.action)
            << " no obstacle";
        append_diagnostics(oss, nullptr);
        std::cout << oss.str() << std::endl;
        return;
    }

    const DetectionItem& item = result.items[nearest_idx];

    std::ostringstream oss;
    oss << "[F" << frame_id << "] "
        << action_display_text(decision.action)
        << " dir=" << item.sector
        << " cls=" << item.label
        << " dist=";
    if (item.distance_m >= 0.0f) {
        oss << std::fixed << std::setprecision(2) << item.distance_m << "m";
    } else {
        oss << "unknown";
    }
    oss << " risk=" << to_upper_text(item.risk_level);
    if (result.items.size() > 1) {
        oss << " objects=" << result.items.size();
    }
    append_diagnostics(oss, &item);

    std::cout << oss.str() << std::endl;
}

int main()
{
    int full_width = env_int_value("A1_FULL_FRAME_WIDTH", 720, 1, 4096);
    int full_height = env_int_value("A1_FULL_FRAME_HEIGHT", 1280, 1, 4096);
    int capture_width = env_int_value("A1_CAPTURE_WIDTH", 720, 1, full_width);
    int capture_height = env_int_value("A1_CAPTURE_HEIGHT", full_height, 1, full_height);

    std::array<int, 2> det_shape = {384, 384};
    std::string path_det = env_string_value(
        "A1_MODEL_PATH",
        std::string("/app_demo/app_assets/models/") + A1_MODEL_FILENAME);

    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }

    std::array<int, 2> full_shape = {full_width, full_height};
    std::array<int, 2> capture_shape = {capture_width, capture_height};

    IMAGEPROCESSOR processor;
    processor.Initialize(&capture_shape);

    YOLOV8GRAY detector;
    detector.Initialize(path_det, &capture_shape, &det_shape);

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
    const int sensor_fps = env_int_value("A1_SENSOR_FPS", 90, 1, 240);
    std::string last_osd_action;

    std::cout << "====================================================" << std::endl;
    std::cout << "[INFO] obstacle_detect demo started." << std::endl;
    std::cout << "[INFO] full image shape    = [" << full_shape[0] << ", " << full_shape[1] << "]" << std::endl;
    std::cout << "[INFO] capture image shape = [" << capture_shape[0] << ", " << capture_shape[1] << "]" << std::endl;
    std::cout << "[INFO] det input shape = [" << det_shape[0] << ", " << det_shape[1] << "]" << std::endl;
    std::cout << "[INFO] model path      = " << path_det << std::endl;
    std::cout << "[INFO] output json     = " << (output_json_lines ? "on" : "off") << std::endl;
    std::cout << "[INFO] output human    = " << (output_human_summary ? "on" : "off") << std::endl;
    std::cout << "[INFO] output diag     = " << (output_serial_diagnostics ? "on" : "off") << std::endl;
    std::cout << "[INFO] output interval = " << output_interval_frames << " frames" << std::endl;
    std::cout << "[INFO] OSD interval    = " << osd_interval_frames << " frames" << std::endl;
    std::cout << "[INFO] capture restart = " << (capture_auto_restart ? "on" : "off") << std::endl;
    std::cout << "====================================================" << std::endl;

    while (!check_exit_flag()) {
        const std::chrono::steady_clock::time_point loop_start = std::chrono::steady_clock::now();
        frame_id++;
        frame_stats = runtime_meter.Tick();

        const std::chrono::steady_clock::time_point capture_start = std::chrono::steady_clock::now();
        if (!processor.GetImage(&img_sensor)) {
            capture_failures++;
            if (capture_failures == 1 || capture_failures % 30 == 0) {
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
                    visualizer.Draw(empty_result, safe_decision);
                }
                voice_notifier.Update(frame_id, empty_result, safe_decision);
#else
                visualizer.Draw(empty_result, safe_decision);
#endif
                if (output_human_summary && frame_id % output_interval_frames == 0) {
                    std::cout << "[HEALTH][WARN] frame=" << frame_id
                              << " state=" << system_health.state
                              << " reason=" << system_health.reason
                              << " capture_failures=" << system_health.capture_failures
                              << std::endl;
                }
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
        if ((output_serial_diagnostics &&
             (frame_id == 1 || frame_id % perf_interval_frames == 0)) ||
            system_health.data_fault_frames == 1 ||
            system_health.data_fault_frames == 8 ||
            (system_health.data_fault_frames > 8 &&
             system_health.data_fault_frames % 30 == 0)) {
            std::cout << "[HEALTH][DATA] frame=" << frame_id
                      << " state=" << light_stats.state
                      << " mean=" << light_stats.mean
                      << " std=" << light_stats.stddev
                      << " dark=" << light_stats.dark_ratio
                      << " bright=" << light_stats.bright_ratio
                      << " bad_frames=" << system_health.data_fault_frames
                      << " frozen_frames=" << system_health.frozen_frames
                      << std::endl;
        }
        const bool inference_ok = detector.Predict(&img_sensor, det_result, 0.20f);
        system_health.inference_failures = inference_ok ? 0 : system_health.inference_failures + 1;
        system_health.UpdateResource(frame_stats, *det_result);
        system_health.RefreshState();
        if (system_health.inference_failures >= 30) {
            std::cout << "[HEALTH][FATAL] inference recovery exhausted; request supervisor restart" << std::endl;
            exit_code = 20;
            g_exit_flag.store(true);
            break;
        }
        const std::chrono::steady_clock::time_point tracker_start = std::chrono::steady_clock::now();
        tracker.Update(*det_result, frame_id);
        const float tracker_ms = std::chrono::duration_cast<std::chrono::duration<float, std::milli> >(
            std::chrono::steady_clock::now() - tracker_start).count();

        const DetectionResult& stable_result = tracker.StableResult();
        const AvoidanceDecision& tracker_decision = tracker.Decision();
        AvoidanceDecision health_decision = tracker_decision;
        if (system_health.FaultActive()) {
            health_decision = system_health.SafeDecision();
            if (frame_id % output_interval_frames == 0) {
                std::cout << "[HEALTH][WARN] frame=" << frame_id
                          << " state=" << system_health.state
                          << " reason=" << system_health.reason
                          << " infer_failures=" << system_health.inference_failures
                          << " data_fault_frames=" << system_health.data_fault_frames
                          << " frozen_frames=" << system_health.frozen_frames
                          << " resource_fault_frames=" << system_health.resource_fault_frames
                          << " mem_available_kb=" << system_health.memory_available_kb
                          << " candidate_burst_frames=" << system_health.candidate_burst_frames
                          << std::endl;
            }
        }
#if A1_ENABLE_VOICE
        const bool refresh_osd = system_health.FaultActive() ||
                                 health_decision.action != last_osd_action ||
                                 frame_id % osd_interval_frames == 0;
        if (voice_notifier.WantsOsd() && refresh_osd) {
            visualizer.Draw(system_health.FaultActive() ? empty_result : stable_result,
                            health_decision);
            last_osd_action = health_decision.action;
        }
        voice_notifier.Update(frame_id,
                              system_health.FaultActive() ? empty_result : stable_result,
                              health_decision);
#else
        if (health_decision.action != last_osd_action || frame_id % osd_interval_frames == 0) {
            visualizer.Draw(system_health.FaultActive() ? empty_result : stable_result,
                            health_decision);
            last_osd_action = health_decision.action;
        }
#endif

        if (output_serial_diagnostics && frame_id % perf_interval_frames == 0) {
            const DetectorTiming detector_timing = detector.GetLastTiming();
            const float loop_ms = std::chrono::duration_cast<std::chrono::duration<float, std::milli> >(
                std::chrono::steady_clock::now() - loop_start).count();
            std::cout << "[PERF] frame=" << frame_id
                      << " view=" << det_result->view_id
                      << " capture_ms=" << capture_ms
                      << " preprocess_ms=" << detector_timing.preprocess_ms
                      << " inference_ms=" << detector_timing.inference_ms
                      << " output_ms=" << detector_timing.output_ms
                      << " decode_nms_ms=" << detector_timing.postprocess_ms
                      << " track_range_plan_ms=" << tracker_ms
                      << " loop_ms=" << loop_ms
                      << " fps_avg=" << frame_stats.fps_avg
                      << " fps_ratio=" << frame_stats.fps_avg / std::max(1.0f, static_cast<float>(sensor_fps))
                      << " frame_p95_ms=" << frame_stats.p95_ms
                      << " jitter_pct=" << frame_stats.jitter_pct
                      << std::endl;
        }

        if (output_json_lines && frame_id % output_interval_frames == 0) {
            print_json_packet(frame_id, stable_result, health_decision);
        }
        if (output_human_summary && frame_id % output_interval_frames == 0) {
            print_human_packet(frame_id,
                               stable_result,
                               health_decision,
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
    detector.Release();
    processor.Release();
    visualizer.Release();

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return exit_code;
}
