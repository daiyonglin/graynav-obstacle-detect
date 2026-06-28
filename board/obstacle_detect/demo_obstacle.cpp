#include <cmath>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "include/tracker.hpp"
#include "include/utils.hpp"
#include "include/voice_notifier.hpp"

constexpr bool kOutputJsonLines = false;
constexpr bool kOutputHumanSummary = true;
constexpr bool kOutputSerialDiagnostics = false;
constexpr int kOutputIntervalFrames = 5;

bool g_exit_flag = false;
std::mutex g_mtx;

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

struct FrameStats {
    float fps;
    float fps_avg;
    FrameStats() : fps(0.0f), fps_avg(0.0f) {}
};

struct LightStats {
    float mean;
    float stddev;
    float dark_ratio;
    float bright_ratio;
    std::string state;

    LightStats()
        : mean(0.0f), stddev(0.0f), dark_ratio(0.0f), bright_ratio(0.0f), state("unknown") {}
};

class RuntimeMeter {
public:
    RuntimeMeter() : initialized_(false), fps_avg_(0.0f) {}

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
            fps_avg_ = (fps_avg_ <= 0.0f) ? stats.fps : (0.85f * fps_avg_ + 0.15f * stats.fps);
            stats.fps_avg = fps_avg_;
        }
        return stats;
    }

private:
    bool initialized_;
    float fps_avg_;
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
    std::string input;
    std::cout << "[INFO] keyboard listener started, input q to exit." << std::endl;

    while (true) {
        std::cin >> input;
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "[INFO] exit command received." << std::endl;
            break;
        }
    }
}

bool check_exit_flag()
{
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
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
    (void)frame_stats;

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
    int img_width = 720;
    int img_height = 1280;

    std::array<int, 2> det_shape = {384, 384};
    std::string path_det = "/app_demo/app_assets/models/yolov8n_head6.m1model";

    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }

    std::array<int, 2> img_shape = {img_width, img_height};

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);

    YOLOV8GRAY detector;
    detector.Initialize(path_det, &img_shape, &det_shape);

    DetectionResult* det_result = new DetectionResult;
    RuntimeMeter runtime_meter;
    FrameStats frame_stats;
    LightStats light_stats;
    obstacle::ObstacleTracker tracker;
    tracker.Initialize(img_shape);

    VISUALIZER visualizer;
    visualizer.Initialize(img_shape);
    obstacle::VoiceNotifier voice_notifier;
    voice_notifier.InitializeFromEnv();

    std::cout << "sleep for 0.2 second!" << std::endl;
    usleep(200000);

    ssne_tensor_t img_sensor = {};

    std::thread listener_thread(keyboard_listener);

    int frame_id = 0;
    int capture_failures = 0;
    const bool output_json_lines = env_flag_enabled("A1_OUTPUT_JSON", kOutputJsonLines);
    const bool output_human_summary = env_flag_enabled("A1_OUTPUT_HUMAN", kOutputHumanSummary);
    const bool output_serial_diagnostics = env_flag_enabled("A1_OUTPUT_SERIAL_DIAG", kOutputSerialDiagnostics);
    const int output_interval_frames = env_int_value("A1_OUTPUT_INTERVAL_FRAMES",
                                                     kOutputIntervalFrames,
                                                     1,
                                                     300);

    std::cout << "====================================================" << std::endl;
    std::cout << "[INFO] obstacle_detect demo started." << std::endl;
    std::cout << "[INFO] image shape     = [" << img_shape[0] << ", " << img_shape[1] << "]" << std::endl;
    std::cout << "[INFO] det input shape = [" << det_shape[0] << ", " << det_shape[1] << "]" << std::endl;
    std::cout << "[INFO] model path      = " << path_det << std::endl;
    std::cout << "[INFO] output json     = " << (output_json_lines ? "on" : "off") << std::endl;
    std::cout << "[INFO] output human    = " << (output_human_summary ? "on" : "off") << std::endl;
    std::cout << "[INFO] output diag     = " << (output_serial_diagnostics ? "on" : "off") << std::endl;
    std::cout << "[INFO] output interval = " << output_interval_frames << " frames" << std::endl;
    std::cout << "====================================================" << std::endl;

    while (!check_exit_flag()) {
        frame_id++;
        frame_stats = runtime_meter.Tick();

        if (!processor.GetImage(&img_sensor)) {
            capture_failures++;
            if (capture_failures == 1 || capture_failures % 30 == 0) {
                std::cout << "[WARN] skip frame " << frame_id
                          << " because image capture failed, consecutive="
                          << capture_failures << std::endl;
            }
            if (capture_failures <= 3 ||
                capture_failures == 10 ||
                (capture_failures > 10 && capture_failures % 60 == 0)) {
                processor.Restart();
            }
            usleep(capture_failures <= 3 ? 120000 : 30000);
            continue;
        }
        capture_failures = 0;

        light_stats = analyze_light_stats(&img_sensor);
        detector.Predict(&img_sensor, det_result, 0.20f);
        tracker.Update(*det_result, frame_id);

        const DetectionResult& stable_result = tracker.StableResult();
        if (voice_notifier.WantsOsd()) {
            visualizer.Draw(stable_result, tracker.Decision());
        }
        voice_notifier.Update(frame_id, stable_result, tracker.Decision());

        if (output_json_lines && frame_id % output_interval_frames == 0) {
            print_json_packet(frame_id, stable_result, tracker.Decision());
        }
        if (output_human_summary && frame_id % output_interval_frames == 0) {
            print_human_packet(frame_id,
                               stable_result,
                               tracker.Decision(),
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
    voice_notifier.Release();
    detector.Release();
    processor.Release();
    visualizer.Release();

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0;
}
