#include "../include/voice_notifier.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <termios.h>
#include <unistd.h>

namespace obstacle {
namespace {

std::string getenv_string(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::string(value);
}

int getenv_int(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

std::string upper_ascii(std::string s)
{
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] >= 'a' && s[i] <= 'z') {
            s[i] = static_cast<char>(s[i] - 'a' + 'A');
        } else if (s[i] == '/' || s[i] == ' ' || s[i] == '-') {
            s[i] = '_';
        }
    }
    return s;
}

std::string compact_dir(const std::string& sector)
{
    if (sector == "left") return "L";
    if (sector == "right") return "R";
    if (sector == "wide") return "WIDE";
    if (sector == "left_center") return "LC";
    if (sector == "center_right") return "CR";
    return "C";
}

std::string dist_bucket(float distance_m)
{
    if (distance_m < 0.0f) return "UNK";
    if (distance_m < 1.0f) return "NEAR";
    if (distance_m < 2.0f) return "WARN";
    return "FAR";
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

speed_t baud_to_termios(int baud)
{
    switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        default: return B115200;
    }
}

}  // namespace

VoiceNotifier::VoiceNotifier()
    : mode_(Mode::Disabled),
      fd_(-1),
      stable_count_(0),
      frame_interval_(5),
      stable_needed_(3),
      clear_stable_needed_(18),
      cooldown_ms_(4000),
      last_sent_frame_(-100000),
      last_action_(""),
      last_key_(""),
      last_sent_time_(std::chrono::steady_clock::now() - std::chrono::seconds(60)) {}

bool VoiceNotifier::InitializeFromEnv()
{
    const std::string mode = getenv_string("A1_OUTPUT_MODE", "osd");
    if (mode == "voice") {
        mode_ = Mode::VoiceOnly;
    } else if (mode == "both") {
        mode_ = Mode::Both;
    } else {
        mode_ = Mode::Disabled;
        return false;
    }

    frame_interval_ = std::max(1, getenv_int("A1_VOICE_INTERVAL_FRAMES", 5));
    stable_needed_ = std::max(1, getenv_int("A1_VOICE_STABLE_FRAMES", 3));
    clear_stable_needed_ = std::max(stable_needed_, getenv_int("A1_VOICE_CLEAR_STABLE_FRAMES", 18));
    cooldown_ms_ = std::max(500, getenv_int("A1_VOICE_COOLDOWN_MS", 4000));

    const std::string device = getenv_string("A1_VOICE_UART", "/dev/ttyS1");
    const int baud = getenv_int("A1_VOICE_BAUD", 115200);
    if (!OpenUart(device, baud)) {
        std::cout << "[VOICE][WARN] disabled because UART open failed: "
                  << device << std::endl;
        mode_ = Mode::Disabled;
        return false;
    }

    std::cout << "[VOICE][INFO] enabled mode=" << mode
              << " uart=" << device
              << " baud=" << baud
              << " interval=" << frame_interval_
              << " stable=" << stable_needed_
              << " clear_stable=" << clear_stable_needed_
              << " cooldown_ms=" << cooldown_ms_
              << std::endl;
    SendLine("HELLO,A1_OBSTACLE_V1");
    return true;
}

bool VoiceNotifier::OpenUart(const std::string& device, int baud)
{
    fd_ = open(device.c_str(), O_WRONLY | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::cout << "[VOICE][WARN] open(" << device << ") failed: "
                  << std::strerror(errno) << std::endl;
        return false;
    }
    if (!ConfigureUart(baud)) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    return true;
}

bool VoiceNotifier::ConfigureUart(int baud)
{
    if (fd_ < 0) return false;

    struct termios tio;
    std::memset(&tio, 0, sizeof(tio));
    if (tcgetattr(fd_, &tio) != 0) {
        std::cout << "[VOICE][WARN] tcgetattr failed: " << std::strerror(errno) << std::endl;
        return false;
    }

    cfmakeraw(&tio);
    const speed_t speed = baud_to_termios(baud);
    cfsetispeed(&tio, speed);
    cfsetospeed(&tio, speed);
    tio.c_cflag |= CLOCAL | CREAD;
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |= CS8;
    tio.c_cflag &= ~PARENB;
    tio.c_cflag &= ~CSTOPB;
    tio.c_cflag &= ~CRTSCTS;
    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 0;

    if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
        std::cout << "[VOICE][WARN] tcsetattr failed: " << std::strerror(errno) << std::endl;
        return false;
    }
    return true;
}

bool VoiceNotifier::SendLine(const std::string& line)
{
    if (fd_ < 0) {
        return false;
    }
    const std::string payload = line + "\r\n";
    const ssize_t written = write(fd_, payload.c_str(), payload.size());
    if (written < 0) {
        std::cout << "[VOICE][WARN] write failed: " << std::strerror(errno) << std::endl;
        return false;
    }
    return written == static_cast<ssize_t>(payload.size());
}

std::string VoiceNotifier::BuildCommand(int frame_id,
                                        const DetectionResult& result,
                                        const AvoidanceDecision& decision)
{
    const int nearest_idx = find_nearest_index(result);
    std::string sector = "C";
    std::string cls = "NONE";
    std::string risk = "UNK";
    std::string dist = "UNK";
    int track = -1;

    if (nearest_idx >= 0) {
        const DetectionItem& item = result.items[nearest_idx];
        sector = compact_dir(item.sector);
        cls = upper_ascii(item.label.empty() ? item.semantic_class : item.label);
        risk = upper_ascii(item.risk_level);
        dist = dist_bucket(item.distance_m);
        track = item.track_id;
    }

    std::ostringstream oss;
    oss << "NAV"
        << ",F=" << frame_id
        << ",A=" << upper_ascii(decision.action)
        << ",D=" << sector
        << ",C=" << cls
        << ",R=" << risk
        << ",Z=" << dist
        << ",T=" << track;
    return oss.str();
}

bool VoiceNotifier::ShouldSend(const std::string& action, const std::string& key)
{
    if (action == last_action_) {
        stable_count_++;
    } else {
        last_action_ = action;
        stable_count_ = 1;
    }

    const int gate = (action == "clear") ? clear_stable_needed_ : stable_needed_;
    if (action != "stop" && stable_count_ < gate) {
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const int since_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sent_time_).count());

    if (action == "stop") {
        if (key != last_key_ || since_ms >= 1000) {
            last_key_ = key;
            last_sent_time_ = now;
            return true;
        }
        return false;
    }

    if (key == last_key_ && since_ms < cooldown_ms_) {
        return false;
    }
    last_key_ = key;
    last_sent_time_ = now;
    return true;
}

void VoiceNotifier::Update(int frame_id,
                           const DetectionResult& result,
                           const AvoidanceDecision& decision)
{
    if (mode_ == Mode::Disabled || fd_ < 0) {
        return;
    }
    if (frame_id % frame_interval_ != 0) {
        return;
    }

    const std::string action = decision.action.empty() ? "clear" : decision.action;
    const std::string line = BuildCommand(frame_id, result, decision);
    if (ShouldSend(action, line)) {
        SendLine(line);
        last_sent_frame_ = frame_id;
    }
}

void VoiceNotifier::Release()
{
    if (fd_ >= 0) {
        SendLine("BYE,A1_OBSTACLE_V1");
        close(fd_);
        fd_ = -1;
    }
    mode_ = Mode::Disabled;
}

}  // namespace obstacle

