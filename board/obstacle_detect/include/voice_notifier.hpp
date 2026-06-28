#pragma once

#include "common.hpp"

#include <chrono>
#include <string>

namespace obstacle {

class VoiceNotifier {
public:
    enum class Mode {
        Disabled,
        VoiceOnly,
        Both
    };

    VoiceNotifier();

    bool InitializeFromEnv();
    void Update(int frame_id,
                const DetectionResult& result,
                const AvoidanceDecision& decision);
    void Release();

    bool Enabled() const { return mode_ != Mode::Disabled; }
    bool WantsOsd() const { return mode_ != Mode::VoiceOnly; }

private:
    bool OpenUart(const std::string& device, int baud);
    bool ConfigureUart(int baud);
    bool SendLine(const std::string& line);
    bool ShouldSend(const std::string& action, const std::string& key);
    std::string BuildCommand(int frame_id,
                             const DetectionResult& result,
                             const AvoidanceDecision& decision);

private:
    Mode mode_;
    int fd_;
    int stable_count_;
    int frame_interval_;
    int stable_needed_;
    int clear_stable_needed_;
    int cooldown_ms_;
    int last_sent_frame_;
    std::string last_action_;
    std::string last_key_;
    std::chrono::steady_clock::time_point last_sent_time_;
};

}  // namespace obstacle

