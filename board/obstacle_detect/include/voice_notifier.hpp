#pragma once

#include "common.hpp"
#include "smartsoc/gpio_api.h"
#include "smartsoc/uart_api.h"

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace obstacle {

/**
 * Sends navigation decisions to the external SYN6288 speech module.
 *
 * The class owns the A1-side UART connection, converts stable obstacle
 * decisions into one-word action prompts, wraps them in the SYN6288 binary
 * command frame, and rate-limits repeated announcements so speech does not
 * lag behind the current avoidance decision.
 */
class VoiceNotifier {
public:
    enum class Mode {
        Disabled,
        VoiceOnly,
        Both
    };

    enum class Backend {
        A1UartApi,
        TtyDevice
    };

    enum class ModuleState {
        Unknown,
        Idle,
        WaitAccept,
        Speaking,
        ErrorRecover,
        Offline
    };

    VoiceNotifier();

    // Reads runtime environment switches and opens the selected UART backend.
    bool InitializeFromEnv();

    // Converts the latest stable avoidance decision into a rate-limited speech frame.
    void Update(int frame_id,
                const DetectionResult& result,
                const AvoidanceDecision& decision);

    // Releases UART/GPIO resources owned by the notifier.
    void Release();

    bool Enabled() const { return mode_ != Mode::Disabled; }
    bool WantsOsd() const { return mode_ != Mode::VoiceOnly; }

private:
    // Opens the official A1 UART_TX0/RX0 API path and configures PIN0/PIN2 mux.
    bool OpenA1UartApi(int baud);

    // Opens a Linux tty device path used as a fallback/debug UART backend.
    bool OpenTtyDevice(const std::string& device, int baud);

    // Configures an already-open tty as 8N1 raw UART.
    bool ConfigureTtyDevice(int baud);

    // Sends one complete SYN6288 frame through the active backend.
    bool SendBytes(const std::vector<uint8_t>& bytes);

    // Reopens the UART backend before a speech transaction if configured.
    bool ReopenBackend();

    // Reads and discards pending SYN6288 status bytes before a new transaction.
    void DrainRx(int timeout_ms);

    // Reads one SYN6288 response byte from UART_TX0/RX0 or tty fallback.
    bool ReadResponseByte(uint8_t* value, int timeout_ms);

    // Sends a SYN6288 frame and records receive/ACK status for diagnostics.
    bool SendFrameWithStatus(const std::vector<uint8_t>& bytes, const char* tag);

    // Queries SYN6288 busy/idle state with FD 00 02 21 DE.
    bool QueryBusyState(uint8_t* value);

    // Polls the software UART status channel until the SYN6288 reports idle.
    bool WaitUntilIdle(int timeout_ms, bool allow_unknown);

    // Sends a short action prompt, optionally interrupting any previous speech.
    bool SendPrompt(const std::string& action, bool interrupt_current);

    // Applies action-change, stability, and cooldown gates before speech.
    bool ShouldSend(const std::string& action, const std::string& key, std::string* reason);

    // Commits cooldown state only after a frame has been accepted or blind-sent.
    void CommitSent(const std::string& action, const std::string& key);

    // Builds a frame-independent key so repeated identical states can be suppressed.
    std::string BuildVoiceKey(const DetectionResult& result,
                              const AvoidanceDecision& decision) const;

    // Maps an avoidance action to the GBK payload used by the SYN6288 examples.
    std::vector<uint8_t> BuildPromptPayload(const std::string& action) const;

    // Encodes a GBK prompt into the SYN6288 binary command frame.
    std::vector<uint8_t> BuildSyn6288Frame(const std::vector<uint8_t>& payload) const;

    // Returns a precomputed SYN6288 frame for the five short navigation prompts.
    std::vector<uint8_t> BuildFixedPromptFrame(const std::string& action) const;

    // Plays a deterministic sequence to verify that the voice module accepts
    // multiple UART commands after boot.
    void RunStartupSelfTest();

    // Performs potentially slow UART transactions outside the inference loop.
    void WorkerLoop();

    // Continuously drains and parses all status bytes currently in the RX FIFO.
    void PumpRx();

    // Advances the SYN6288 transaction state from one protocol status byte.
    void HandleStatusByte(uint8_t code);

    // Starts or preempts one atomic speech transaction from the latest action.
    bool StartProtocolSpeech(int frame_id, const std::string& action, bool preempt);

    // Handles ACK/playback timeouts, one retry, status query and resynchronization.
    void HandleProtocolTimeouts();

    // Reopens UART only after a complete protocol transaction has failed.
    void RecoverProtocol(const char* reason);

    // Returns action priority and post-completion repeat interval.
    int ActionPriority(const std::string& action) const;
    int RepeatIntervalMs(const std::string& action) const;

    // Closes whichever UART/GPIO backend was opened.
    void CloseBackend();

private:
    Mode mode_;
    Backend backend_;
    int fd_;
    uart_handle_t uart_;
    gpio_handle_t gpio_;
    int stable_count_;
    int frame_interval_;
    int stable_needed_;
    int clear_stable_needed_;
    int cooldown_ms_;
    int clear_repeat_ms_;
    int stop_repeat_ms_;
    int fault_repeat_ms_;
    int fault_hold_ms_;
    int switch_min_ms_;
    int tx_gap_ms_;
    bool pre_stop_;
    bool ack_enabled_;
    bool require_ack_;
    bool query_idle_;
    bool fixed_frame_;
    bool use_prompt_prefix_;
    bool reopen_each_tx_;
    bool passive_rx_;
    bool diagnostic_;
    int ack_timeout_ms_;
    int idle_timeout_ms_;
    int recover_wait_ms_;
    int retry_count_;
    int baud_;
    int byte_gap_us_;
    int post_tx_delay_ms_;
    int passive_rx_ms_;
    int play_timeout_ms_;
    int inter_frame_ms_;
    int rx_poll_ms_;
    std::atomic<int> consecutive_no_rx_;
    std::atomic<int> consecutive_tx_failures_;
    std::atomic<int> tx_failure_count_;
    std::atomic<int> recovery_count_;
    std::atomic<int> tx_count_;
    std::atomic<int> rx_accepted_count_;
    std::atomic<int> rx_idle_count_;
    std::atomic<int> rx_rejected_count_;
    std::atomic<int> rx_completed_count_;
    std::atomic<int> rx_unknown_count_;
    std::atomic<int> rx_byte_count_;
    std::atomic<int> ack_timeout_count_;
    std::atomic<int> play_timeout_count_;
    std::atomic<unsigned int> transaction_seq_;
    uint8_t last_rx_code_;
    std::atomic<ModuleState> module_state_;
    int last_sent_frame_;
    std::string last_action_;
    std::string last_key_;
    std::string last_tx_detail_;
    std::string tty_device_;
    std::chrono::steady_clock::time_point last_sent_time_;
    std::chrono::steady_clock::time_point last_fault_seen_time_;
    std::thread worker_;
    std::mutex worker_mutex_;
    std::condition_variable worker_cv_;
    bool worker_stop_;
    bool pending_ready_;
    bool tx_in_flight_;
    int pending_frame_id_;
    std::string pending_action_;
    std::string pending_key_;
    std::string pending_reason_;
    std::string in_flight_key_;
    std::deque<uint8_t> rx_queue_;
    std::vector<uint8_t> transaction_frame_;
    int transaction_frame_id_;
    int transaction_retry_;
    bool transaction_accepted_;
    bool status_query_pending_;
    std::chrono::steady_clock::time_point protocol_started_time_;
    std::chrono::steady_clock::time_point transaction_tx_time_;
    std::chrono::steady_clock::time_point transaction_accept_time_;
    std::chrono::steady_clock::time_point status_query_time_;
    std::chrono::steady_clock::time_point last_frame_tx_time_;
};

}  // namespace obstacle
