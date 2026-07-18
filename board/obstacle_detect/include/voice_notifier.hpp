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
 * @brief A1 到 SYN6288 的异步导盲语音控制器。
 *
 * 主推理线程调用 Update() 时只覆盖“最新动作邮箱”，不会等待 UART 或语音播放。
 * WorkerLoop 独立完成固定语音帧选择、优先级抢占、重复节流、逐字节发送和状态
 * 回收。队列始终只保存最新动作，因此盲人不会听到已经过期的避障指令。
 *
 * 当前成功部署路径使用 UART0：A1 D0/TX -> 电平转换 -> SYN6288 RX，SYN6288 TX
 * -> 电平转换 -> A1 D2/RX，并共地。固定帧分别对应直行、减速、停下、左转、
 * 右转和异常；system_fault 具有最高优先级且在故障锁存期间持续重复。
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

    /** 读取 A1_VOICE_* 参数，配置 GPIO 复用和 9600 8N1 UART，并启动工作线程。 */
    bool InitializeFromEnv();

    /** 将 planner 最新动作写入邮箱；该函数必须保持非阻塞。 */
    void Update(int frame_id,
                const DetectionResult& result,
                const AvoidanceDecision& decision);

    /** 停止工作线程并释放 UART/GPIO 句柄。 */
    void Release();

    bool Enabled() const { return mode_ != Mode::Disabled; }
    bool WantsOsd() const { return mode_ != Mode::VoiceOnly; }

private:
    /** 使用官方 A1 API 打开 UART_TX0/RX0，并把 PIN0/PIN2 切换到 UART 复用功能。 */
    bool OpenA1UartApi(int baud);

    // Opens a Linux tty device path used as a fallback/debug UART backend.
    bool OpenTtyDevice(const std::string& device, int baud);

    // Configures an already-open tty as 8N1 raw UART.
    bool ConfigureTtyDevice(int baud);

    /** 按配置的字节间隔发送完整 SYN6288 帧，避免模块 FIFO/时序拒收。 */
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

    /** 语音状态机主循环：收状态、处理超时、选择最新动作并启动一次原子事务。 */
    void WorkerLoop();

    // Continuously drains and parses all status bytes currently in the RX FIFO.
    void PumpRx();

    // Advances the SYN6288 transaction state from one protocol status byte.
    void HandleStatusByte(uint8_t code);

    /** 建立一次原子语音事务；提交后由 ACK/空闲码或兼容定时器判定播放完成。 */
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
