#include "../include/voice_notifier.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <termios.h>
#include <unistd.h>

namespace obstacle {
namespace {

constexpr uint8_t kSyn6288FrameHead = 0xFD;
constexpr uint8_t kSyn6288CommandSpeak = 0x01;
constexpr uint8_t kSyn6288ParamGbkNoMusic = 0x01;
constexpr size_t kA1UartFifoBytes = 32;

const std::vector<uint8_t> kSyn6288Stop = {0xFD, 0x00, 0x02, 0x02, 0xFD};
const std::vector<uint8_t> kSyn6288Query = {0xFD, 0x00, 0x02, 0x21, 0xDE};
const uint8_t kPromptPrefix[] = {0x5B, 0x76, 0x38, 0x5D, 0x5B, 0x74, 0x35, 0x5D};
const uint8_t kPromptClear[] = {0xD6, 0xB1, 0xD0, 0xD0};
const uint8_t kPromptSlow[] = {0xBC, 0xF5, 0xCB, 0xD9};
// “停下”的 GBK 两字节序列。使用完整短词而非单字“停”，事务完成和动作保持
// 由工作线程控制，不能通过缩短文本来掩盖重复发送或半帧问题。
const uint8_t kPromptStop[] = {0xCD, 0xA3, 0xCF, 0xC2};
const uint8_t kPromptLeft[] = {0xD7, 0xF3, 0xD7, 0xAA};
const uint8_t kPromptRight[] = {0xD3, 0xD2, 0xD7, 0xAA};
const uint8_t kPromptFault[] = {0xD2, 0xEC, 0xB3, 0xA3};

const std::vector<uint8_t> kFixedClearFrame = {0xFD, 0x00, 0x07, 0x01, 0x01, 0xD6, 0xB1, 0xD0, 0xD0, 0x9D};
const std::vector<uint8_t> kFixedSlowFrame = {0xFD, 0x00, 0x07, 0x01, 0x01, 0xBC, 0xF5, 0xCB, 0xD9, 0xA1};
const std::vector<uint8_t> kFixedStopFrame = {0xFD, 0x00, 0x07, 0x01, 0x01, 0xCD, 0xA3, 0xCF, 0xC2, 0x99};
const std::vector<uint8_t> kFixedLeftFrame = {0xFD, 0x00, 0x07, 0x01, 0x01, 0xD7, 0xF3, 0xD7, 0xAA, 0xA3};
const std::vector<uint8_t> kFixedRightFrame = {0xFD, 0x00, 0x07, 0x01, 0x01, 0xD3, 0xD2, 0xD7, 0xAA, 0x86};
const std::vector<uint8_t> kFixedFaultFrame = {0xFD, 0x00, 0x07, 0x01, 0x01, 0xD2, 0xEC, 0xB3, 0xA3, 0xD4};

// 读取字符串运行参数；默认值与 scripts/run.sh 保持一致。
std::string getenv_string(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::string(value);
}

// 读取整数运行参数；具体安全下限在 InitializeFromEnv 中统一收紧。
int getenv_int(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

// 兼容 1/y/Y/t/T 的布尔运行参数。
bool getenv_bool(const char* name, bool fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T';
}

// tty 调试后端的波特率映射；生产路径使用 A1 官方 UART API。
speed_t baud_to_termios(int baud)
{
    switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        default: return B9600;
    }
}

void append_bytes(std::vector<uint8_t>* dst, const uint8_t* data, size_t len)
{
    dst->insert(dst->end(), data, data + len);
}

const char* syn6288_status_name(uint8_t value)
{
    switch (value) {
        case 0x41: return "accepted";
        case 0x45: return "receive_failed";
        case 0x4A: return "init_ok";
        case 0x4E: return "busy";
        case 0x4F: return "idle";
        default: return "unknown";
    }
}

std::string hex_byte(uint8_t value)
{
    char buf[8];
    std::snprintf(buf, sizeof(buf), "0x%02X", value);
    return std::string(buf);
}

std::string hex_bytes(const std::vector<uint8_t>& bytes)
{
    std::string out;
    char buf[8];
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i > 0) out += " ";
        std::snprintf(buf, sizeof(buf), "%02X", bytes[i]);
        out += buf;
    }
    return out;
}

const char* module_state_name(VoiceNotifier::ModuleState state)
{
    switch (state) {
        case VoiceNotifier::ModuleState::Unknown: return "UNKNOWN";
        case VoiceNotifier::ModuleState::Idle: return "IDLE";
        case VoiceNotifier::ModuleState::WaitAccept: return "WAIT_ACCEPT";
        case VoiceNotifier::ModuleState::Speaking: return "PLAYING";
        case VoiceNotifier::ModuleState::ErrorRecover: return "RECOVER";
        case VoiceNotifier::ModuleState::Offline: return "OFFLINE";
    }
    return "UNKNOWN";
}

}  // namespace

VoiceNotifier::VoiceNotifier()
    : mode_(Mode::Disabled),
      backend_(Backend::A1UartApi),
      fd_(-1),
      uart_(nullptr),
      gpio_(nullptr),
      stable_count_(0),
      frame_interval_(2),
      stable_needed_(1),
      clear_stable_needed_(3),
      cooldown_ms_(0),
      clear_repeat_ms_(0),
      stop_repeat_ms_(500),
      fault_repeat_ms_(0),
      fault_hold_ms_(2500),
      switch_min_ms_(0),
      tx_gap_ms_(0),
      stop_followup_hold_ms_(0),
      turn_followup_hold_ms_(0),
      pre_stop_(false),
      ack_enabled_(true),
      require_ack_(true),
      query_idle_(true),
      fixed_frame_(true),
      use_prompt_prefix_(false),
      reopen_each_tx_(false),
      passive_rx_(true),
      diagnostic_(false),
      ack_timeout_ms_(200),
      idle_timeout_ms_(80),
      recover_wait_ms_(1000),
      retry_count_(0),
      baud_(9600),
      byte_gap_us_(0),
      post_tx_delay_ms_(30),
      passive_rx_ms_(80),
      play_timeout_ms_(3000),
      action_prompt_ms_(1000),
      inter_frame_ms_(12),
      rx_poll_ms_(3),
      consecutive_no_rx_(0),
      consecutive_tx_failures_(0),
      tx_failure_count_(0),
      recovery_count_(0),
      tx_count_(0),
      rx_accepted_count_(0),
      rx_idle_count_(0),
      rx_rejected_count_(0),
      rx_completed_count_(0),
      rx_unknown_count_(0),
      rx_byte_count_(0),
      ack_timeout_count_(0),
      play_timeout_count_(0),
      transaction_seq_(0),
      last_rx_code_(0),
      module_state_(ModuleState::Unknown),
      last_sent_frame_(-100000),
      last_action_(""),
      last_key_(""),
      last_requested_action_(""),
      last_tx_detail_("not_sent"),
      tty_device_("/dev/ttyS1"),
      last_sent_time_(std::chrono::steady_clock::now() - std::chrono::seconds(60)),
      last_fault_seen_time_(std::chrono::steady_clock::now() - std::chrono::seconds(60)),
      worker_stop_(false),
      pending_ready_(false),
      tx_in_flight_(false),
      pending_frame_id_(0),
      transaction_frame_id_(0),
      transaction_retry_(0),
      transaction_accepted_(false),
      status_query_pending_(false),
      protocol_started_time_(std::chrono::steady_clock::now()),
      transaction_tx_time_(std::chrono::steady_clock::now()),
      transaction_accept_time_(std::chrono::steady_clock::now()),
      status_query_time_(std::chrono::steady_clock::now()),
      last_frame_tx_time_(std::chrono::steady_clock::now() - std::chrono::seconds(1)) {}

bool VoiceNotifier::InitializeFromEnv()
{
    /*
     * 运行参数分为三组：动作稳定/重复周期、UART 发送节拍、SYN6288 状态兼容策略。
     * 默认固定帧和 latest-action mailbox 是已上板验证的生产路径。
     */
    if (!getenv_bool("A1_VOICE_ENABLE", true)) {
        mode_ = Mode::Disabled;
        std::cout << "[VOICE][INFO] disabled by A1_VOICE_ENABLE=0" << std::endl;
        return false;
    }
    const std::string mode = getenv_string("A1_OUTPUT_MODE", "both");
    if (mode == "voice") {
        mode_ = Mode::VoiceOnly;
    } else if (mode == "both") {
        mode_ = Mode::Both;
    } else {
        mode_ = Mode::Disabled;
        return false;
    }

    frame_interval_ = std::max(1, getenv_int("A1_VOICE_INTERVAL_FRAMES", 1));
    stable_needed_ = std::max(1, getenv_int("A1_VOICE_STABLE_FRAMES", 1));
    clear_stable_needed_ = std::max(stable_needed_, getenv_int("A1_VOICE_CLEAR_STABLE_FRAMES", 3));
    // Action-only prompts are intentionally continuous: after the fixed short
    // word's timed transaction completes, the latest non-STOP action may be
    // sent immediately.  STOP keeps a separately configured ~1 s pause.
    cooldown_ms_ = std::max(0, getenv_int("A1_VOICE_COOLDOWN_MS", 0));
    // Zero means no post-completion silence.  CLEAR therefore follows the same
    // continuous short-word cadence as SLOW/LEFT/RIGHT unless an integrator
    // explicitly requests a longer repeat interval.
    clear_repeat_ms_ = std::max(0, getenv_int("A1_VOICE_CLEAR_REPEAT_MS", 0));
    stop_repeat_ms_ = std::max(0, getenv_int("A1_VOICE_STOP_REPEAT_MS", 500));
    fault_repeat_ms_ = std::max(0, getenv_int("A1_VOICE_FAULT_REPEAT_MS", 0));
    fault_hold_ms_ = std::max(1000, getenv_int("A1_VOICE_FAULT_HOLD_MS", 2500));
    switch_min_ms_ = std::max(0, getenv_int("A1_VOICE_SWITCH_MIN_MS", 0));
    tx_gap_ms_ = std::max(0, getenv_int("A1_VOICE_TX_GAP_MS", 0));
    stop_followup_hold_ms_ = std::max(
        0, getenv_int("A1_VOICE_STOP_FOLLOWUP_HOLD_MS", 0));
    turn_followup_hold_ms_ = std::max(
        0, getenv_int("A1_VOICE_TURN_FOLLOWUP_HOLD_MS", 0));
    pre_stop_ = getenv_bool("A1_VOICE_PRE_STOP", false);
    // The deployed SYN6288 carrier does not provide dependable ACK/IDLE
    // feedback.  A direct binary launch must therefore use the same paced,
    // timer-completed protocol as scripts/run.sh; otherwise the first prompt
    // can remain in WaitAccept/Speaking forever and all later actions vanish.
    ack_enabled_ = getenv_bool("A1_VOICE_ACK", false);
    require_ack_ = getenv_bool("A1_VOICE_REQUIRE_ACK", false);
    query_idle_ = getenv_bool("A1_VOICE_QUERY_IDLE", false);
    fixed_frame_ = getenv_bool("A1_VOICE_FIXED_FRAME", true);
    use_prompt_prefix_ = getenv_bool("A1_VOICE_USE_PREFIX", false);
    reopen_each_tx_ = getenv_bool("A1_VOICE_REOPEN_EACH_TX", false);
    passive_rx_ = getenv_bool("A1_VOICE_PASSIVE_RX", true);
    diagnostic_ = getenv_bool("A1_VOICE_DIAG", false);
    ack_timeout_ms_ = std::max(20, getenv_int("A1_VOICE_ACK_TIMEOUT_MS", 200));
    idle_timeout_ms_ = std::max(20, getenv_int("A1_VOICE_IDLE_TIMEOUT_MS", 180));
    recover_wait_ms_ = std::max(80, getenv_int("A1_VOICE_RECOVER_WAIT_MS", 1000));
    retry_count_ = std::max(0, getenv_int("A1_VOICE_RETRY", 1));
    byte_gap_us_ = std::max(0, getenv_int("A1_VOICE_BYTE_GAP_US", 2000));
    post_tx_delay_ms_ = std::max(9, getenv_int("A1_VOICE_POST_TX_DELAY_MS", 30));
    passive_rx_ms_ = std::max(0, getenv_int("A1_VOICE_PASSIVE_RX_MS", 80));
    play_timeout_ms_ = std::max(1000, getenv_int("A1_VOICE_PLAY_TIMEOUT_MS", 3000));
    // Use a dedicated fixed-prompt timer when ACK/IDLE feedback is unavailable.
    // A 600 ms blind timer could resend while the attached SYN6288 was still
    // speaking.  The module then rejected every following frame, which sounded
    // like one startup prompt followed by permanent silence.  One second is
    // still a continuous cadence for these short words but leaves the module a
    // deterministic idle window even when status bytes are absent.
    action_prompt_ms_ = std::max(700, getenv_int("A1_VOICE_ACTION_PROMPT_MS", 1000));
    inter_frame_ms_ = std::max(9, getenv_int("A1_VOICE_INTER_FRAME_MS", 12));
    rx_poll_ms_ = std::max(1, getenv_int("A1_VOICE_RX_POLL_MS", 3));

    const int baud = getenv_int("A1_VOICE_BAUD", 9600);
    baud_ = baud;
    const std::string backend = getenv_string("A1_VOICE_BACKEND", "a1_uart");
    bool opened = false;
    if (backend == "tty") {
        backend_ = Backend::TtyDevice;
        const std::string device = getenv_string("A1_VOICE_UART", "/dev/ttyS1");
        tty_device_ = device;
        opened = OpenTtyDevice(device, baud);
        std::cout << "[VOICE][INFO] backend=tty device=" << device << std::endl;
    } else {
        backend_ = Backend::A1UartApi;
        opened = OpenA1UartApi(baud);
        std::cout << "[VOICE][INFO] backend=a1_uart pins=UART_TX0/UART_RX0" << std::endl;
    }

    if (!opened) {
        std::cout << "[VOICE][WARN] disabled because UART initialization failed." << std::endl;
        CloseBackend();
        mode_ = Mode::Disabled;
        return false;
    }

    if (diagnostic_) {
        std::cout << "[VOICE][INFO] enabled mode=" << mode
              << " baud=" << baud
              << " interval=" << frame_interval_
              << " stable=" << stable_needed_
              << " clear_stable=" << clear_stable_needed_
              << " cooldown_ms=" << cooldown_ms_
              << " clear_repeat_ms=" << clear_repeat_ms_
              << " stop_repeat_ms=" << stop_repeat_ms_
              << " fault_repeat_ms=" << fault_repeat_ms_
              << " fault_hold_ms=" << fault_hold_ms_
              << " switch_min_ms=" << switch_min_ms_
              << " tx_gap_ms=" << tx_gap_ms_
              << " stop_followup_hold_ms=" << stop_followup_hold_ms_
              << " turn_followup_hold_ms=" << turn_followup_hold_ms_
              << " pre_stop=" << (pre_stop_ ? 1 : 0)
              << " ack=" << (ack_enabled_ ? 1 : 0)
              << " require_ack=" << (require_ack_ ? 1 : 0)
              << " query_idle=" << (query_idle_ ? 1 : 0)
              << " fixed_frame=" << (fixed_frame_ ? 1 : 0)
              << " prefix=" << (use_prompt_prefix_ ? 1 : 0)
              << " reopen_each_tx=" << (reopen_each_tx_ ? 1 : 0)
              << " byte_gap_us=" << byte_gap_us_
              << " passive_rx=" << (passive_rx_ ? 1 : 0)
              << " passive_rx_ms=" << passive_rx_ms_
              << " recover_wait_ms=" << recover_wait_ms_
              << " retry=" << retry_count_
                  << std::endl;
    } else {
        std::cout << "[VOICE][INFO] ready mode=" << mode
                  << " baud=" << baud
                  << " protocol=fixed_frame"
                  << " protocol=syn6288_compat_state_machine_v2"
                  << " transport=persistent_paced_duplex"
                  << " tx_gap_ms=" << tx_gap_ms_
                  << " latest_action_mailbox=on"
                  << std::endl;
    }

    // 生产模式不主动丢弃 RX；工作线程启动后持续消费 0x41/0x45/0x4A/0x4E/0x4F，
    // 防止旧状态字节滞留并污染后续语音事务。
    protocol_started_time_ = std::chrono::steady_clock::now();
    // 不同 SYN6288 载板的状态查询回传并不完全一致，因此初始按可发送处理，
    // RX 状态只作为辅助确认，不能成为阻塞持续播报的唯一条件。
    module_state_ = ModuleState::Idle;
    last_sent_time_ = std::chrono::steady_clock::now();

    worker_stop_ = false;
    pending_ready_ = false;
    worker_ = std::thread(&VoiceNotifier::WorkerLoop, this);
    std::cout << "[VOICE][INFO] asynchronous UART worker started" << std::endl;
    return true;
}

bool VoiceNotifier::OpenA1UartApi(int baud)
{
    // 先配置 P4 对应引脚复用，再分别设置 TX0/RX0 波特率和无校验格式。
    gpio_ = gpio_init();
    if (gpio_ == nullptr) {
        std::cout << "[VOICE][WARN] gpio_init failed. Is gpio_kmod.ko loaded?" << std::endl;
        return false;
    }

    // A1 官方 UART API 使用 UART_TX0/RX0，将 PIN0/PIN2 配置到对应复用功能。
    if (gpio_set_alternate(gpio_, GPIO_PIN_0, GPIO_AF_INPUT_NONE, GPIO_AF_OUTPUT_UART_TX0) != GPIO_SUCCESS) {
        std::cout << "[VOICE][WARN] gpio_set_alternate TX0 failed." << std::endl;
        return false;
    }
    if (gpio_set_alternate(gpio_, GPIO_PIN_2, GPIO_AF_INPUT_UART_RX0, GPIO_AF_OUTPUT_NONE) != GPIO_SUCCESS) {
        std::cout << "[VOICE][WARN] gpio_set_alternate RX0 failed." << std::endl;
        return false;
    }

    uart_ = uart_init();
    if (uart_ == nullptr) {
        std::cout << "[VOICE][WARN] uart_init failed. Is uart_kmod.ko loaded?" << std::endl;
        return false;
    }
    if (uart_set_baudrate(uart_, UART_TX0, static_cast<uint32_t>(baud)) != UART_SUCCESS ||
        uart_set_baudrate(uart_, UART_RX0, static_cast<uint32_t>(baud)) != UART_SUCCESS) {
        std::cout << "[VOICE][WARN] uart_set_baudrate failed." << std::endl;
        return false;
    }
    if (uart_set_parity(uart_, UART_TX0, UART_PARITY_NONE) != UART_SUCCESS ||
        uart_set_parity(uart_, UART_RX0, UART_PARITY_NONE) != UART_SUCCESS) {
        std::cout << "[VOICE][WARN] uart_set_parity failed." << std::endl;
        return false;
    }
    return true;
}

bool VoiceNotifier::OpenTtyDevice(const std::string& device, int baud)
{
    fd_ = open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::cout << "[VOICE][WARN] open(" << device << ") failed: "
                  << std::strerror(errno) << std::endl;
        return false;
    }
    if (!ConfigureTtyDevice(baud)) {
        return false;
    }
    return true;
}

bool VoiceNotifier::ConfigureTtyDevice(int baud)
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

bool VoiceNotifier::SendBytes(const std::vector<uint8_t>& bytes)
{
    /*
     * A1 UART 与当前 SYN6288 载板实测要求逐字节节拍发送。整帧一次写入容易收到
     * 0x45 或被静默丢弃，因此这里保持持久句柄并在字节间插入微秒级间隔。
     */
    if (bytes.empty()) {
        return true;
    }

    if (backend_ == Backend::A1UartApi) {
        if (uart_ == nullptr) return false;
        // 当前 A1 UART 驱动与 SYN6288 组合已验证必须按字节节拍发送。整帧一次写入
        // 会使模块返回 0x45，因此保持 UART 句柄常开，并在字节间插入规定延时。
        const size_t chunk_size = byte_gap_us_ > 0 ? 1 : kA1UartFifoBytes;
        for (size_t offset = 0; offset < bytes.size(); offset += chunk_size) {
            const size_t chunk = std::min(chunk_size, bytes.size() - offset);
            if (uart_send_data(uart_, UART_TX0, &bytes[offset], static_cast<uint32_t>(chunk)) != UART_SUCCESS) {
                std::cout << "[VOICE][WARN] uart_send_data frame failed at " << offset << std::endl;
                return false;
            }
            if (byte_gap_us_ > 0 && offset + chunk < bytes.size()) {
                usleep(static_cast<useconds_t>(byte_gap_us_));
            }
        }
        if (post_tx_delay_ms_ > 0) {
            usleep(static_cast<useconds_t>(post_tx_delay_ms_) * 1000);
        }
        return true;
    }

    if (fd_ < 0) {
        return false;
    }
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t written = write(fd_, &bytes[offset], bytes.size() - offset);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cout << "[VOICE][WARN] write failed: " << std::strerror(errno) << std::endl;
            return false;
        }
        offset += static_cast<size_t>(written);
        if (byte_gap_us_ > 0 && offset < bytes.size()) {
            usleep(static_cast<useconds_t>(byte_gap_us_));
        }
    }
    if (post_tx_delay_ms_ > 0) {
        usleep(static_cast<useconds_t>(post_tx_delay_ms_) * 1000);
    }
    return true;
}

bool VoiceNotifier::ReopenBackend()
{
    const bool backend_open = backend_ == Backend::A1UartApi ? uart_ != nullptr : fd_ >= 0;
    if (!reopen_each_tx_ && backend_open) {
        return true;
    }
    for (int attempt = 0; attempt < 3; ++attempt) {
        CloseBackend();
        usleep(static_cast<useconds_t>(20000 + attempt * 40000));
        const bool opened = backend_ == Backend::A1UartApi
            ? OpenA1UartApi(baud_)
            : OpenTtyDevice(tty_device_, baud_);
        if (opened) {
            return true;
        }
    }
    return false;
}

void VoiceNotifier::DrainRx(int timeout_ms)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    uint8_t value = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        if (!ReadResponseByte(&value, 1)) {
            break;
        }
    }
}

bool VoiceNotifier::ReadResponseByte(uint8_t* value, int timeout_ms)
{
    if (value == nullptr) {
        return false;
    }
    if (!rx_queue_.empty()) {
        *value = rx_queue_.front();
        rx_queue_.pop_front();
        return true;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    do {
        if (backend_ == Backend::A1UartApi) {
            if (uart_ == nullptr) {
                return false;
            }
            uint8_t buf[8] = {0};
            uint32_t received = 0;
            const int ret = uart_receive_data(uart_, UART_RX0, buf, sizeof(buf), &received);
            if (ret == UART_SUCCESS && received > 0) {
                *value = buf[0];
                for (uint32_t i = 1; i < received; ++i) rx_queue_.push_back(buf[i]);
                return true;
            }
        } else {
            if (fd_ < 0) {
                return false;
            }
            uint8_t buf[8] = {0};
            const ssize_t n = read(fd_, buf, sizeof(buf));
            if (n > 0) {
                *value = buf[0];
                for (ssize_t i = 1; i < n; ++i) rx_queue_.push_back(buf[i]);
                return true;
            }
            if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                return false;
            }
        }
        usleep(5000);
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

bool VoiceNotifier::QueryBusyState(uint8_t* value)
{
    if (value == nullptr) return false;
    *value = 0;

    // 上一句结束后模块可能已异步返回 0x4F。这里只消费有效状态字节，
    // ACK 或线路噪声不得被误判为本次查询的忙/闲结果。
    const auto scan_state = [this, value](int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        uint8_t code = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            if (!ReadResponseByte(&code, 5)) continue;
            if (code == 0x4F) {
                ++rx_idle_count_;
                *value = code;
                return true;
            }
            if (code == 0x4E) {
                *value = code;
                return true;
            }
            if (code == 0x41) ++rx_accepted_count_;
            if (code == 0x45) ++rx_rejected_count_;
        }
        return false;
    };

    if (scan_state(20)) return true;
    if (!SendBytes(kSyn6288Query)) return false;
    return scan_state(idle_timeout_ms_);
}

bool VoiceNotifier::WaitUntilIdle(int timeout_ms, bool allow_unknown)
{
    if (!query_idle_) {
        return true;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    uint8_t state = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            const bool urgent_pending = pending_ready_ &&
                (pending_action_ == "stop" || pending_action_ == "system_fault");
            if (urgent_pending) {
                last_tx_detail_ = "superseded_by_urgent";
                return false;
            }
        }
        if (QueryBusyState(&state)) {
            last_rx_code_ = state;
            if (state == 0x4F) {
                module_state_ = ModuleState::Idle;
                last_tx_detail_ = "idle=" + hex_byte(state) + "(idle)";
                return true;
            }
            if (state == 0x4E) {
                module_state_ = ModuleState::Speaking;
                last_tx_detail_ = "idle=" + hex_byte(state) + "(busy)";
                usleep(120000);
                continue;
            }
            if (state == 0x4A) {
                module_state_ = ModuleState::Idle;
                last_tx_detail_ = "idle=" + hex_byte(state) + "(init_ok)";
                return true;
            }
            last_tx_detail_ = "idle=" + hex_byte(state) + "(" + syn6288_status_name(state) + ")";
            return allow_unknown;
        }
        last_tx_detail_ = "idle_timeout";
        if (allow_unknown || !require_ack_) {
            module_state_ = ModuleState::Unknown;
            if (!allow_unknown) last_tx_detail_ = "idle_timeout_fallback";
            return true;
        }
        usleep(80000);
    }
    module_state_ = ModuleState::Speaking;
    return false;
}

bool VoiceNotifier::SendFrameWithStatus(const std::vector<uint8_t>& bytes, const char* tag)
{
    last_tx_detail_ = "tx_pending";
    last_rx_code_ = 0;
    if (ack_enabled_) DrainRx(10);
    if (!SendBytes(bytes)) {
        last_tx_detail_ = std::string(tag) + ":uart_send_failed";
        return false;
    }

    bool accepted = true;
    if (ack_enabled_) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(ack_timeout_ms_);
        uint8_t code = 0;
        bool saw_ack = false;
        bool saw_reject = false;
        std::string rx_codes;
        while (std::chrono::steady_clock::now() < deadline) {
            if (!ReadResponseByte(&code, 5)) continue;
            if (!rx_codes.empty()) rx_codes += "/";
            rx_codes += hex_byte(code);
            if (code == 0x41) {
                ++rx_accepted_count_;
                last_rx_code_ = code;
                saw_ack = true;
                module_state_ = ModuleState::Speaking;
                break;
            }
            if (code == 0x45) {
                ++rx_rejected_count_;
                last_rx_code_ = code;
                saw_reject = true;
                module_state_ = ModuleState::ErrorRecover;
                break;
            }
            if (code == 0x4F) {
                ++rx_idle_count_;
                module_state_ = ModuleState::Idle;
            } else if (code == 0x4E) {
                module_state_ = ModuleState::Speaking;
            }
        }
        if (saw_ack) {
            last_tx_detail_ = std::string(tag) + ":ack=0x41(accepted),rx=" + rx_codes;
            accepted = true;
            consecutive_no_rx_ = 0;
        } else if (saw_reject) {
            last_tx_detail_ = std::string(tag) + ":ack=0x45(receive_failed),rx=" + rx_codes;
            accepted = false;
        } else {
            last_rx_code_ = 0;
            last_tx_detail_ = std::string(tag) + ":ack_timeout,rx=" +
                              (rx_codes.empty() ? "none" : rx_codes);
            accepted = !require_ack_;
            ++consecutive_no_rx_;
            if (accepted) {
                module_state_ = ModuleState::Unknown;
            }
        }
    } else {
        last_tx_detail_ = std::string(tag) + ":ack_disabled";
        module_state_ = ModuleState::Unknown;
    }

    return accepted;
}

bool VoiceNotifier::SendPrompt(const std::string& action, bool interrupt_current)
{
    std::string pre_detail;
    last_tx_detail_ = "not_sent";
    if (interrupt_current) {
        // 可选取消帧被拒绝时绝不能抑制真正的导航指令。除非已验证的模块版本
        // 显式启用 pre-stop，否则直接发送最新动作帧。
        if (pre_stop_) {
            const bool stop_ok = SendBytes(kSyn6288Stop);
            usleep(static_cast<useconds_t>(recover_wait_ms_) * 1000);
            DrainRx(80);
            pre_detail = std::string("pre_stop=") + (stop_ok ? "ok" : "fail") + ",";
            module_state_ = ModuleState::Idle;
        } else {
            pre_detail = "interrupt_direct,";
        }
    } else {
        const int wait_ms = std::max(120, getenv_int("A1_VOICE_WAIT_IDLE_MS", 650));
        if (!WaitUntilIdle(wait_ms, false)) {
            if (last_tx_detail_ != "superseded_by_urgent") {
                last_tx_detail_ = pre_detail + "module_busy_pending";
            }
            return false;
        }
        if (!last_tx_detail_.empty() && last_tx_detail_ != "not_sent") {
            pre_detail += last_tx_detail_ + ",";
        }
    }

    if (pre_stop_ && !interrupt_current) {
        const bool stop_ok = SendBytes(kSyn6288Stop);
        usleep(static_cast<useconds_t>(recover_wait_ms_) * 1000);
        DrainRx(80);
        pre_detail += std::string("cfg_pre_stop=") + (stop_ok ? "ok" : "fail") + ",";
        module_state_ = ModuleState::Idle;
    }

    const std::vector<uint8_t> frame =
        fixed_frame_ ? BuildFixedPromptFrame(action) : BuildSyn6288Frame(BuildPromptPayload(action));
    if (!ack_enabled_ && !query_idle_) {
        const bool reopen_ok = ReopenBackend();
        if (!reopen_ok) {
            last_tx_detail_ = pre_detail + "reopen=fail,frame_hex=" + hex_bytes(frame);
            return false;
        }
        if (!SendBytes(frame)) {
            last_tx_detail_ = pre_detail + "reopen=ok,blind_tx:uart_send_failed,frame_hex=" + hex_bytes(frame);
            return false;
        }
        std::string rx_detail;
        if (passive_rx_ && passive_rx_ms_ > 0) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(passive_rx_ms_);
            uint8_t value = 0;
            int rx_count = 0;
            while (std::chrono::steady_clock::now() < deadline) {
                if (ReadResponseByte(&value, 5)) {
                    if (!rx_detail.empty()) rx_detail += "/";
                    rx_detail += hex_byte(value) + "(" + syn6288_status_name(value) + ")";
                    rx_count++;
                    if (value == 0x41) ++rx_accepted_count_;
                    if (value == 0x4F) ++rx_idle_count_;
                    if (value == 0x45) ++rx_rejected_count_;
                    continue;
                }
                usleep(5000);
            }
            if (rx_count == 0) {
                rx_detail = "none";
            }
        } else {
            rx_detail = "off";
        }
        module_state_ = ModuleState::Unknown;
        last_tx_detail_ = pre_detail +
                          std::string("reopen=") + (reopen_each_tx_ ? "ok" : "skip") +
                          "," + (fixed_frame_ ? "fixed_frame:blind_tx" : "dynamic_frame:blind_tx") +
                          ",frame_hex=" + hex_bytes(frame) +
                          ",rx=" + rx_detail +
                          ",transport=persistent_atomic_blind";
        ++tx_count_;
        return true;
    }

    std::string previous_failure;
    for (int attempt = 0; attempt <= retry_count_; ++attempt) {
        if (SendFrameWithStatus(frame, "speak")) {
            ++tx_count_;
            last_tx_detail_ = pre_detail + last_tx_detail_;
            if (attempt > 0) {
                last_tx_detail_ += ",prev=" + previous_failure + ",retry_ok";
            }
            return true;
        }
        const std::string failed_detail = last_tx_detail_;
        if (attempt < retry_count_) {
            // 0x45 表示语音帧接收失败，应重发原语音帧；此处再发停止帧也可能被拒绝，
            // 并曾导致模块保持静默，因此不在失败恢复路径插入额外停止命令。
            usleep(last_rx_code_ == 0x45 ? 120000 : 180000);
            previous_failure = failed_detail + ",direct_resend";
        }
    }
    if (!previous_failure.empty()) {
        last_tx_detail_ = pre_detail + last_tx_detail_ + ",prev=" + previous_failure;
    } else {
        last_tx_detail_ = pre_detail + last_tx_detail_;
    }
    return false;
}

std::vector<uint8_t> VoiceNotifier::BuildPromptPayload(const std::string& action) const
{
    std::vector<uint8_t> payload;
    payload.reserve(16);
    if (use_prompt_prefix_) {
        append_bytes(&payload, kPromptPrefix, sizeof(kPromptPrefix));
    }
    if (action == "stop") {
        append_bytes(&payload, kPromptStop, sizeof(kPromptStop));
        return payload;
    }
    if (action == "turn_left") {
        append_bytes(&payload, kPromptLeft, sizeof(kPromptLeft));
        return payload;
    }
    if (action == "turn_right") {
        append_bytes(&payload, kPromptRight, sizeof(kPromptRight));
        return payload;
    }
    if (action == "system_fault") {
        append_bytes(&payload, kPromptFault, sizeof(kPromptFault));
        return payload;
    }
    if (action == "slow") {
        append_bytes(&payload, kPromptSlow, sizeof(kPromptSlow));
        return payload;
    }
    append_bytes(&payload, kPromptClear, sizeof(kPromptClear));
    return payload;
}

std::vector<uint8_t> VoiceNotifier::BuildSyn6288Frame(const std::vector<uint8_t>& payload) const
{
    // 协议结构：0xFD + 两字节长度 + 0x01(文本合成) + 参数字节 + GBK 文本 + XOR 校验。
    const size_t payload_len = std::min<size_t>(payload.size(), 200);

    const uint16_t data_len = static_cast<uint16_t>(payload_len + 3);
    std::vector<uint8_t> frame;
    frame.reserve(payload_len + 6);
    frame.push_back(kSyn6288FrameHead);
    frame.push_back(static_cast<uint8_t>((data_len >> 8) & 0xFF));
    frame.push_back(static_cast<uint8_t>(data_len & 0xFF));
    frame.push_back(kSyn6288CommandSpeak);
    frame.push_back(kSyn6288ParamGbkNoMusic);
    frame.insert(frame.end(), payload.begin(), payload.begin() + payload_len);

    uint8_t ecc = 0;
    for (size_t i = 0; i < frame.size(); ++i) {
        ecc ^= frame[i];
    }
    frame.push_back(ecc);
    return frame;
}

std::vector<uint8_t> VoiceNotifier::BuildFixedPromptFrame(const std::string& action) const
{
    // 生产模式直接返回离线核验过的短词帧，避免板端字符编码转换差异。
    if (action == "stop") {
        return kFixedStopFrame;
    }
    if (action == "turn_left") {
        return kFixedLeftFrame;
    }
    if (action == "turn_right") {
        return kFixedRightFrame;
    }
    if (action == "system_fault") {
        return kFixedFaultFrame;
    }
    if (action == "slow") {
        return kFixedSlowFrame;
    }
    return kFixedClearFrame;
}

void VoiceNotifier::RunStartupSelfTest()
{
    const char* actions[] = {"clear", "slow", "stop", "turn_left", "turn_right"};
    for (size_t i = 0; i < sizeof(actions) / sizeof(actions[0]); ++i) {
        const bool ok = SendPrompt(actions[i], std::string(actions[i]) == "stop");
        std::cout << "[VOICE][SELFTEST] action=" << actions[i]
                  << " status=" << (ok ? "ok" : "fail")
                  << " detail=" << last_tx_detail_ << std::endl;
        WaitUntilIdle(2500, true);
        usleep(300000);
    }
}

bool VoiceNotifier::ShouldSend(const std::string& action, const std::string& key, std::string* reason)
{
    /*
     * 发送门控同时考虑动作稳定帧数、同动作冷却和最小 UART 间隔。STOP/异常可
     * 更快响应，但仍不能从另一个数据帧中间插入，防止模块接收半帧后锁死。
     */
    if (action == last_action_) {
        stable_count_++;
    } else {
        last_action_ = action;
        stable_count_ = 1;
    }

    const auto now = std::chrono::steady_clock::now();
    const int since_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sent_time_).count());
    const bool changed_action = key != last_key_;
    const bool risk_upgrade = changed_action && action != "clear";
    // 新 UART 帧必须避开前一句短语的事务窗口。高优先级安全动作可缩短等待，
    // 但不能从帧中间插入，否则驱动层虽写入成功，SYN6288 仍可能丢弃语音。
    const bool safety_action = action == "stop" || action == "system_fault";
    int minimum_gap_ms = safety_action ? std::min(tx_gap_ms_, 600) : tx_gap_ms_;
    if ((last_key_ == "stop" || last_key_ == "system_fault") && action != last_key_) {
        minimum_gap_ms = std::max(minimum_gap_ms, 1200);
    }
    if (since_ms < minimum_gap_ms) {
        if (reason != nullptr) *reason = "tx_gap";
        return false;
    }

    if (action == "system_fault") {
        if (key != last_key_ || since_ms >= fault_repeat_ms_) {
            if (reason != nullptr) {
                *reason = key != last_key_ ? "fault_changed" : "fault_repeat";
            }
            return true;
        }
        if (reason != nullptr) *reason = "fault_cooldown";
        return false;
    }

    if (action == "stop") {
        const bool changed = key != last_key_;
        if (key != last_key_ || since_ms >= stop_repeat_ms_) {
            if (reason != nullptr) *reason = changed ? "stop_changed" : "stop_repeat";
            return true;
        }
        if (reason != nullptr) *reason = "stop_cooldown";
        return false;
    }

    const int gate = (action == "clear") ? clear_stable_needed_ : stable_needed_;
    if (stable_count_ < gate) {
        if (reason != nullptr) *reason = "wait_stable";
        return false;
    }

    const bool changed = key != last_key_;
    if (changed) {
        if (since_ms < switch_min_ms_ && action == "clear") {
            if (reason != nullptr) *reason = "switch_gap";
            return false;
        }
        if (reason != nullptr) *reason = risk_upgrade ? "risk_changed" : "changed_stable";
        return true;
    }

    const int repeat_ms = (action == "clear") ? clear_repeat_ms_ : cooldown_ms_;
    if (key == last_key_ && since_ms < repeat_ms) {
        if (reason != nullptr) *reason = "cooldown";
        return false;
    }
    if (reason != nullptr) *reason = "cooldown_elapsed";
    return true;
}

void VoiceNotifier::CommitSent(const std::string& action, const std::string& key)
{
    last_action_ = action;
    last_key_ = key;
    last_sent_time_ = std::chrono::steady_clock::now();
}

int VoiceNotifier::ActionPriority(const std::string& action) const
{
    if (action == "system_fault") return 100;
    if (action == "stop") return 90;
    if (action == "turn_left" || action == "turn_right") return 70;
    if (action == "slow") return 50;
    return 10;
}

int VoiceNotifier::RepeatIntervalMs(const std::string& action) const
{
    if (action == "system_fault") return fault_repeat_ms_;
    if (action == "stop") return stop_repeat_ms_;
    if (action == "clear") return clear_repeat_ms_;
    return cooldown_ms_;
}

void VoiceNotifier::PumpRx()
{
    if (backend_ != Backend::A1UartApi || uart_ == nullptr) return;

    for (int round = 0; round < 4; ++round) {
        uint8_t data[32] = {0};
        uint32_t received = 0;
        const int ret = uart_receive_data(uart_, UART_RX0, data, sizeof(data), &received);
        if (ret != UART_SUCCESS || received == 0) break;
        rx_byte_count_ += static_cast<int>(received);
        for (uint32_t i = 0; i < received; ++i) HandleStatusByte(data[i]);
        if (received < sizeof(data)) break;
    }
}

void VoiceNotifier::HandleStatusByte(uint8_t code)
{
    /* SYN6288 回传：0x41 接收成功、0x45 接收失败、0x4A 初始化、0x4E 忙、0x4F 空闲。 */
    const auto now = std::chrono::steady_clock::now();
    last_rx_code_ = code;
    if (code == 0x41) {
        if (status_query_pending_) return;
        if (module_state_.load() == ModuleState::WaitAccept && tx_in_flight_) {
            transaction_accepted_ = true;
            transaction_accept_time_ = now;
            module_state_ = ModuleState::Speaking;
            ++rx_accepted_count_;
            consecutive_no_rx_ = 0;
            const int latency = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - transaction_tx_time_).count());
            if (diagnostic_) {
                std::cout << "[VOICE] seq=" << transaction_seq_.load()
                          << " ACK code=0x41 latency_ms=" << latency << std::endl;
            }
        }
        return;
    }
    if (code == 0x45) {
        ++rx_rejected_count_;
        if (require_ack_ && module_state_.load() == ModuleState::WaitAccept && tx_in_flight_) {
            module_state_ = ModuleState::ErrorRecover;
            transaction_tx_time_ = now;
            std::cout << "[VOICE][WARN] seq=" << transaction_seq_.load()
                      << " REJECT code=0x45 retry=" << transaction_retry_ << std::endl;
        }
        return;
    }
    if (code == 0x4A) {
        status_query_pending_ = false;
        if (!tx_in_flight_) module_state_ = ModuleState::Idle;
        return;
    }
    if (code == 0x4E) {
        status_query_pending_ = false;
        module_state_ = ModuleState::Speaking;
        if (tx_in_flight_) {
            transaction_accept_time_ = now;
        }
        return;
    }
    if (code == 0x4F) {
        ++rx_idle_count_;
        status_query_pending_ = false;
        // 节拍兼容模式下，RX 完成字节可能属于上一句。用确定性短语计时器界定事务，
        // 防止迟到的 0x4F 提前结束刚选中的安全提示。
        if (!require_ack_) return;
        if (tx_in_flight_) {
            if (module_state_.load() != ModuleState::Speaking) {
                // 被抢占短语可能先返回 IDLE，而替代帧尚未返回 0x41；
                // 这个旧 IDLE 不能被当作新事务完成。
                if (diagnostic_) {
                    std::cout << "[VOICE][WARN] stale_idle_while_waiting_ack seq="
                              << transaction_seq_.load() << std::endl;
                }
                return;
            }
            const int duration = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - transaction_accept_time_).count());
            ++rx_completed_count_;
            {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                CommitSent(in_flight_action_, in_flight_key_);
                last_sent_frame_ = transaction_frame_id_;
                tx_in_flight_ = false;
                in_flight_action_.clear();
                in_flight_key_.clear();
            }
            if (diagnostic_) {
                std::cout << "[VOICE] seq=" << transaction_seq_.load()
                          << " DONE code=0x4F duration_ms=" << duration << std::endl;
            }
        }
        module_state_ = ModuleState::Idle;
        return;
    }
    ++rx_unknown_count_;
    if (diagnostic_) {
        std::cout << "[VOICE][WARN] unknown_rx=" << hex_byte(code) << std::endl;
    }
}

bool VoiceNotifier::StartProtocolSpeech(int frame_id,
                                        const std::string& action,
                                        const std::string& transaction_key,
                                        bool preempt)
{
    // 发送成功后只标记 in-flight；播放完成时才 CommitSent，保证冷却基于真实事务边界。
    const auto now = std::chrono::steady_clock::now();
    const int gap = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame_tx_time_).count());
    if (gap < inter_frame_ms_) {
        usleep(static_cast<useconds_t>(inter_frame_ms_ - gap) * 1000);
    }

    const std::vector<uint8_t> frame =
        fixed_frame_ ? BuildFixedPromptFrame(action) : BuildSyn6288Frame(BuildPromptPayload(action));
    if (frame.empty() || frame.size() > kA1UartFifoBytes || !SendBytes(frame)) {
        ++tx_failure_count_;
        ++consecutive_tx_failures_;
        return false;
    }

    last_frame_tx_time_ = std::chrono::steady_clock::now();
    transaction_frame_ = frame;
    transaction_frame_id_ = frame_id;
    transaction_retry_ = 0;
    transaction_accepted_ = false;
    status_query_pending_ = false;
    transaction_tx_time_ = last_frame_tx_time_;
    transaction_accept_time_ = last_frame_tx_time_;
    // 兼容模式中 ACK 只作辅助；模块可能不返回 0x41/0x4F，因此播放事务由
    // RX 空闲或短语计时器任一条件完成，保证无回传时仍能持续前进。
    module_state_ = require_ack_ ? ModuleState::WaitAccept : ModuleState::Speaking;
    ++transaction_seq_;
    ++tx_count_;
    consecutive_tx_failures_ = 0;
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        tx_in_flight_ = true;
        in_flight_action_ = action;
        in_flight_key_ = transaction_key;
    }
    if (diagnostic_) {
        std::cout << "[VOICE] seq=" << transaction_seq_.load()
                  << (preempt ? " PREEMPT" : " TX")
                  << " frame=" << frame_id
                  << " action=" << action
                  << " bytes=" << frame.size() << std::endl;
    }
    return true;
}

void VoiceNotifier::RecoverProtocol(const char* reason)
{
    ++recovery_count_;
    std::cout << "[VOICE][ERROR] transport_resync reason=" << reason
              << " recoveries=" << recovery_count_.load() << std::endl;
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        tx_in_flight_ = false;
        in_flight_action_.clear();
        in_flight_key_.clear();
    }
    status_query_pending_ = false;
    transaction_accepted_ = false;
    CloseBackend();
    usleep(100000);
    if (ReopenBackend()) {
        module_state_ = ModuleState::Unknown;
        protocol_started_time_ = std::chrono::steady_clock::now();
        consecutive_tx_failures_ = 0;
    } else {
        module_state_ = ModuleState::Offline;
        protocol_started_time_ = std::chrono::steady_clock::now();
        ++tx_failure_count_;
    }
}

void VoiceNotifier::HandleProtocolTimeouts()
{
    /*
     * 兼容部分载板不稳定回传 ACK 的情况：required_ack 关闭时用短词最大播放时长
     * 完成事务；真正发送失败才重开 UART，避免频繁复位破坏连续播报。
     */
    const auto now = std::chrono::steady_clock::now();
    const ModuleState state = module_state_.load();
    if (state == ModuleState::Unknown) {
        if (!query_idle_) {
            module_state_ = ModuleState::Idle;
            return;
        }
        const int elapsed = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now - protocol_started_time_).count());
        if (!status_query_pending_ && elapsed >= 500) {
            if (SendBytes(kSyn6288Query)) {
                status_query_pending_ = true;
                status_query_time_ = std::chrono::steady_clock::now();
                last_frame_tx_time_ = status_query_time_;
            }
        } else if (status_query_pending_) {
            const int query_age = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - status_query_time_).count());
            if (query_age >= ack_timeout_ms_) RecoverProtocol("startup_query_timeout");
        }
        return;
    }
    if (state == ModuleState::Offline) {
        const int elapsed = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now - protocol_started_time_).count());
        if (elapsed >= 3000) RecoverProtocol("offline_retry");
        return;
    }
    if (state == ModuleState::WaitAccept || state == ModuleState::ErrorRecover) {
        const int age = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now - transaction_tx_time_).count());
        const bool timeout = state == ModuleState::WaitAccept && age >= ack_timeout_ms_;
        const bool rejected_ready = state == ModuleState::ErrorRecover && age >= 20;
        if (!timeout && !rejected_ready) return;
        if (timeout) ++ack_timeout_count_;
        if (transaction_retry_ < 1) {
            usleep(static_cast<useconds_t>(inter_frame_ms_) * 1000);
            if (SendBytes(transaction_frame_)) {
                ++transaction_retry_;
                ++tx_count_;
                transaction_tx_time_ = std::chrono::steady_clock::now();
                last_frame_tx_time_ = transaction_tx_time_;
                module_state_ = ModuleState::WaitAccept;
                std::cout << "[VOICE][WARN] seq=" << transaction_seq_.load()
                          << " RETRY reason=" << (timeout ? "ack_timeout" : "reject") << std::endl;
                return;
            }
        }
        RecoverProtocol(timeout ? "ack_timeout" : "receive_failed");
        return;
    }
    if (state == ModuleState::Speaking && tx_in_flight_) {
        const int age = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now - transaction_tx_time_).count());
        // 导航短词会在该上限内完成。即使 RX 状态缺失或载板偶发 0x45，
        // 超时兜底也会结束当前事务，避免发送状态永久卡死。
        const int timed_completion_ms = fixed_frame_ ? action_prompt_ms_ :
            (transaction_frame_.size() <= 8 ? 650 : 900);
        if (!require_ack_ && age >= timed_completion_ms) {
            ++rx_completed_count_;
            {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                CommitSent(in_flight_action_, in_flight_key_);
                last_sent_frame_ = transaction_frame_id_;
                tx_in_flight_ = false;
                in_flight_action_.clear();
                in_flight_key_.clear();
            }
            module_state_ = ModuleState::Idle;
            if (diagnostic_) {
                std::cout << "[VOICE] seq=" << transaction_seq_.load()
                          << " DONE source=timer duration_ms=" << age << std::endl;
            }
            return;
        }
        if (!status_query_pending_ && age >= play_timeout_ms_) {
            ++play_timeout_count_;
            if (SendBytes(kSyn6288Query)) {
                status_query_pending_ = true;
                status_query_time_ = std::chrono::steady_clock::now();
                last_frame_tx_time_ = status_query_time_;
            } else {
                RecoverProtocol("status_query_send_failed");
            }
        } else if (status_query_pending_) {
            const int query_age = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - status_query_time_).count());
            if (query_age >= ack_timeout_ms_) RecoverProtocol("play_query_timeout");
        }
    }
}

void VoiceNotifier::WorkerLoop()
{
    /*
     * 工作线程每次读取邮箱最新值。普通动作等待当前短词完成；高优先级动作在
     * 协议允许时覆盖尚未发送的旧动作，不形成会滞后的 FIFO 语音队列。
     */
    while (true) {
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            worker_cv_.wait_for(lock, std::chrono::milliseconds(rx_poll_ms_),
                                [this]() { return worker_stop_; });
            if (worker_stop_) break;
        }

        PumpRx();
        HandleProtocolTimeouts();

        int frame_id = 0;
        std::string action;
        std::string key;
        bool have_action = false;
        bool active = false;
        std::string active_action;
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            have_action = pending_ready_;
            frame_id = pending_frame_id_;
            action = pending_action_;
            key = pending_key_;
            active = tx_in_flight_;
            active_action = in_flight_action_;
        }
        if (!have_action) continue;

        const ModuleState state = module_state_.load();
        if (active) {
            const bool direction_switch =
                (active_action == "turn_left" || active_action == "turn_right") &&
                (action == "turn_left" || action == "turn_right") && action != active_action;
            if (require_ack_ && action != active_action &&
                (ActionPriority(action) > ActionPriority(active_action) || direction_switch)) {
                StartProtocolSpeech(frame_id, action, key, true);
            }
            continue;
        }
        if (state != ModuleState::Idle) continue;

        const auto now = std::chrono::steady_clock::now();
        const int since_complete = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sent_time_).count());
        const bool changed = key != last_key_;
        if (changed) {
            int followup_hold_ms = 0;
            if (last_action_ == "stop") {
                followup_hold_ms = stop_followup_hold_ms_;
            } else if (last_action_ == "turn_left" || last_action_ == "turn_right") {
                followup_hold_ms = turn_followup_hold_ms_;
            }

            // 播报完成后为使用者保留执行动作的时间。风险升级不等待：异常总是
            // 立即覆盖，STOP 也可立即覆盖转向；降低风险或反向转向才受保持门控。
            const bool safety_upgrade =
                action == "system_fault" ||
                (ActionPriority(action) >= 90 &&
                 ActionPriority(action) > ActionPriority(last_action_));
            if (!safety_upgrade && since_complete < followup_hold_ms) {
                continue;
            }
        }
        if (changed || since_complete >= RepeatIntervalMs(action)) {
            StartProtocolSpeech(frame_id, action, key, false);
        }
    }
}

void VoiceNotifier::Update(int frame_id,
                           const DetectionResult& result,
                           const AvoidanceDecision& decision)
{
    // 故障解除后仍保持 fault_hold_ms，防止图像短暂恢复一帧便穿插播报“直行”。
    if (mode_ == Mode::Disabled) {
        return;
    }
    (void)result;
    // Do not prepend object or scene descriptions.  The final stabilized
    // action is the complete voice contract: 直行/减速/停下/左转/右转/异常。
    std::string action = decision.action.empty() ? "clear" : decision.action;
    const auto now = std::chrono::steady_clock::now();
    if (action == "system_fault") {
        last_fault_seen_time_ = now;
    } else {
        const int since_fault_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fault_seen_time_).count());
        if (since_fault_ms < fault_hold_ms_) action = "system_fault";
    }
    const bool critical = action == "stop" || action == "system_fault";
    const bool action_changed = action != last_requested_action_;
    if (!critical && !action_changed && frame_id % frame_interval_ != 0) {
        return;
    }
    // Commit the requested action only after it is eligible for enqueueing;
    // otherwise a change observed between interval frames would be marked as
    // handled and the short LEFT/RIGHT/SLOW prompt could be lost entirely.
    last_requested_action_ = action;
    std::lock_guard<std::mutex> lock(worker_mutex_);
    if (diagnostic_ && frame_id % 300 == 0) {
        std::cout << "[VOICE][HEALTH] frame=" << frame_id
                  << " state=" << module_state_name(module_state_.load())
                  << " current=" << (in_flight_key_.empty() ? "none" : in_flight_key_)
                  << " latest=" << (pending_action_.empty() ? action : pending_action_)
                  << " tx=" << tx_count_.load()
                  << " accepted=" << rx_accepted_count_.load()
                  << " completed=" << rx_completed_count_.load()
                  << " rejected=" << rx_rejected_count_.load()
                  << " ack_timeout=" << ack_timeout_count_.load()
                  << " play_timeout=" << play_timeout_count_.load()
                  << " rx_bytes=" << rx_byte_count_.load()
                  << " recoveries=" << recovery_count_.load()
                  << std::endl;
    }
    pending_frame_id_ = frame_id;
    pending_action_ = action;
    pending_key_ = action;
    pending_ready_ = true;
    worker_cv_.notify_one();
}

void VoiceNotifier::CloseBackend()
{
    rx_queue_.clear();
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    if (uart_ != nullptr) {
        uart_close(uart_);
        uart_ = nullptr;
    }
    if (gpio_ != nullptr) {
        gpio_close(gpio_);
        gpio_ = nullptr;
    }
}

void VoiceNotifier::Release()
{
    {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        worker_stop_ = true;
        pending_ready_ = false;
    }
    worker_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    CloseBackend();
    mode_ = Mode::Disabled;
}

}  // namespace obstacle
