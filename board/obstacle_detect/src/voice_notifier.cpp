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

// Reads a string environment variable with an explicit fallback.
std::string getenv_string(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::string(value);
}

// Reads an integer environment variable with an explicit fallback.
int getenv_int(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

// Reads a boolean environment variable using common true-like first characters.
bool getenv_bool(const char* name, bool fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T';
}

// Maps user-specified tty baud values to Linux termios constants.
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
      cooldown_ms_(1200),
      clear_repeat_ms_(1200),
      stop_repeat_ms_(900),
      fault_repeat_ms_(2000),
      fault_hold_ms_(200),
      switch_min_ms_(0),
      tx_gap_ms_(650),
      pre_stop_(false),
      ack_enabled_(true),
      require_ack_(false),
      query_idle_(true),
      fixed_frame_(true),
      use_prompt_prefix_(false),
      reopen_each_tx_(false),
      passive_rx_(false),
      diagnostic_(false),
      ack_timeout_ms_(500),
      idle_timeout_ms_(80),
      recover_wait_ms_(1000),
      retry_count_(0),
      baud_(9600),
      byte_gap_us_(1000),
      post_tx_delay_ms_(30),
      passive_rx_ms_(80),
      soft_reset_every_tx_(0),
      consecutive_no_rx_(0),
      consecutive_tx_failures_(0),
      tx_failure_count_(0),
      recovery_count_(0),
      tx_count_(0),
      rx_accepted_count_(0),
      rx_idle_count_(0),
      rx_rejected_count_(0),
      last_rx_code_(0),
      module_state_(ModuleState::Unknown),
      last_sent_frame_(-100000),
      last_action_(""),
      last_key_(""),
      last_tx_detail_("not_sent"),
      tty_device_("/dev/ttyS1"),
      last_sent_time_(std::chrono::steady_clock::now() - std::chrono::seconds(60)),
      last_fault_seen_time_(std::chrono::steady_clock::now() - std::chrono::seconds(60)),
      worker_stop_(false),
      pending_ready_(false),
      tx_in_flight_(false),
      pending_frame_id_(0) {}

bool VoiceNotifier::InitializeFromEnv()
{
    const std::string mode = getenv_string("A1_OUTPUT_MODE", "both");
    if (mode == "voice") {
        mode_ = Mode::VoiceOnly;
    } else if (mode == "both") {
        mode_ = Mode::Both;
    } else {
        mode_ = Mode::Disabled;
        return false;
    }

    frame_interval_ = std::max(1, getenv_int("A1_VOICE_INTERVAL_FRAMES", 2));
    stable_needed_ = std::max(1, getenv_int("A1_VOICE_STABLE_FRAMES", 2));
    clear_stable_needed_ = std::max(stable_needed_, getenv_int("A1_VOICE_CLEAR_STABLE_FRAMES", 3));
    cooldown_ms_ = std::max(600, getenv_int("A1_VOICE_COOLDOWN_MS", 1200));
    clear_repeat_ms_ = std::max(600, getenv_int("A1_VOICE_CLEAR_REPEAT_MS", 1200));
    stop_repeat_ms_ = std::max(600, getenv_int("A1_VOICE_STOP_REPEAT_MS", 900));
    fault_repeat_ms_ = std::max(1200, getenv_int("A1_VOICE_FAULT_REPEAT_MS", 2000));
    fault_hold_ms_ = std::max(0, getenv_int("A1_VOICE_FAULT_HOLD_MS", 200));
    switch_min_ms_ = std::max(0, getenv_int("A1_VOICE_SWITCH_MIN_MS", 0));
    tx_gap_ms_ = std::max(0, getenv_int("A1_VOICE_TX_GAP_MS", 650));
    pre_stop_ = getenv_bool("A1_VOICE_PRE_STOP", false);
    ack_enabled_ = getenv_bool("A1_VOICE_ACK", true);
    require_ack_ = getenv_bool("A1_VOICE_REQUIRE_ACK", false);
    query_idle_ = getenv_bool("A1_VOICE_QUERY_IDLE", true);
    fixed_frame_ = getenv_bool("A1_VOICE_FIXED_FRAME", true);
    use_prompt_prefix_ = getenv_bool("A1_VOICE_USE_PREFIX", false);
    reopen_each_tx_ = getenv_bool("A1_VOICE_REOPEN_EACH_TX", false);
    passive_rx_ = getenv_bool("A1_VOICE_PASSIVE_RX", false);
    diagnostic_ = getenv_bool("A1_VOICE_DIAG", false);
    ack_timeout_ms_ = std::max(20, getenv_int("A1_VOICE_ACK_TIMEOUT_MS", 500));
    idle_timeout_ms_ = std::max(20, getenv_int("A1_VOICE_IDLE_TIMEOUT_MS", 180));
    recover_wait_ms_ = std::max(80, getenv_int("A1_VOICE_RECOVER_WAIT_MS", 1000));
    retry_count_ = std::max(0, getenv_int("A1_VOICE_RETRY", 1));
    // The A1 UART driver is reliable with SYN6288 speech frames when bytes are
    // submitted individually. Keep this below the SYN6288 8 ms limit.
    byte_gap_us_ = std::max(0, getenv_int("A1_VOICE_BYTE_GAP_US", 1000));
    post_tx_delay_ms_ = std::max(9, getenv_int("A1_VOICE_POST_TX_DELAY_MS", 30));
    passive_rx_ms_ = std::max(0, getenv_int("A1_VOICE_PASSIVE_RX_MS", 80));
    soft_reset_every_tx_ = std::max(0, getenv_int("A1_VOICE_SOFT_RESET_EVERY_TX", 0));

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
              << " soft_reset_every_tx=" << soft_reset_every_tx_
              << " retry=" << retry_count_
                  << std::endl;
    } else {
        std::cout << "[VOICE][INFO] ready mode=" << mode
                  << " baud=" << baud
                  << " protocol=fixed_frame latest_action_mailbox=on"
                  << std::endl;
    }

    DrainRx(80);
    if (query_idle_) {
        uint8_t state = 0;
        if (QueryBusyState(&state)) {
            module_state_ = (state == 0x4F) ? ModuleState::Idle :
                            (state == 0x4E) ? ModuleState::Speaking :
                            ModuleState::Unknown;
            std::cout << "[VOICE][INFO] initial module state="
                      << hex_byte(state) << "(" << syn6288_status_name(state) << ")"
                      << std::endl;
        } else {
            std::cout << "[VOICE][WARN] initial module state query timed out; will use ACK/blind fallback." << std::endl;
        }
    }

    if (getenv_bool("A1_VOICE_TEST_ON_START", false)) {
        const bool ok = SendPrompt("clear", false);
        last_key_ = "clear";
        last_sent_time_ = std::chrono::steady_clock::now();
        std::cout << "[VOICE][TX] frame=0 action=clear reason=startup status="
                  << (ok ? "ok" : "fail")
                  << " detail=" << last_tx_detail_ << std::endl;
    }
    if (getenv_bool("A1_VOICE_SELFTEST", false)) {
        RunStartupSelfTest();
    }
    worker_stop_ = false;
    pending_ready_ = false;
    worker_ = std::thread(&VoiceNotifier::WorkerLoop, this);
    std::cout << "[VOICE][INFO] asynchronous UART worker started" << std::endl;
    return true;
}

bool VoiceNotifier::OpenA1UartApi(int baud)
{
    gpio_ = gpio_init();
    if (gpio_ == nullptr) {
        std::cout << "[VOICE][WARN] gpio_init failed. Is gpio_kmod.ko loaded?" << std::endl;
        return false;
    }

    // Official A1 UART API uses UART_TX0/RX0; configure PIN0/PIN2 to that mux.
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
    if (bytes.empty()) {
        return true;
    }

    if (backend_ == Backend::A1UartApi) {
        if (uart_ == nullptr) {
            return false;
        }
        if (byte_gap_us_ > 0) {
            for (size_t i = 0; i < bytes.size(); ++i) {
                if (uart_send_data(uart_, UART_TX0, &bytes[i], 1) != UART_SUCCESS) {
                    std::cout << "[VOICE][WARN] uart_send_data byte failed at " << i << std::endl;
                    return false;
                }
                usleep(static_cast<useconds_t>(byte_gap_us_));
            }
            if (post_tx_delay_ms_ > 0) {
                usleep(static_cast<useconds_t>(post_tx_delay_ms_) * 1000);
            }
            return true;
        }
        for (size_t offset = 0; offset < bytes.size(); offset += kA1UartFifoBytes) {
            const size_t chunk = std::min(kA1UartFifoBytes, bytes.size() - offset);
            if (uart_send_data(uart_, UART_TX0, &bytes[offset], static_cast<uint32_t>(chunk)) != UART_SUCCESS) {
                std::cout << "[VOICE][WARN] uart_send_data failed." << std::endl;
                return false;
            }
            if (offset + chunk < bytes.size()) {
                usleep(1000);
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

    // The module may have already emitted an asynchronous 0x4F after the
    // previous phrase. Consume only meaningful state bytes; ACK bytes and
    // electrical noise must not be mistaken for a busy/idle answer.
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
        // A rejected optional cancel frame must never suppress the actual
        // navigation instruction. Directly send the newest action unless
        // pre-stop is explicitly enabled for a verified module revision.
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
        bool recovery_sent = false;
        const int sent_count = tx_count_.load();
        if (action != "system_fault" && soft_reset_every_tx_ > 0 && sent_count > 0 &&
            sent_count % soft_reset_every_tx_ == 0) {
            recovery_sent = SendBytes(kSyn6288Stop);
            usleep(220000);
            DrainRx(40);
            ++recovery_count_;
        }
        const int no_rx_recover_threshold = std::max(
            3, getenv_int("A1_VOICE_NO_RX_RECOVER", 5));
        if (!recovery_sent && passive_rx_ &&
            consecutive_no_rx_.load() >= no_rx_recover_threshold) {
            // The module previously returned IDLE and then went silent in long
            // runs. Clear a potentially wedged synthesis transaction, retain
            // only the latest action, and immediately send that action again.
            recovery_sent = SendBytes(kSyn6288Stop) || recovery_sent;
            usleep(180000);
            DrainRx(40);
            consecutive_no_rx_ = 0;
            ++recovery_count_;
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
            bool recognized_status = false;
            bool receive_failed = false;
            while (std::chrono::steady_clock::now() < deadline) {
                if (ReadResponseByte(&value, 5)) {
                    if (!rx_detail.empty()) rx_detail += "/";
                    rx_detail += hex_byte(value) + "(" + syn6288_status_name(value) + ")";
                    rx_count++;
                    if (value == 0x41 || value == 0x4A || value == 0x4E || value == 0x4F) {
                        recognized_status = true;
                        last_rx_code_ = value;
                    }
                    if (value == 0x45) {
                        receive_failed = true;
                        last_rx_code_ = value;
                    }
                    continue;
                }
                usleep(5000);
            }
            if (rx_count == 0) {
                rx_detail = "none";
            }
            if (receive_failed) {
                // SYN6288 explicitly rejected the command. Clear the current
                // transaction and retry the newest action once; reporting UART
                // write success alone would otherwise hide a silent module.
                const bool stop_ok = SendBytes(kSyn6288Stop);
                usleep(260000);
                DrainRx(60);
                const bool reopen_retry_ok = ReopenBackend();
                const bool retry_ok = reopen_retry_ok && SendBytes(frame);
                rx_detail += std::string("/retry_stop=") + (stop_ok ? "ok" : "fail") +
                             ",retry_tx=" + (retry_ok ? "ok" : "fail");
                bool retry_rejected = false;
                if (retry_ok) {
                    uint8_t retry_status = 0;
                    if (ReadResponseByte(&retry_status, 120)) {
                        rx_detail += ",retry_rx=" + hex_byte(retry_status) +
                                     "(" + syn6288_status_name(retry_status) + ")";
                        recognized_status = retry_status == 0x41 || retry_status == 0x4A ||
                                            retry_status == 0x4E || retry_status == 0x4F;
                        retry_rejected = retry_status == 0x45;
                    }
                }
                if (!retry_ok || retry_rejected) {
                    last_tx_detail_ = pre_detail + "receive_failed," + rx_detail;
                    return false;
                }
            }
            consecutive_no_rx_ = recognized_status ? 0 : consecutive_no_rx_ + 1;
        } else {
            rx_detail = "off";
        }
        module_state_ = ModuleState::Unknown;
        last_tx_detail_ = pre_detail +
                          std::string("reopen=") + (reopen_each_tx_ ? "ok" : "skip") +
                          "," + (fixed_frame_ ? "fixed_frame:blind_tx" : "dynamic_frame:blind_tx") +
                          ",frame_hex=" + hex_bytes(frame) +
                          ",rx=" + rx_detail +
                          ",no_rx=" + std::to_string(consecutive_no_rx_.load()) +
                          ",recover=" + (recovery_sent ? "stop_sent" : "none");
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
            // A 0x45 means the speech frame itself must be resent. Sending a
            // stop frame here can also be rejected and previously left the
            // module permanently silent.
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

std::string VoiceNotifier::BuildVoiceKey(const DetectionResult& result,
                                         const AvoidanceDecision& decision) const
{
    (void)result;
    const std::string action = decision.action.empty() ? "clear" : decision.action;
    return action;
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
    const bool urgent_override = action == "stop" || action == "system_fault";
    // STOP and device faults may interrupt immediately. LEFT/RIGHT/SLOW must
    // obey the UART gap even when they represent a risk increase; otherwise
    // rapidly alternating planner states overload SYN6288 and cause 0x45.
    if (!urgent_override && since_ms < tx_gap_ms_) {
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

void VoiceNotifier::WorkerLoop()
{
    while (true) {
        int frame_id = 0;
        std::string action;
        std::string key;
        std::string reason;
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            worker_cv_.wait(lock, [this]() { return worker_stop_ || pending_ready_; });
            if (worker_stop_ && !pending_ready_) break;
            frame_id = pending_frame_id_;
            action = pending_action_;
            key = pending_key_;
            reason = pending_reason_;
            pending_ready_ = false;
            tx_in_flight_ = true;
            in_flight_key_ = key;
        }

        const bool changed_risk_action =
            (reason == "risk_changed" || reason == "changed_stable") && action != "clear";
        const bool interrupt = action == "stop" || action == "system_fault" || changed_risk_action;
        const bool ok = SendPrompt(action, interrupt);
        const bool busy_defer = !ok && last_tx_detail_.find("module_busy_pending") != std::string::npos;
        const bool superseded = !ok && last_tx_detail_.find("superseded_by_urgent") != std::string::npos;
        if (diagnostic_ || action == "system_fault" || (!ok && !busy_defer && !superseded) ||
            last_tx_detail_.find("receive_failed") != std::string::npos) {
            std::cout << (ok ? "[VOICE][TX]" : "[VOICE][WARN]")
                      << " frame=" << frame_id
                      << " action=" << action
                      << " reason=" << reason
                      << " status=" << (ok ? "ok" : "fail")
                      << " detail=" << last_tx_detail_ << std::endl;
        }
        if (ok) {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            consecutive_tx_failures_ = 0;
            CommitSent(action, key);
            last_sent_frame_ = frame_id;
            tx_in_flight_ = false;
            in_flight_key_.clear();
        } else if (superseded) {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            tx_in_flight_ = false;
            in_flight_key_.clear();
        } else if (busy_defer) {
            usleep(20000);
            std::lock_guard<std::mutex> lock(worker_mutex_);
            tx_in_flight_ = false;
            in_flight_key_.clear();
            if (!worker_stop_ && !pending_ready_) {
                pending_frame_id_ = frame_id;
                pending_action_ = action;
                pending_key_ = key;
                pending_reason_ = "wait_module_idle";
                pending_ready_ = true;
                worker_cv_.notify_one();
            }
        } else {
            ++consecutive_tx_failures_;
            ++tx_failure_count_;
            ++recovery_count_;
            usleep(150000);
            bool recovered = true;
            if (consecutive_tx_failures_.load() >= 3) {
                CloseBackend();
                recovered = ReopenBackend();
            }
            std::lock_guard<std::mutex> lock(worker_mutex_);
            tx_in_flight_ = false;
            in_flight_key_.clear();
            if (!worker_stop_ && !pending_ready_) {
                pending_frame_id_ = frame_id;
                pending_action_ = action;
                pending_key_ = key;
                pending_reason_ = recovered ? "uart_recovered_retry" : "uart_reopen_retry";
                pending_ready_ = true;
                worker_cv_.notify_one();
            }
        }
    }
}

void VoiceNotifier::Update(int frame_id,
                           const DetectionResult& result,
                           const AvoidanceDecision& decision)
{
    if (mode_ == Mode::Disabled) {
        return;
    }
    std::string action = decision.action.empty() ? "clear" : decision.action;
    const auto now = std::chrono::steady_clock::now();
    if (action == "system_fault") {
        last_fault_seen_time_ = now;
    } else {
        const int since_fault_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_fault_seen_time_).count());
        if (since_fault_ms < fault_hold_ms_) {
            action = "system_fault";
        }
    }
    if (action != "stop" && frame_id % frame_interval_ != 0) {
        return;
    }
    const std::string key = action;
    std::string reason;
    std::lock_guard<std::mutex> lock(worker_mutex_);
    if (frame_id % 300 == 0) {
        const int since_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - last_sent_time_).count());
        std::cout << "[VOICE][HEALTH] frame=" << frame_id
                  << " tx_count=" << tx_count_.load()
                  << " ack=" << rx_accepted_count_.load()
                  << " idle=" << rx_idle_count_.load()
                  << " reject=" << rx_rejected_count_.load()
                  << " tx_failures=" << tx_failure_count_.load()
                  << " consecutive_failures=" << consecutive_tx_failures_.load()
                  << " no_rx=" << consecutive_no_rx_.load()
                  << " recoveries=" << recovery_count_.load()
                  << " last_tx_age_ms=" << since_ms
                  << " pending=" << (pending_ready_ ? 1 : 0)
                  << " in_flight=" << (tx_in_flight_ ? 1 : 0)
                  << std::endl;
    }
    if ((pending_ready_ && pending_key_ == key) ||
        (tx_in_flight_ && in_flight_key_ == key)) {
        return;
    }
    if (ShouldSend(action, key, &reason)) {
        // Latest-state mailbox: a new decision replaces any unsent old one.
        pending_frame_id_ = frame_id;
        pending_action_ = action;
        pending_key_ = key;
        pending_reason_ = reason;
        pending_ready_ = true;
        worker_cv_.notify_one();
    }
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
