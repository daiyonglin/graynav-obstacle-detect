#include "../include/common.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>

/*
 * 图像采集层：只负责把 SC132GS 的 Y8 灰度帧稳定送入应用。
 * 模型使用的双 ROI、384x384 letterbox 和灰度增强属于 YOLOV8GRAY，不能
 * 放在这里，否则 Aurora 显示坐标、异常检测画面和模型输入会失去统一基准。
 */
namespace {

int env_int_value(const char* name, int fallback, int min_value, int max_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return std::max(min_value, std::min(max_value, static_cast<int>(parsed)));
}

bool env_flag_value(const char* name, bool fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T';
}

std::string env_string_value(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::string(value);
}

} // namespace

void IMAGEPROCESSOR::Initialize(std::array<int, 2>* in_img_shape)
{
    // img_shape 按 [width,height] 保存；SSNE_Y_8 表示每像素 8 bit 单通道。
    img_shape = *in_img_shape;
    format_online = SSNE_Y_8;
    pipeline_open = false;
    ConfigureAndOpen();
}

bool IMAGEPROCESSOR::ConfigureAndOpen()
{
    /*
     * online pipeline 的裁剪参数属于传感器输出层。正式配置为 720x1280
     * 全画面；若 A1_CAPTURE_HEIGHT 或 CROP_Y0 被误改，裁剪外目标永远不会
     * 进入模型，所以启动时打印覆盖率并对部分裁剪给出显式警告。
     */
    const uint16_t img_width = static_cast<uint16_t>(img_shape[0]);
    const uint16_t img_height = static_cast<uint16_t>(img_shape[1]);
    const int full_width = env_int_value("A1_FULL_FRAME_WIDTH", 720, 1, 4096);
    const int full_height = env_int_value("A1_FULL_FRAME_HEIGHT", 1280, 1, 4096);
    const int crop_x0 = env_int_value("A1_CAPTURE_CROP_X0", 0, 0, full_width - 1);
    const int crop_y0 = env_int_value("A1_CAPTURE_CROP_Y0", 0, 0, full_height - 1);
    const int crop_x1 = std::min(full_width, crop_x0 + static_cast<int>(img_width));
    const int crop_y1 = std::min(full_height, crop_y0 + static_cast<int>(img_height));
    const int crop_w = std::max(0, crop_x1 - crop_x0);
    const int crop_h = std::max(0, crop_y1 - crop_y0);

    int ret = OnlineSetCrop(kPipeline0,
                            static_cast<uint16_t>(crop_x0),
                            static_cast<uint16_t>(crop_x1),
                            static_cast<uint16_t>(crop_y0),
                            static_cast<uint16_t>(crop_y1));
    if (ret != 0) {
        printf("[IMAGEPROCESSOR][ERROR] OnlineSetCrop failed, ret=%d, crop=(%d,%d,%d,%d)\n",
               ret, crop_x0, crop_y0, crop_x1, crop_y1);
        pipeline_open = false;
        return false;
    }

    ret = OnlineSetOutputImage(kPipeline0, format_online, img_width, img_height);
    if (ret != 0) {
        printf("[IMAGEPROCESSOR][ERROR] OnlineSetOutputImage failed, ret=%d\n", ret);
        pipeline_open = false;
        return false;
    }

    ret = OpenOnlinePipeline(kPipeline0);
    if (ret != 0) {
        printf("[IMAGEPROCESSOR][ERROR] OpenOnlinePipeline failed, ret=%d\n", ret);
        pipeline_open = false;
        return false;
    }

    pipeline_open = true;
    printf("[IMAGEPROCESSOR][INFO] open online pipeline0 success\n");
    printf("[IMAGEPROCESSOR][INFO] gray crop enabled: crop=(%d,%d,%d,%d), output=%dx%d, full=%dx%d\n",
           crop_x0, crop_y0, crop_x1, crop_y1,
           img_width, img_height, full_width, full_height);
    printf("[IMAGEPROCESSOR][INFO] crop coverage: %.1f%% width, %.1f%% height, %.1f%% area\n",
           100.0 * static_cast<double>(crop_w) / static_cast<double>(std::max(1, full_width)),
           100.0 * static_cast<double>(crop_h) / static_cast<double>(std::max(1, full_height)),
           100.0 * static_cast<double>(crop_w * crop_h) /
               static_cast<double>(std::max(1, full_width * full_height)));
    if (crop_h < full_height || crop_w < full_width) {
        printf("[IMAGEPROCESSOR][WARN] partial-frame crop is active; objects outside this rectangle never reach YOLO. "
               "Set A1_CAPTURE_HEIGHT=%d A1_CAPTURE_CROP_Y0=0 for full-view diagnosis.\n",
               full_height);
    }
    const int open_delay_ms = env_int_value("A1_CAPTURE_OPEN_DELAY_MS", 800, 0, 5000);
    if (open_delay_ms > 0) {
        usleep(static_cast<useconds_t>(open_delay_ms) * 1000);
    }
    return true;
}

bool IMAGEPROCESSOR::GetImage(ssne_tensor_t* img_sensor)
{
    /*
     * GetImageData 直接返回 SSNE tensor，不在主循环复制整帧。连续失败数
     * 只用于节流底层日志；系统级阈值和恢复动作由 demo_obstacle.cpp 中的
     * SystemHealth 统一管理。
     */
    static int consecutive_failures = 0;

    if (!pipeline_open) {
        consecutive_failures++;
        if (consecutive_failures <= 3 || consecutive_failures % 30 == 0) {
            printf("[IMAGEPROCESSOR][ERROR] GetImage skipped because pipeline is closed, consecutive=%d\n",
                   consecutive_failures);
        }
        return false;
    }

    const int capture_code = GetImageData(img_sensor, kPipeline0, kSensor0, 0);
    if (capture_code != 0) {
        consecutive_failures++;
        if (consecutive_failures <= 3 || consecutive_failures % 30 == 0) {
            printf("[IMAGEPROCESSOR][ERROR] GetImageData failed, ret=%d, consecutive=%d\n",
                   capture_code, consecutive_failures);
        }
        return false;
    }

    if (consecutive_failures > 0) {
        printf("[IMAGEPROCESSOR][INFO] image capture recovered after %d failed frame(s)\n",
               consecutive_failures);
        consecutive_failures = 0;
    }

    static bool capture_dumped = false;
    if (env_flag_value("A1_DUMP_CAPTURE_ONCE", false) && !capture_dumped) {
        const std::string dump_path = env_string_value("A1_CAPTURE_DUMP_PATH", "/tmp/a1_capture_crop_y8.bin");
        const int save_ret = save_tensor_buffer(*img_sensor, dump_path.c_str());
        printf("[IMAGEPROCESSOR][DEBUG] capture crop dump ret=%d path=%s\n",
               save_ret, dump_path.c_str());
        capture_dumped = true;
    }
    return true;
}

bool IMAGEPROCESSOR::Restart()
{
    // 先关闭旧 pipeline，等待硬件状态释放，再按同一参数重新配置并打开。
    printf("[IMAGEPROCESSOR][WARN] restarting online pipeline after capture failures\n");
    if (pipeline_open) {
        CloseOnlinePipeline(kPipeline0);
        pipeline_open = false;
        usleep(200000);
    }
    const bool ok = ConfigureAndOpen();
    usleep(ok ? 200000 : 300000);
    return ok;
}

void IMAGEPROCESSOR::Release()
{
    if (pipeline_open) {
        CloseOnlinePipeline(kPipeline0);
        pipeline_open = false;
        printf("[IMAGEPROCESSOR][INFO] Online pipeline closed\n");
    }
}
