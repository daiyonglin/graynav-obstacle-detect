#include "../include/common.hpp"

#include <iostream>
#include <unistd.h>

void IMAGEPROCESSOR::Initialize(std::array<int, 2>* in_img_shape)
{
    img_shape = *in_img_shape;
    format_online = SSNE_Y_8;
    pipeline_open = false;
    ConfigureAndOpen();
}

bool IMAGEPROCESSOR::ConfigureAndOpen()
{
    const uint16_t img_width = static_cast<uint16_t>(img_shape[0]);
    const uint16_t img_height = static_cast<uint16_t>(img_shape[1]);

    int ret = OnlineSetOutputImage(kPipeline0, format_online, img_width, img_height);
    if (ret != 0) {
        printf("[IMAGEPROCESSOR][ERROR] OnlineSetOutputImage failed, ret=%d\n", ret);
        pipeline_open = false;
        return false;
    }

    ret = UpdateOnlineParam();
    if (ret != 0) {
        printf("[IMAGEPROCESSOR][ERROR] UpdateOnlineParam failed, ret=%d\n", ret);
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
    printf("[IMAGEPROCESSOR][INFO] full-frame gray image enabled: %dx%d\n",
           img_width, img_height);
    usleep(200000);
    return true;
}

bool IMAGEPROCESSOR::GetImage(ssne_tensor_t* img_sensor)
{
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
    return true;
}

bool IMAGEPROCESSOR::Restart()
{
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
