#include "smartsoc/ssne_api.h"

// Host-only link stubs for the pure CPU SurfaceDepth postprocess tests.
// The test binary never calls these functions.  Production/A1 builds must link
// the vendor SSNE runtime and must not compile this file.

uint16_t ssne_loadmodel(char*, uint8_t)
{
    return 0;
}

int ssne_get_model_input_dtype(uint16_t, int* dtype)
{
    if (dtype != nullptr) {
        *dtype = SSNE_FLOAT32;
    }
    return SSNE_ERRCODE_NO_ERROR;
}

int ssne_inference(uint16_t, uint8_t, ssne_tensor_t[])
{
    return SSNE_ERRCODE_JOB_ERROR;
}

int ssne_getoutput(uint16_t, uint8_t, ssne_tensor_t[])
{
    return SSNE_ERRCODE_OUTPUT_ERROR;
}

ssne_tensor_t create_tensor(uint32_t, uint32_t, uint8_t, ssne_buffer_type)
{
    ssne_tensor_t tensor{};
    return tensor;
}

int release_tensor(ssne_tensor_t)
{
    return SSNE_ERRCODE_NO_ERROR;
}

uint32_t get_total_size(ssne_tensor_t)
{
    return 0;
}

uint32_t get_width(ssne_tensor_t)
{
    return 0;
}

uint32_t get_height(ssne_tensor_t)
{
    return 0;
}

uint8_t get_data_type(ssne_tensor_t)
{
    return SSNE_FLOAT32;
}

void* get_data(ssne_tensor_t)
{
    return nullptr;
}

AiPreprocessPipe GetAIPreprocessPipe()
{
    return nullptr;
}

int ReleaseAIPreprocessPipe(AiPreprocessPipe)
{
    return SSNE_ERRCODE_NO_ERROR;
}

void Clear(AiPreprocessPipe)
{
}

int RunAiPreprocessPipe(AiPreprocessPipe, ssne_tensor_t, ssne_tensor_t)
{
    return SSNE_ERRCODE_INPUT_ERROR;
}

int SetCrop(AiPreprocessPipe, uint16_t, uint16_t, uint16_t, uint16_t)
{
    return SSNE_ERRCODE_NO_ERROR;
}

int SetNormalize(AiPreprocessPipe, uint16_t)
{
    return SSNE_ERRCODE_NO_ERROR;
}
