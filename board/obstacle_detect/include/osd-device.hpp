/*
 * @Author: Jingwen Bai
 * @Date: 2024-07-04 11:07:00
 * @Description: 
 * @Filename: osd-device.hpp
 */
#ifndef SST_OSD_DEVICE_HPP_
#define SST_OSD_DEVICE_HPP_

#include <vector>
#include <string>

#include "osd_lib_api.h"
#include "common.hpp"

#define BUFFER_TYPE_DMABUF  0x1
#define OSD_LAYER_SIZE 5

namespace sst{
namespace device{
namespace osd{

typedef struct {
    std::array<float, 4> box;
    int border;
    int layer_id;
    fdevice::QUADRANGLETYPE type;
    fdevice::ALPHATYPE alpha;
    int color;
}OsdQuadRangle;

/**
 * @brief A1 OSD 硬件抽象层。
 *
 * 五个硬件图层在初始化时一次分配 DMA buffer：动作/风险文字使用 RLE 位图层，
 * 检测框和状态条使用矢量四边形层。上层 VISUALIZER 只提交逻辑图元，不直接
 * 操作 OSD 句柄、DMA 或颜色查找表。
 */
class OsdDevice {
public:
    OsdDevice();
    ~OsdDevice();

    /** 打开 OSD 设备，加载颜色 LUT，并为五个固定图层一次性分配 DMA。 */
    void Initialize(int width, int height);

    /** 销毁所有图层和 DMA buffer，随后关闭 OSD 设备句柄。 */
    void Release();

    /** 兼容接口：向默认矢量层提交一组矩形；空集合会清空全部图层。 */
    void Draw(std::vector<OsdQuadRangle> &quad_rangle);

    /** 将 xyxy 框转换为硬件四边形并刷新指定矢量层。 */
    void Draw(std::vector<std::array<float, 4>>& boxes, int border, int layer_id, fdevice::QUADRANGLETYPE type, fdevice::ALPHATYPE alpha, int color);

    /** 清空指定层后提交结构化四边形，是检测框层的主要刷新入口。 */
    void Draw(std::vector<OsdQuadRangle> &quad_rangle, int layer_id);

    /** 在 RLE 图像层显示预生成 .ssbmp 动作/风险文字。 */
    bool DrawTexture(const std::string& filename, int x, int y, int layer_id);

    /** 清空单个图层，不影响 Aurora 原始灰度画面和其他 OSD 层。 */
    void CleanLayer(int layer_id);

private:
    /** 读取 A1 OSD 颜色查找表，供设备初始化时注册。 */
    int LoadLutFile(const char* filename);

    /** 根据检测框生成内外两组顶点，二者间环带形成空心边框。 */
    void GenQrangleBox(std::array<float, 4>& det, int border);

    /** 区分 RLE 纹理层和 quadrangle 矢量层，禁止混用硬件接口。 */
    bool IsImageLayer(int layer_id) const;

private:
    handle_t m_osd_handle;
    std::string m_osd_lut_path = "/app_demo/app_assets/colorLUT.sscl";
    // std::string m_texture_path = "/ai/imgs/test_24.ssbmp";
    uint8_t *m_pcolor_lut = nullptr;
    int m_file_size = 0;
    int m_height, m_width;
    
    fdevice::DMA_BUFFER_ATTR_S m_layer_dma[OSD_LAYER_SIZE];
    fdevice::VERTEXS_S m_qrangle_out={0}, m_qrangle_in={0};
    bool m_layer_is_image[OSD_LAYER_SIZE] = {false};
    bool m_texture_disabled[OSD_LAYER_SIZE] = {false};
};

} // namespace osd
} // namespace device
} // namespace sst

#endif // SST_OSD_DEVICE_HPP_
