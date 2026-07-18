/*
 * @Author: Jingwen Bai
 * @Date: 2024-07-04 11:07:00
 * @Description: osd device
 * @Filename: osd-device.cpp
 */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "../include/osd-device.hpp"

using namespace fdevice;

namespace sst {
namespace device {
namespace osd {

OsdDevice::OsdDevice()
    : m_osd_handle(INVALID_HANDLE),
      m_height(0),
      m_width(0) {}

OsdDevice::~OsdDevice()
{
    std::cout << "OsdDevice Destructor" << std::endl;
}

void OsdDevice::Initialize(int width, int height)
{
    // 图层与 DMA 在启动时一次创建，运行期间只更新内容，避免逐帧申请连续内存。
    m_width = width;
    m_height = height;

    LoadLutFile(m_osd_lut_path.c_str());

    m_osd_handle = osd_open_device();
    osd_init_device(m_osd_handle, OSD_LAYER_SIZE, (char*)m_pcolor_lut);

    // 图层约定：0 危险条，1 动作词位图，2 方向/风险位图，3 走廊状态，4 检测框。
    const int quad_dma_size = 0x20000;
    const int image_dma_size = 0x40000;
    for (int layer_index = 0; layer_index < OSD_LAYER_SIZE; ++layer_index) {
        const bool image_layer = (layer_index == 1 || layer_index == 2);
        m_layer_is_image[layer_index] = image_layer;
        m_texture_disabled[layer_index] = false;

        const int dma_size = image_layer ? image_dma_size : quad_dma_size;
        osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma, dma_size);
        if (!image_layer) {
            osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma_2, dma_size);
        }

        const int dma_fd = osd_get_buffer_fd(m_osd_handle, m_layer_dma[layer_index].dma);

        LAYER_ATTR_S osd_layer;
        std::memset(&osd_layer, 0, sizeof(osd_layer));
        osd_layer.codeTYPE = image_layer ? SS_TYPE_RLE : SS_TYPE_QUADRANGLE;
        osd_layer.layerStart.layer_start_x = 0;
        osd_layer.layerStart.layer_start_y = 0;
        osd_layer.layerSize.layer_width = m_width;
        osd_layer.layerSize.layer_height = m_height;

        if (image_layer) {
            osd_layer.layer_data_RLE.osd_buf.buf_type = BUFFER_TYPE_DMABUF;
            osd_layer.layer_data_RLE.osd_buf.buf.fd_dmabuf = dma_fd;
            osd_layer.layer_rgn = {TYPE_IMAGE, {m_width, m_height}};
        } else {
            osd_layer.layer_data_QR.osd_buf.buf_type = BUFFER_TYPE_DMABUF;
            osd_layer.layer_data_QR.osd_buf.buf.fd_dmabuf = dma_fd;
            osd_layer.layer_rgn = {TYPE_GRAPHIC, {m_width, m_height}};
        }

        osd_create_layer(m_osd_handle, (ssLAYER_HANDLE)layer_index, &osd_layer);
        osd_set_layer_buffer(m_osd_handle, (ssLAYER_HANDLE)layer_index, m_layer_dma[layer_index]);
    }
}

void OsdDevice::Release()
{
    std::cout << "OsdDevice Release" << std::endl;

    for (int i = 0; i < OSD_LAYER_SIZE; ++i) {
        osd_destroy_layer(m_osd_handle, (ssLAYER_HANDLE)i);

        if (m_layer_dma[i].dma != nullptr) {
            osd_delete_buffer(m_osd_handle, m_layer_dma[i].dma);
            m_layer_dma[i].dma = nullptr;
        }
        if (m_layer_dma[i].dma_2 != nullptr) {
            osd_delete_buffer(m_osd_handle, m_layer_dma[i].dma_2);
            m_layer_dma[i].dma_2 = nullptr;
        }
    }

    if (m_pcolor_lut != nullptr) {
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
    }

    osd_close_device(m_osd_handle);
    m_osd_handle = INVALID_HANDLE;
}

int OsdDevice::LoadLutFile(const char* filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::in | std::ios::ate);
    if (!file) {
        std::cerr << "cannot open lut file " << filename << std::endl;
        return -1;
    }

    m_file_size = static_cast<int>(file.tellg());
    m_pcolor_lut = new uint8_t[m_file_size];
    file.seekg(0, std::ios::beg);
    file.read((char*)m_pcolor_lut, m_file_size);
    file.close();
    return 0;
}

void OsdDevice::Draw(std::vector<OsdQuadRangle>& quad_rangle)
{
    if (quad_rangle.empty()) {
        osd_clean_all_layer(m_osd_handle);
        return;
    }

    for (auto& q : quad_rangle) {
        GenQrangleBox(q.box, q.border);
        COVER_ATTR_S qrangle_attr = {q.color, q.type, q.alpha, m_qrangle_out, m_qrangle_in};
        osd_add_quad_rangle(m_osd_handle, &qrangle_attr);
    }

    osd_flush_quad_rangle(m_osd_handle);
}

void OsdDevice::Draw(std::vector<OsdQuadRangle>& quad_rangle, int layer_id)
{
    // 每次先清理目标矢量层再 flush 新图元，从根源上消除目标移出后的残留框。
    static bool warned_add[OSD_LAYER_SIZE] = {false};
    static bool warned_flush[OSD_LAYER_SIZE] = {false};

    if (layer_id < 0 || layer_id >= OSD_LAYER_SIZE) {
        std::cerr << "[OSD][WARN] invalid layer id " << layer_id << std::endl;
        return;
    }
    if (IsImageLayer(layer_id)) {
        std::cerr << "[OSD][WARN] layer " << layer_id << " is texture layer, skip quad draw" << std::endl;
        return;
    }

    osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (quad_rangle.empty()) {
        return;
    }

    for (auto& q : quad_rangle) {
        GenQrangleBox(q.box, q.border);
        COVER_ATTR_S qrangle_attr = {q.color, q.type, q.alpha, m_qrangle_out, m_qrangle_in};
        const int ret = osd_add_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &qrangle_attr);
        if (ret != 0 && !warned_add[layer_id]) {
            std::cerr << "[OSD][WARN] layer " << layer_id
                      << " quad add failed ret=" << ret
                      << ", count=" << quad_rangle.size() << std::endl;
            warned_add[layer_id] = true;
        }
    }

    const int flush_ret = osd_flush_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (flush_ret != 0 && !warned_flush[layer_id]) {
        std::cerr << "[OSD][WARN] layer " << layer_id
                  << " quad flush failed ret=" << flush_ret
                  << ", count=" << quad_rangle.size() << std::endl;
        warned_flush[layer_id] = true;
    }
}

void OsdDevice::Draw(std::vector<std::array<float, 4>>& boxes,
                     int border,
                     int layer_id,
                     tagQUADRANGLETYPE type,
                     tagALPHATYPE alpha,
                     int color)
{
    if (layer_id < 0 || layer_id >= OSD_LAYER_SIZE || IsImageLayer(layer_id)) {
        return;
    }
    if (boxes.empty()) {
        osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
        return;
    }

    osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
    for (auto& box : boxes) {
        GenQrangleBox(box, border);
        COVER_ATTR_S qrangle_attr = {color, type, alpha, m_qrangle_out, m_qrangle_in};
        osd_add_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &qrangle_attr);
    }

    osd_flush_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
}

bool OsdDevice::DrawTexture(const std::string& filename, int x, int y, int layer_id)
{
    // 位图层加载失败后本次运行禁用该层，防止连续失败拖慢主循环并刷屏日志。
    static bool warned_add[OSD_LAYER_SIZE] = {false};
    static bool warned_flush[OSD_LAYER_SIZE] = {false};

    if (layer_id < 0 || layer_id >= OSD_LAYER_SIZE) {
        std::cerr << "[OSD][WARN] invalid texture layer id " << layer_id << std::endl;
        return false;
    }
    if (!IsImageLayer(layer_id)) {
        std::cerr << "[OSD][WARN] layer " << layer_id << " is not texture layer" << std::endl;
        return false;
    }
    if (m_texture_disabled[layer_id]) {
        return false;
    }

    osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);

    BITMAP_INFO_S bm_info;
    bm_info.pSSbmpFile = filename.c_str();
    bm_info.alpha = TYPE_ALPHA100;
    bm_info.position = {x, y};

    const int ret = osd_add_texture_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &bm_info);
    if (ret != 0) {
        if (!warned_add[layer_id]) {
            std::cerr << "[OSD][WARN] texture add failed layer=" << layer_id
                      << " ret=" << ret << " path=" << filename << std::endl;
            warned_add[layer_id] = true;
        }
        m_texture_disabled[layer_id] = true;
        osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
        return false;
    }

    const int flush_ret = osd_flush_texture_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (flush_ret != 0) {
        if (!warned_flush[layer_id]) {
            std::cerr << "[OSD][WARN] texture flush failed layer=" << layer_id
                      << " ret=" << flush_ret << std::endl;
            warned_flush[layer_id] = true;
        }
        m_texture_disabled[layer_id] = true;
        osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
        return false;
    }

    return true;
}

void OsdDevice::CleanLayer(int layer_id)
{
    if (layer_id < 0 || layer_id >= OSD_LAYER_SIZE) {
        return;
    }
    osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
}

void OsdDevice::GenQrangleBox(std::array<float, 4>& det, int border)
{
    // 由 xyxy 框生成内外两个四边形，OSD 用两者之间的环带绘制空心边框。
    std::array<int, 16> box;

    box[0] = std::min(m_width, std::max(0, int(det[0] + border)));
    box[1] = std::min(m_height, std::max(0, int(det[1] + border)));
    box[2] = std::min(m_width, std::max(0, int(det[0] + border)));
    box[3] = std::min(m_height, std::max(0, int(det[3] - border)));
    box[4] = std::min(m_width, std::max(0, int(det[2] - border)));
    box[5] = std::min(m_height, std::max(0, int(det[3] - border)));
    box[6] = std::min(m_width, std::max(0, int(det[2] - border)));
    box[7] = std::min(m_height, std::max(0, int(det[1] + border)));

    box[8] = std::min(m_width, std::max(0, int(det[0] - border)));
    box[9] = std::min(m_height, std::max(0, int(det[1] - border)));
    box[10] = std::min(m_width, std::max(0, int(det[0] - border)));
    box[11] = std::min(m_height, std::max(0, int(det[3] + border)));
    box[12] = std::min(m_width, std::max(0, int(det[2] + border)));
    box[13] = std::min(m_height, std::max(0, int(det[3] + border)));
    box[14] = std::min(m_width, std::max(0, int(det[2] + border)));
    box[15] = std::min(m_height, std::max(0, int(det[1] - border)));

    m_qrangle_in.points[0] = {box[0], box[1]};
    m_qrangle_in.points[1] = {box[2], box[3]};
    m_qrangle_in.points[2] = {box[4], box[5]};
    m_qrangle_in.points[3] = {box[6], box[7]};
    m_qrangle_out.points[0] = {box[8], box[9]};
    m_qrangle_out.points[1] = {box[10], box[11]};
    m_qrangle_out.points[2] = {box[12], box[13]};
    m_qrangle_out.points[3] = {box[14], box[15]};
}

bool OsdDevice::IsImageLayer(int layer_id) const
{
    return layer_id >= 0 && layer_id < OSD_LAYER_SIZE && m_layer_is_image[layer_id];
}

} // namespace osd
} // namespace device
} // namespace sst
