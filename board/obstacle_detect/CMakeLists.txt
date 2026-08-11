cmake_minimum_required(VERSION 3.0.0)

project(ssne_ai_demo)

include(cmake_config/Paths.cmake)

# 这四项构成板端模型契约。更换 .m1model 时必须同步核对输入通道、分类头
# 通道数和语音链接开关；否则模型虽然可能加载成功，CPU 后处理仍会错读 tensor。
option(A1_ENABLE_VOICE "Compile UART voice output support" ON)
option(A1_ENABLE_SURFACE_SEG "Compile single-channel road-surface and depth model" ON)
option(A1_BUILD_SURFACE_TESTS "Build board-side surface postprocess/fusion test" OFF)
set(A1_YOLO_NUM_CLASSES "80" CACHE STRING "YOLO raw class count: 80 for COCO, 25 for ROD25, 8 for obstacle8")
set(A1_YOLO_INPUT_CHANNELS "3" CACHE STRING "YOLO model input channels: 3 for gray-copy models, 1 for true grayscale models")
set(A1_MODEL_FILENAME "yolov8n80_graycopy_head6.m1model" CACHE STRING "Model filename under /app_demo/app_assets/models")
set(A1_SEG_MODEL_FILENAME "graynav_surface_depth_e3_gray1.m1model" CACHE STRING "Surface-depth model filename under /app_demo/app_assets/models")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
add_compile_options(-Wno-psabi)

if(A1_ENABLE_VOICE)
    set(A1_ENABLE_VOICE_DEFINE 1)
else()
    set(A1_ENABLE_VOICE_DEFINE 0)
endif()

if(A1_ENABLE_SURFACE_SEG)
    set(A1_ENABLE_SURFACE_SEG_DEFINE 1)
else()
    set(A1_ENABLE_SURFACE_SEG_DEFINE 0)
endif()

add_definitions(
    -DA1_ENABLE_VOICE=${A1_ENABLE_VOICE_DEFINE}
    -DA1_YOLO_NUM_CLASSES=${A1_YOLO_NUM_CLASSES}
    -DA1_YOLO_INPUT_CHANNELS=${A1_YOLO_INPUT_CHANNELS}
    -DA1_MODEL_FILENAME=\"${A1_MODEL_FILENAME}\"
    -DA1_ENABLE_SURFACE_SEG=${A1_ENABLE_SURFACE_SEG_DEFINE}
    -DA1_SEG_MODEL_FILENAME=\"${A1_SEG_MODEL_FILENAME}\"
)

set(OBSTACLE_DETECT_SOURCES
    # 主循环和各算法模块显式列出，不使用 GLOB；这样旧实验源文件不会被误编译。
    ${PROJECT_SOURCE_DIR}/demo_obstacle.cpp
    ${PROJECT_SOURCE_DIR}/src/osd-device.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline_image.cpp
    ${PROJECT_SOURCE_DIR}/src/avoidance_planner.cpp
    ${PROJECT_SOURCE_DIR}/src/ranging.cpp
    ${PROJECT_SOURCE_DIR}/src/semantic_config.cpp
    ${PROJECT_SOURCE_DIR}/src/surface_fusion.cpp
    ${PROJECT_SOURCE_DIR}/src/tracker.cpp
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/yolov8_gray.cpp
)

if(A1_ENABLE_SURFACE_SEG)
    list(APPEND OBSTACLE_DETECT_SOURCES
        ${PROJECT_SOURCE_DIR}/src/surface_segmentation.cpp
    )
endif()

if(A1_ENABLE_VOICE)
    # 语音关闭时不编译 UART/GPIO 依赖，可用于纯 OSD 定位问题。
    list(APPEND OBSTACLE_DETECT_SOURCES ${PROJECT_SOURCE_DIR}/src/voice_notifier.cpp)
endif()

add_executable(ssne_ai_demo
    ${OBSTACLE_DETECT_SOURCES}
)

if(A1_BUILD_SURFACE_TESTS AND A1_ENABLE_SURFACE_SEG)
    add_executable(surface_logic_test
        ${PROJECT_SOURCE_DIR}/tests/test_surface_logic.cpp
        ${PROJECT_SOURCE_DIR}/src/surface_segmentation.cpp
        ${PROJECT_SOURCE_DIR}/src/surface_fusion.cpp
        ${PROJECT_SOURCE_DIR}/src/semantic_config.cpp
    )
endif()

include_directories(${M1_SDK_INC_DIR})
include_directories(${M1_SDK_INC_DIR}/smartsoc)
include_directories(${CMAKE_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME}
    ${M1_SSNE_LIB}
    ${M1_CMABUFFER_LIB}
    ${M1_OSD_LIB}
    ${M1_SSZLOG_LIB}
    ${M1_ZLOG_LIB}
    ${M1_EMB_LIB}
)

if(TARGET surface_logic_test)
    target_link_libraries(surface_logic_test
        ${M1_SSNE_LIB}
        ${M1_CMABUFFER_LIB}
        ${M1_OSD_LIB}
        ${M1_SSZLOG_LIB}
        ${M1_ZLOG_LIB}
        ${M1_EMB_LIB}
    )
endif()

if(A1_ENABLE_VOICE)
    target_link_libraries(${PROJECT_NAME}
        ${M1_UART_LIB}
        ${M1_GPIO_LIB}
    )
endif()

install(TARGETS ssne_ai_demo DESTINATION bin)
install(PROGRAMS ${PROJECT_SOURCE_DIR}/scripts/run_dual_model_smoke.sh DESTINATION bin)

# initramfs/reboot.bin 有 16 MiB 烧录限制，只安装当前模型；历史模型不得随
# app_assets 整目录进入镜像，否则会造成可执行文件正常但最终镜像无法烧录。
install(DIRECTORY ${PROJECT_SOURCE_DIR}/app_assets/
        DESTINATION app_assets
        PATTERN "models" EXCLUDE)
install(DIRECTORY DESTINATION app_assets/models)
install(FILES ${PROJECT_SOURCE_DIR}/app_assets/models/${A1_MODEL_FILENAME}
        DESTINATION app_assets/models)
if(A1_ENABLE_SURFACE_SEG)
    if(EXISTS ${PROJECT_SOURCE_DIR}/app_assets/models/${A1_SEG_MODEL_FILENAME})
        install(FILES ${PROJECT_SOURCE_DIR}/app_assets/models/${A1_SEG_MODEL_FILENAME}
                DESTINATION app_assets/models)
    else()
        message(WARNING "Surface model is not present; image will boot in detector-only fallback mode: ${A1_SEG_MODEL_FILENAME}")
    endif()
endif()
