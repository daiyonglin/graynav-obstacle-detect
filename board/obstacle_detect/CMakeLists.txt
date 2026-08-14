cmake_minimum_required(VERSION 3.0.0)

project(ssne_ai_demo)

include(cmake_config/Paths.cmake)

# 这四项构成板端模型契约。更换 .m1model 时必须同步核对输入通道、分类头
# 通道数和语音链接开关；否则模型虽然可能加载成功，CPU 后处理仍会错读 tensor。
option(A1_ENABLE_VOICE "Compile UART voice output support" ON)
option(A1_BUILD_SURFACE_TESTS "Build board-side surface postprocess/fusion test" OFF)
option(A1_REQUIRE_MODEL "Fail deploy builds when the unified model is absent" ON)
set(A1_YOLO_NUM_CLASSES "8" CACHE STRING "Unified indoor detector raw class count")
set(A1_YOLO_INPUT_CHANNELS "1" CACHE STRING "Unified model input channels")
set(A1_MODEL_FILENAME "graynav_unified_indoor8_scene21.m1model" CACHE STRING "Unified model filename under /app_demo/app_assets/models")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
add_compile_options(-Wno-psabi)

if(A1_ENABLE_VOICE)
    set(A1_ENABLE_VOICE_DEFINE 1)
else()
    set(A1_ENABLE_VOICE_DEFINE 0)
endif()

add_definitions(
    -DA1_ENABLE_VOICE=${A1_ENABLE_VOICE_DEFINE}
    -DA1_YOLO_NUM_CLASSES=${A1_YOLO_NUM_CLASSES}
    -DA1_YOLO_INPUT_CHANNELS=${A1_YOLO_INPUT_CHANNELS}
    -DA1_MODEL_FILENAME=\"${A1_MODEL_FILENAME}\"
)

set(OBSTACLE_DETECT_SOURCES
    # 主循环和各算法模块显式列出，不使用 GLOB；这样旧实验源文件不会被误编译。
    ${PROJECT_SOURCE_DIR}/demo_obstacle.cpp
    ${PROJECT_SOURCE_DIR}/src/osd-device.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline_image.cpp
    ${PROJECT_SOURCE_DIR}/src/avoidance_planner.cpp
    ${PROJECT_SOURCE_DIR}/src/guidance_stabilizer.cpp
    ${PROJECT_SOURCE_DIR}/src/ranging.cpp
    ${PROJECT_SOURCE_DIR}/src/semantic_config.cpp
    ${PROJECT_SOURCE_DIR}/src/surface_fusion.cpp
    ${PROJECT_SOURCE_DIR}/src/surface_segmentation.cpp
    ${PROJECT_SOURCE_DIR}/src/tracker.cpp
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/yolov8_gray.cpp
)

if(A1_ENABLE_VOICE)
    # 语音关闭时不编译 UART/GPIO 依赖，可用于纯 OSD 定位问题。
    list(APPEND OBSTACLE_DETECT_SOURCES ${PROJECT_SOURCE_DIR}/src/voice_notifier.cpp)
endif()

add_executable(ssne_ai_demo
    ${OBSTACLE_DETECT_SOURCES}
)

if(A1_BUILD_SURFACE_TESTS)
    add_executable(surface_logic_test
        ${PROJECT_SOURCE_DIR}/tests/test_surface_logic.cpp
        ${PROJECT_SOURCE_DIR}/src/guidance_stabilizer.cpp
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

# initramfs/reboot.bin has a 16 MiB flashing limit. Install only the unified
# model and the fixed-budget monochrome HUD assets used by this executable.
install(FILES ${PROJECT_SOURCE_DIR}/app_assets/colorLUT.sscl DESTINATION app_assets)
set(A1_HUD_ASSETS
    STOP SLOW CLEAR LEFT RIGHT
    PERSON CHAIR TABLE BAG COUCH BENCH ITEM
    STAIR BLOCKED PATH UNKNOWN AI_FAIL)
foreach(A1_HUD_NAME ${A1_HUD_ASSETS})
    install(FILES ${PROJECT_SOURCE_DIR}/app_assets/osd/${A1_HUD_NAME}.ssbmp
            DESTINATION app_assets/osd)
endforeach()
file(GLOB A1_HUD_INFO_ASSETS
    "${PROJECT_SOURCE_DIR}/app_assets/osd/INFO_*.ssbmp")
list(SORT A1_HUD_INFO_ASSETS)
if(NOT A1_HUD_INFO_ASSETS)
    message(FATAL_ERROR "Two-line INFO_ HUD assets are missing; run tools/generate_osd_ssbmp.py --info-only")
endif()
install(FILES ${A1_HUD_INFO_ASSETS} DESTINATION app_assets/osd)
install(DIRECTORY DESTINATION app_assets/models)
set(A1_MODEL_PATH ${PROJECT_SOURCE_DIR}/app_assets/models/${A1_MODEL_FILENAME})
if(EXISTS ${A1_MODEL_PATH})
    install(FILES ${A1_MODEL_PATH} DESTINATION app_assets/models)
elseif(A1_REQUIRE_MODEL)
    message(FATAL_ERROR
        "Unified model is missing: ${A1_MODEL_PATH}. "
        "Do not produce a flashable image until the trained model is converted.")
else()
    message(WARNING
        "Compile-only build: unified model is absent and no deployable image may be used.")
endif()
