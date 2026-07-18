# cmake_minimum_required(VERSION 3.0.0)

# project(ssne_ai_demo)

# # 引入路径配置文件
# include(cmake_config/Paths.cmake)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
# # 核心：禁用ABI变更提示
# add_compile_options(-Wno-psabi)

# # add_executable(ssne_ai_demo ${PROJECT_SOURCE_DIR}/online_ped_osd_thread.cpp)
# # add_executable(ssne_ai_demo ${PROJECT_SOURCE_DIR}/online_ped_osd.cpp)
# add_executable(ssne_ai_demo ${PROJECT_SOURCE_DIR}/demo_obstacle.cpp)
# # add_executable(ssne_ai_demo ${PROJECT_SOURCE_DIR}/test_isp_debug.cpp)

# # 添加头文件目录
# include_directories(${M1_SDK_INC_DIR})
# include_directories(${M1_SDK_INC_DIR}/smartsoc)

# # 定义调用目录来源
# set(SRC_DIR "src")
# set(TARGET $ENV{BASE_DIR}/target)
# include_directories(${CMAKE_SOURCE_DIR}/include)
# message([<mode>] "message to display" ${TARGET})


# # 添加source文件夹下的所有cpp文件
# file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_DIR}/*.cpp")
# target_sources(ssne_ai_demo PRIVATE ${SRC_FILES})

# target_link_libraries(${PROJECT_NAME} 
#                         ${M1_SSNE_LIB}
#                         ${M1_CMABUFFER_LIB}
#                         ${M1_OSD_LIB}
#                         ${M1_SSZLOG_LIB}
#                         ${M1_ZLOG_LIB}
#                         ${M1_EMB_LIB}
# )

# install(TARGETS ssne_ai_demo DESTINATION bin)


cmake_minimum_required(VERSION 3.0.0)

project(ssne_ai_demo)

include(cmake_config/Paths.cmake)

option(A1_ENABLE_VOICE "Compile UART voice output support" ON)
set(A1_YOLO_NUM_CLASSES "25" CACHE STRING "YOLO raw class count: 80 for COCO, 25 for ROD25, 8 for obstacle8")
set(A1_YOLO_INPUT_CHANNELS "1" CACHE STRING "YOLO model input channels: 3 for gray-copy models, 1 for true grayscale models")
set(A1_MODEL_FILENAME "graynav_rod25_gray1_dce_b3_head6.m1model" CACHE STRING "Model filename under /app_demo/app_assets/models")

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
    ${PROJECT_SOURCE_DIR}/demo_obstacle.cpp
    ${PROJECT_SOURCE_DIR}/src/osd-device.cpp
    ${PROJECT_SOURCE_DIR}/src/pipeline_image.cpp
    ${PROJECT_SOURCE_DIR}/src/avoidance_planner.cpp
    ${PROJECT_SOURCE_DIR}/src/ranging.cpp
    ${PROJECT_SOURCE_DIR}/src/semantic_config.cpp
    ${PROJECT_SOURCE_DIR}/src/tracker.cpp
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/yolov8_gray.cpp
)

if(A1_ENABLE_VOICE)
    list(APPEND OBSTACLE_DETECT_SOURCES ${PROJECT_SOURCE_DIR}/src/voice_notifier.cpp)
endif()

add_executable(ssne_ai_demo
    ${OBSTACLE_DETECT_SOURCES}
)

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

if(A1_ENABLE_VOICE)
    target_link_libraries(${PROJECT_NAME}
        ${M1_UART_LIB}
        ${M1_GPIO_LIB}
    )
endif()

install(TARGETS ssne_ai_demo DESTINATION bin)

# Keep the initramfs below the board flashing limit by installing only the
# selected runtime model instead of every historical .m1model under app_assets.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/app_assets/
        DESTINATION app_assets
        PATTERN "models" EXCLUDE)
install(DIRECTORY DESTINATION app_assets/models)
install(FILES ${PROJECT_SOURCE_DIR}/app_assets/models/${A1_MODEL_FILENAME}
        DESTINATION app_assets/models)
