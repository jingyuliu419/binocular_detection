cmake_minimum_required(VERSION 3.0)

project(OPENCV)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
add_compile_options(-Wall)

set(OpenCV_DIR "/home/liu/opencv_build/OpenCV")

find_package(OpenCV REQUIRED)

aux_source_directory(${PROJECT_SOURCE_DIR}/src srcFiles)#源文件列表存入变量srcFiles

include_directories(${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

link_directories(/home/liu/Doc/software/TensorRT-8.4.3.1/lib)
link_directories(/home/liu/Doc/software/onnxruntime/build/Linux/Release)

# add_executable(compressionDepth ${PROJECT_SOURCE_DIR}/src/main.cpp ${PROJECT_SOURCE_DIR}/src/depthCompute.cpp ${PROJECT_SOURCE_DIR}/src/HRNet-W32.cpp)
add_executable(compressionDepth ${srcFiles})
target_link_libraries(compressionDepth ${OpenCV_LIBS} onnxruntime onnxruntime_providers_shared onnxruntime_providers_cuda custom_op_library onnxruntime_providers_tensorrt nvonnxparser pthread)
# target_link_libraries(compressionDepth ${OpenCV_LIBS} onnxruntime)

