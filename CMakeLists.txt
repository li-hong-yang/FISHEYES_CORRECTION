cmake_minimum_required(VERSION 3.4.1)

project(FISHEYES_CORRECTION)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include ${PROJECT_SOURCE_DIR}/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt

include_directories(/root/TensorRT-7.0.0.11/include)
link_directories(/root/TensorRT-7.0.0.11/lib)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -Wno-error=deprecated-declarations -Wno-deprecated-declarations -DOCRTEST")
# set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC  -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -Wno-error=deprecated-declarations -Wno-deprecated-declarations -DOCRTEST")
# file(GLOB_RECURSE CPP_FILE src/*.cpp) # 搜索出所有后缀名为".cpp"的文件

# file(GLOB SRC_CPP src/*.cpp)
# add_library(myplugins SHARED ${SRC_CPP})
# aux_source_directory(src SRC_LIST)
# file(GLOB SRC_CU src/*.cu)

# CUDA_ADD_EXECUTABLE(lp_corner_detect ${SRC_CU})

add_executable(lp_recognize fisheyes.cpp)
target_link_libraries(lp_recognize nvinfer cudart ${OpenCV_LIBS})
