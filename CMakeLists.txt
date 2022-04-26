project(blazefacencnn)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR "D:/software/opencv_4.5_msvc/build")
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
file(GLOB HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(blazefacencnn ${SRCS} ${HEAD})

target_link_libraries(blazefacencnn ncnn ${OpenCV_LIBS})