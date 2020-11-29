cmake_minimum_required(VERSION 2.6)
project(imageBasics)

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11")

set( CMAKE_BUILD_TYPE "Debug")

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(imageBasics main.cpp)

install(TARGETS imageBasics RUNTIME DESTINATION bin)

# 链接OpenCV库
target_link_libraries( imageBasics ${OpenCV_LIBS} )