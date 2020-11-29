cmake_minimum_required(VERSION 2.6)
project(useSophus)

# 为使用 sophus，您需要使用find_package命令找到它
find_package( Sophus REQUIRED)
include_directories( ${Sophus_INCLUDE_DIRS} )

add_executable(useSophus main.cpp)
target_link_libraries( useSophus ${Sophus_LIBRARIES} )

install(TARGETS useSophus RUNTIME DESTINATION bin)