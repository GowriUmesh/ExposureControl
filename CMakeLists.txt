cmake_minimum_required(VERSION 2.8.3)
project( ExposureControl )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
find_package(PkgConfig)
pkg_check_modules(YamlCpp yaml-cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
include_directories(${YamlCpp_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)


add_executable(ExposureControl main.cpp)
target_link_libraries(ExposureControl ${OpenCV_LIBS})
