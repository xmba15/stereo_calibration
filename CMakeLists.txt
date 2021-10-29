cmake_minimum_required(VERSION 3.10)

project(stereo_rectification)
set (CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(LIBRARY_NAME ${PROJECT_NAME})

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include(Findrapidjson)
if(NOT RAPIDJSON_FOUND)
  message(STATUS "rapidjson not found")
  return()
endif()

add_subdirectory(src)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_EXAMPLES)
