cmake_minimum_required(VERSION 3.8.1)
project(dualRoboNet)
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

add_subdirectory(clBP)
add_subdirectory(FCL)
add_compile_definitions(DEBUG)

option(BUILD_TRACKING "Build the robot tracking executable" OFF)

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED)

add_executable(dualRoboNet main.cpp serialib.cpp neural.cpp LowPassFilter.hpp LowPassFilter.cpp neural.h external.cpp 
external.h bandpass.cpp)

target_link_libraries(dualRoboNet clBP FCL ${OpenCV_LIBS})

if(BUILD_TRACKING)
  add_executable(camtrack tracking/Source.cpp)
  target_link_libraries(camtrack ${OpenCV_LIBS})
endif()




