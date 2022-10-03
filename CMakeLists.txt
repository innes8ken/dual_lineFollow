cmake_minimum_required(VERSION 3.8.1)
project(dualRoboNet)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(clBP)

option(BUILD_TRACKING "Build the robot tracking executable" OFF)

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED)

add_executable(roboStartBCL main.cpp serialib.cpp neural.cpp LowPassFilter.hpp LowPassFilter.cpp neural.h external.cpp external.h bandpass.cpp)
#add_executable(roboStartFCL)

target_link_libraries(dualRoboNet clBP ${OpenCV_LIBS})

if(BUILD_TRACKING)
  add_executable(camtrack tracking/Source.cpp)
  target_link_libraries(camtrack ${OpenCV_LIBS})
endif()
