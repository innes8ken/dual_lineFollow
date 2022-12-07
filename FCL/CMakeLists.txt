cmake_minimum_required(VERSION 3.8.1)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
set(CMAKE_CXX_STANDARD 14)
add_compile_definitions(DEBUG)

project (FCL)

add_library(FCL
        fclNet/layer.cpp
        fclNet/bandpass.cpp
        fclNet/neuron.cpp 
        fcl.cpp
        fcl_util.cpp
        )	

target_include_directories(FCL PUBLIC include)
