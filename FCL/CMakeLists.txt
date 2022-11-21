cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

 set(CMAKE_THREAD_LIBS_INIT "-lpthread")
 Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
 set(CMAKE_HAVE_THREADS_LIBRARY 1)
 set(CMAKE_USE_WIN32_THREADS_INIT 0)
 set(CMAKE_USE_PTHREADS_INIT 1)
 set(THREADS_PREFER_PTHREAD_FLAG ON)

project (FCL)

add_library(FCL
        
        fclNet/bandpass.cpp
        fclNet/neuron.cpp
        fclNet/layer.cpp
        fcl.cpp
        fcl_util.cpp
        )	

target_include_directories(FCL PUBLIC include)
