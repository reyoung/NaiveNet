cmake_minimum_required(VERSION 3.7)
project(NaiveNet)
set(CMAKE_CXX_STANDARD 14)
find_package(Boost REQUIRED)

INCLUDE_DIRECTORIES ( "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen/" )
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easyloggingpp/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/better-enums/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mnist/include)
add_definitions(-DELPP_FEATURE_ALL)
set(EASYLOGGINGPP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easyloggingpp/src/easylogging++.cc)
set(SOURCE_FILES ${EASYLOGGINGPP_SRC} main.cpp ComputationGraph.cpp Error.h
        Register.h SigmoidOp.cpp FullyConnectedOp.cpp
        TensorBuffer.cpp TensorBuffer.h Engine.cpp Typedef.h SoftmaxOp.cpp CrossEntropyOp.cpp MeanOp.cpp)
add_executable(NaiveNet ${SOURCE_FILES})