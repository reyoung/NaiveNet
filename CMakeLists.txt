cmake_minimum_required(VERSION 3.7)
project(NaiveNet)
set(CMAKE_CXX_STANDARD 14)
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen/")
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easyloggingpp/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/better-enums/)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mnist/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Catch/single_include)
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DELPP_FEATURE_ALL)
set(EASYLOGGINGPP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/easyloggingpp/src/easylogging++.cc)
set(SOURCE_FILES ${EASYLOGGINGPP_SRC} graph/ComputationGraph.cpp misc/Error.h
        misc/InitFunction.h ops/SigmoidOp.cpp ops/FullyConnectedOp.cpp
        memory/TensorBuffer.cpp memory/TensorBuffer.h engine/Engine.cpp misc/Typedef.h
        ops/SoftmaxOp.cpp ops/CrossEntropyOp.cpp ops/MeanOp.cpp
        graph/compilers/RequestResource.cpp graph/compilers/ShapeInferer.cpp
        graph/compilers/Backward.cpp ops/SgdOp.cpp graph/compilers/Optimizer.cpp
        ops/LookupTableOp.cpp misc/CastEigen.h ops/EigenOp-inl.h ops/ErrorRateOp.cpp
        misc/InitELPP.cpp)
add_library(nnet SHARED ${SOURCE_FILES})
add_executable(NaiveNet main.cpp)
target_link_libraries(NaiveNet nnet)

enable_testing()
# unittests
add_executable(gc_test ops/GradientCheck_test.cpp)
target_link_libraries(gc_test nnet)