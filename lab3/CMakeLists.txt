cmake_minimum_required (VERSION 2.6)
project(lab3)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message(${CMAKE_SYSTEM_NAME})
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(WINDOWS TRUE)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(LINUX TRUE)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)
endif()

if (${CMAKE_CL_64})
  set(ARCH_64 TRUE)
  set(PROCESSOR_ARCH "x64")
  message(64)
endif(${CMAKE_CL_64})
#####################################################
find_package(CUDA 10.1)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lnvToolExt -gencode arch=compute_52,code=sm_52)

if (WINDOWS)
set(CUDA_TOOLKIT_SAMPLES_DIR "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1")
add_definitions(-DBUILD_THIRDPARTY:BOOL=ON)
else (WINDOWS)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
	
set(CUDA_TOOLKIT_SAMPLES_DIR ${CUDA_TOOLKIT_ROOT_DIR}/samples)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
endif(WINDOWS)
###################################################################
find_package( NVTX REQUIRED )
#find_package( NVML REQUIRED )
find_package( GLEW REQUIRED )
find_package( FREEGLUT REQUIRED )
set(BUILD_THIRDPARTY ON)

add_subdirectory (thirdparty)
add_subdirectory (Src)

