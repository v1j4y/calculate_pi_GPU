cmake_minimum_required (VERSION 2.8)
PROJECT(calcpigpu C CXX)
set(CMAKE_VERBOSE_MAKEFILE OFF)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Wall -Wextra -pedantic")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wextra -pedantic")
SET(CMAKE_SHARED_LINKER_FLAGS "-shared")


SET(CMAKE_INCLUDE_CURRENT_DIR ON)

SET(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(PythonInterp REQUIRED)

cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")


option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/lib/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

if( ("${CMAKE_VERSION}" VERSION_EQUAL 3.8) OR
        ("${CMAKE_VERSION}" VERSION_GREATER 3.8) )
    # Modern CMake
    if(USE_CUDA)
        enable_language("CUDA")
        add_executable(calcpigpu ${cpu_source_files} ${gpu_source_files})
    else(USE_CUDA)
        add_executable(calcpigpu ${cpu_source_files})
    endif()
else()
    # Old CMake
    add_executable(calcpigpu ${cpu_source_files})
    if(USE_CUDA)
        find_package(CUDA 8.0)
        if(NOT CUDA_FOUND)
            message(STATUS "CUDA not found")
            set(USE_CUDA OFF)
        else()
            CUDA_ADD_LIBRARY(calcpigpu_gpu ${gpu_source_files})
            target_link_libraries(calcpigpu calcpigpu_gpu)
        endif()
    endif()
endif()

# Find source files

FILE(GLOB_RECURSE calcpigpu_SOURCES "src/*.cpp" "src/*.cu" "lib/*.cpp")
FILE(GLOB_RECURSE calcpigpu_HEADERS "lib/*.h")

SET(calcpigpu_INCLUDE_DIRS "")
ForEach (_headerFile ${calcpigpu_HEADERS})
    Get_Filename_Component(_dir ${_headerFile} PATH)
    LIST(APPEND calcpigpu_INCLUDE_DIRS ${_dir})
ENDForEach()
LIST(REMOVE_DUPLICATES calcpigpu_INCLUDE_DIRS)

## OpenBLAS
#set(BLAS_DIR /opt/OpenBLAS-develop/)
#FIND_PACKAGE(OpenBLAS REQUIRED)
#FIND_PATH(BLAS_INCLUDE_DIRS cblas.h
#  /opt/OpenBLAS-develop/include)

INCLUDE_DIRECTORIES(${calcpigpu_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# OpenMP
FIND_PACKAGE(OpenMP)
IF (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER clang++)

# copy library
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Create shared library
#ADD_LIBRARY(calcpigpu SHARED ${calcpigpu_SOURCES})

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
