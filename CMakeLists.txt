cmake_minimum_required (VERSION 2.8)
PROJECT(calcpigpu C CXX)
set(CMAKE_VERBOSE_MAKEFILE OFF)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Wall -Wextra -pedantic")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wextra -pedantic")
SET(CMAKE_SHARED_LINKER_FLAGS "-shared")


SET(CMAKE_INCLUDE_CURRENT_DIR ON)

SET(CMAKE_BUILD_TYPE Debug)

find_package(PythonInterp REQUIRED)

# Find source files

FILE(GLOB_RECURSE calcpigpu_SOURCES "src/*.cpp")
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

#INCLUDE_DIRECTORIES(${calcpigpu_INCLUDE_DIRS} /opt/OpenBLAS-develop/include)

# OpenMP
FIND_PACKAGE(OpenMP)
IF (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()


# copy library
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Create shared library
ADD_LIBRARY(calcpigpu SHARED ${calcpigpu_SOURCES})

# Run tests
ENABLE_TESTING()
add_test(NAME n2_test
  COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_BINARY_DIR}/../src/CsfCIPSI/test/n2_test.py
    )
add_test(NAME n2_cmaketest
  COMMAND ${PYTHON_EXECUTABLE}
  ${CMAKE_CURRENT_BINARY_DIR}/../src/CsfCI/test/n2_cmaketest.py
    )
