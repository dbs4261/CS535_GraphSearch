cmake_minimum_required(VERSION 3.10.2)
project(CS535_GraphSearch C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CUDA_COMPILED_ARCH "sm_50")

find_package(CUDA REQUIRED)
message("Found cuda libraries: ${CUDA_LIBRARIES}")
message("Found cuda includes: ${CUDA_INCLUDE_DIRS}")

set(OpenACC_ACCEL_TARGET=CXX)
find_package(OpenACC)
if (OpenACC_CXX_FOUND)
  message("Found OpenACC flags: ${OpenACC_CXX_FLAGS}")
else()
  message("Did not find openacc...")
endif()

add_subdirectory(source)
add_subdirectory(applications)