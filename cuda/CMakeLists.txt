###################
# Main CMake file #
###################

cmake_minimum_required(VERSION 3.20)

project(LinearElasticity)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(GTest)

# All code is in the src subdirectory
add_subdirectory(src)

install(
  TARGETS lippmannschwinger 
)
