cmake_minimum_required(VERSION 3.9)
project(p2)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGES "${CMAKE_CXX_FLAGES}" -std=c++11)

set(CUDA_NVCC_FLAGES ${CUDA_NVCC_FLAGES};
	-03 -gencode arch=compute_22, code=sm_22
	)

include_directories(
	src
)

cuda_add_executable(
	fft2d
    src/fft2d.cu
    src/complex.h
    src/complex.cc
    src/input_image.cc
    src/input_image.h
)
