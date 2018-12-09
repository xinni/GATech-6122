cmake_minimum_required(VERSION 3.9)
project(FinalProject)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)

file( GLOB  cu  *.cu)
cuda_add_executable(
    fft2d
    src/fft2d.cu

)
