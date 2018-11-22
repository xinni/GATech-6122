#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>

using namespace std;
#define width 3
#define height 3
__device__ float k = 0.25;

float startingTemp = 0;
// int DIM = 10;

// #location_x, location_y, width, height, fixed temperature 5, 5, 20, 20, 200
// 500, 500, 10, 10, 300
void setInitHeatMap(float *dst, char str[]) {
    // set 5, 5, 2, 2, 10
    int location_x = 1;
    int location_y = 1;
    int widthFix= 1;
    int heightFix = 1;
    float fixedTemp = 5;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if ((i >= location_y) && (i < location_y + heightFix)) {
                if ((j >= location_x) && (j < location_x + widthFix)){
                    dst[i * width + j] = fixedTemp;
                }
            }
        }
    }
}

__global__ void copy_const_kernel (float *dst, const float *src) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;

    if (src[index] != 0) dst[index] = src[index];
}

__global__ void update_2D_kernel (float *dst, float *src) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;

    int left = index - 1;  
    int right = index + 1;  
    if (x == 0) left++;  
    if (x == width-1) right--;
  
    int top = index - width;
    int bottom = index + width;
    if (y == 0) top += width;
    if (y == width - 1) bottom -= width;
  
    dst[index] = src[index] + k * 
            (src[top] + src[bottom] + src[left] + src[right] - src[index] * 4);
}

int main (int argc, char *argv[]) {
    int timeSteps = atoi(argv[1]);
    int N = width * height;

    float fix[width * height] = {0};
    float current[width * height] = {0};
    float previous[width * height] = {0};

    float *d_c, *d_p, *d_i;

    setInitHeatMap(fix, "500, 500, 10, 10, 300");

    cudaMalloc((void**)&d_c, N * sizeof(float));
    cudaMalloc((void**)&d_p, N * sizeof(float));
    cudaMalloc((void**)&d_i, N * sizeof(float));

    dim3 blocks(1, 1);
    dim3 threads(3, 3);

    cudaMemcpy(d_i, fix, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, previous, N*sizeof(float), cudaMemcpyHostToDevice);

    float *in, *out;
    for (int i = 0; i <= timeSteps; i++) {
        if (i % 2) {
            in = d_c;
            out = d_p;
        } else {
            in = d_p;
            out = d_c;
        }
        
        update_2D_kernel<<<blocks, threads>>>(out, in);
        copy_const_kernel<<<blocks, threads>>>(out, d_i);
    }

    cudaMemcpy(current, out, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < height * width; i++) {
        if (i % width != width - 1) {
            cout << current[i] << ", ";
        } else {
            cout << current[i] << endl;
        }
    }

    cudaFree(d_p);
    cudaFree(d_c);
    cudaFree(d_i);

    return 1;
}