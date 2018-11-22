#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>

using namespace std;
#define width 5
#define height 10

__device__ float k = 0.5;
int timeSteps = 10;
int N = width * height;
float startingTemp = 0;
// int DIM = 10;

// #location_x, location_y, width, height, fixed temperature 5, 5, 20, 20, 200
// 500, 500, 10, 10, 300
void setInitHeatMap(float *dst, char str[]) {
    // set 5, 5, 2, 2, 10
    int location_x = 2;
    int location_y = 4;
    int widthFix= 4;
    int heightFix = 2;
    float fixedTemp = 10;

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

int main (void) {
    float fix[width * height] = {0};
    float current[width * height] = {0};
    float previous[width * height] = {0};

    float *d_c, *d_p, *d_i;

    setInitHeatMap(fix, "500, 500, 10, 10, 300");

    cudaMalloc((void**)&d_c, N * sizeof(float));
    cudaMalloc((void**)&d_p, N * sizeof(float));
    cudaMalloc((void**)&d_i, N * sizeof(float));

    dim3 blocks(1,1);
    dim3 threads(5,10);

    cudaMemcpy(d_i, fix, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, previous, N*sizeof(float), cudaMemcpyHostToDevice);
    
    copy_const_kernel<<<blocks, threads>>>(d_p, d_i);
    update_2D_kernel<<<blocks, threads>>>(d_c, d_p);

    cudaMemcpy(current, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

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