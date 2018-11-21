#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>

using namespace std;
//#define k 0.5f      // only can use define?
#define width 10
#define height 10

__device__ float k = 0.5;
int timeSteps = 10;
// int width = 10;
// int height = 10;
int N = width * height;
float startingTemp = 0;
int location_x, location_y, location_z = 0;
float fixedTemp = 5;
int DIM = 16;

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
  
    dst[index] = src[index] + k * (src[top] +  
                     src[bottom] + src[left] + src[right] -  
                     src[index]*4);
}

int main (void) {
    float current[width][height] = {0};
    float previous[width][height] = {0};

    float *d_c, *d_p;

    cudaMalloc((void**)&d_c, N * sizeof(float));
    cudaMalloc((void**)&d_p, N * sizeof(float));

    dim3 blocks(DIM/16,DIM/16);
    dim3 threads(16,16);
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << current[i][j];
            if (j == width - 1) {
                cout << endl;
            } else {
                cout << ", ";
            }
        }
    }
    // cout << "k = " << k << endl;

    return 1;
}