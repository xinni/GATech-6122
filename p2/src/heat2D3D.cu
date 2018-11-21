#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>

using namespace std;
//#define k 0.5f      // only can use define?

__device__ float k = 0.5;
int timeSteps = 10;
int width = 200;
int height = 200;
float startingTemp = 0;
int location_x, location_y, location_z = 0;
float fixedTemp = 5;
int DIM = 16;

__global__ void update_2D_kernel (float *dst) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;

    float up, down, left, right, current;

    up = 0;
    down = 0;
    left = 0;
    right = 0;
    current = 0;

    dst[index] = current + k * (up + down + left + right - 4 * current);
}

int main (void) {
    float map[width][height] = {0};

    dim3 blocks(DIM/16,DIM/16);
    dim3 threads(16,16);
    
    cout << "k = " << k << endl;

    return 1;
}