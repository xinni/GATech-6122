#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "complex.h"
#include "input_image.h"

#define DIM 32

const float PI = 3.14159265358979f;

using namespace std;

struct DeviceData {
    Complex *d_data;
    Complex *d_res;
};

void cleanup(DeviceData *d) {
    cudaFree(d->d_data);
    cudaFree(d->d_res);
}

/*************** transform by row **********************/
__global__ void transByRow (Complex* dst, Complex* src, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * width;

    if (x < width && y < height) {
        dst[index] = 3;
        // for (int i = 0; i < width; i++) {
        //     // (dst + index)->real += ((src + i)->real * cos((2*PI*index*i) / width))/width;
        //     // (dst + index)->imag += -((src + i)->real * sin((2*PI*index*i) / width))/width;
        //     dst[index] = 3;
        // }
    }
}

int main (int argc, char* argv[]) {
    DeviceData devs;

    char* inFile="Tower256.txt";
    char* outFile="Mytest.txt";

    if (argc > 1) {
        inFile = argv[1];
        outFile = argv[2];
    }

    InputImage image(inFile);
    int height = image.get_height();
    int width = image.get_width();
    int N = height * width;

    // Complex src[N];
    Complex res[N];
    fill_n(res, N, 1);

    Complex* data = image.get_image_data();

    cudaMalloc((void**)&devs.d_data, N * sizeof(Complex));
    cudaMalloc((void**)&devs.d_res, N * sizeof(Complex));

    cudaMemcpy(devs.d_data, data, N * sizeof(Complex), cudaMemcpyHostToDevice);

    dim3 blocks((width + DIM - 1) / DIM, (height + DIM - 1) / DIM);
    dim3 threads(DIM, DIM);

    cout << width << ", " << height << endl;

    transByRow<<<blocks, threads>>>(devs.d_res, devs.d_data, width, height);

    cudaMemcpy(res, devs.d_res, N*sizeof(Complex), cudaMemcpyDeviceToHost);

    image.save_image_data("MyAfter2D.txt", res, width, height);

    cleanup(&devs);

    return 0;
}
