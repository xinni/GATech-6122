#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include "complex.h"
#include "input_image.h"

#define DIM 32

using namespace std;

/*************** transform by row **********************/
// __global__ void transByRow (complex *dst, complex *src, int width, int height) {
//     int x = threadIdx.x + blockIdx.x * blockDim.x;
//     int y = threadIdx.y + blockIdx.y * blockDim.y;
//     int index = x + y * width;

//     if (x < width && y < height) {
//         for (int i = 0; i < width; i++) {
//             dst[index]->real += (src[i]->real * cos((2*PI*index*i) / width))/width;
//             dst[index]->imag += -(src[i]->real * sin((2*PI*index*i) / width))/width;
//         }
//     }
// }

int main (int argc, char* argv[]) {
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

    Complex src[N];
    Complex res[N];
    
    Complex* data = image.get_image_data();

    return 0;
}
