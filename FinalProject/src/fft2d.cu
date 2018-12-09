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

int main (int argc, char *argv[]) {
    string inFile("Tower256.txt");
    string outFile("Mytest.txt");

    if (argc > 1) {
        inFile = string(argv[1]);
        outFile = string(argv[2]);
    }

    // input_image image(inFile);
    // int height = inFile.get_height();
    // int width = inFile.get_width();
    // int N = height * width;

    // complex data[N];
    // complex res[N];
    
    // data = inFile.get_image_data();

    return 0;
}
