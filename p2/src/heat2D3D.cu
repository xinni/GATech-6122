#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <sstream>
#include <fstream>

using namespace std;

struct DeviceData {
    float d_k;
    int d_timeSteps;
    int d_width;
    int d_height;
    int d_depth;
    float d_startTemp;
    float *d_fix;
    float *d_cur;
    float *d_pre;
    bool d_dim2D;
};

void cleanup(DeviceData *d ) {
    // cudaFree(d->d_k);
    // cudaFree(d->d_timeSteps);
    // cudaFree(d->d_width);
    // cudaFree(d->d_height);
    // cudaFree(d->d_depth);
    // cudaFree(d->d_startTemp);
    cudaFree(d->d_fix);
    cudaFree(d->d_cur);
    cudaFree(d->d_pre);
    // cudaFree(d->d_dim2D);
}

void setInitHeatMap2D(float *dst, int width, int height, int location_x, int location_y, int widthFix, int heightFix, int fixedTemp) {
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

/*************** copy src mat to dst mat **********************/
__global__ void copy_const_kernel (float *dst, const float *src) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;

    if (src[index] != 0) dst[index] = src[index];
}

/*************** 2D temp update function **********************/
__global__ void update_2D_kernel (float *dst, float *src, int width, float k) {
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
    DeviceData data;

    // set all value
    bool dim2D = true; // true as 2D, false as 3D, default as 2D
    float k, startTemp, *fix;
    int timeSteps, N, width, height, depth = 0;
    
    /*********************** read the config file *************************/
    ifstream infile("sample.conf");
    string line;
    int index = 0;
    while (getline(infile, line)) {

        // read non# and empty line set index as line number.
        int found= line.find_first_not_of(" \t");
        if(found != string::npos) {
            if(line[found] == '#') continue;
        } else {
            continue;
        }
        istringstream iss(line);
        char comma;
        
        // set every line as param
        if (index == 0) {
            // read first line 2D as true, 3D as false
            if (line == "3D") dim2D = false; 

        } else if (index == 1) {
            // read k value, if error break
            if (!(iss >> k)) break; 

        } else if (index == 2) {
            // read timesteps value
            if (!(iss >> timeSteps)) break;

        } else if (index == 3) {
            // read dim, if 2D read width and height , 3D read w, h ,d
            if (dim2D) {
                if (!(iss >> width >> comma >> height) || (comma != ',')) break;
                N = width * height;
                fix = new float[N];
            } else {
                if (!(iss >> width >> comma >> height >> comma >> depth) || (comma != ',')) break;
                N = width * height * depth;
                fix = new float[N];
            }

        } else if (index == 4) {
            // read start temp
            if (!(iss >> startTemp)) break;

        } else {
            // set fix mat
            if (dim2D) {
                int _x, _y, wf, hf;
                float tf;
                if (!(iss >> _x >> comma >> _y >> comma >> wf >> comma >> hf >> comma >> tf) || (comma != ',')) break;
                setInitHeatMap2D(fix, width, height, _x, _y, wf, hf, tf);
            } else {

                // TODO: finish 3D temp distrabute
            }
        }
        index++;
    }
    /*****************************************************************/

    // TODO: new fix, current, previous

    cudaMalloc((void**)&data.d_cur, N * sizeof(float));
    cudaMalloc((void**)&data.d_pre, N * sizeof(float));
    cudaMalloc((void**)&data.d_fix, N * sizeof(float));


    // cout << dim2D << endl;
    cout << k << endl;
    cout << timeSteps << endl;
    // cout << "w = " << width << " h = " << height << " d = " << depth << endl;
    cout << startTemp << endl;

    for (int i = 0; i < height * width; i++) {
        if (i % width != width - 1) {
            cout << fix[i] << ", ";
        } else {
            cout << fix[i] << endl;
        }
    }

    cleanup(&data);

    return 1;
}