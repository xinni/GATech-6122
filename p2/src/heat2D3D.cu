#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <sstream>
#include <fstream>

#define DIM 8

using namespace std;

struct DeviceData {
    float *d_fix;
    float *d_cur;
    float *d_pre;
    float *in;
    float *out;
    // bool d_dim2D;
};

void cleanup(DeviceData *d ) {
    cudaFree(d->d_fix);
    cudaFree(d->d_cur);
    cudaFree(d->d_pre);
    cudaFree(d->in);
    cudaFree(d->out);
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
__global__ void copy_const_kernel (float *dst, const float *src, int N) {
    // x as width, y as height
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;

    if ((index < N) && (src[index] != 0)) dst[index] = src[index];
}

/*************** 2D temp update function **********************/
__global__ void update_2D_kernel (float *dst, float *src, int width, int height, float k, int N) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;

    int left = index - 1;  
    int right = index + 1;  
    if (x == 0) left++;  
    if (x == width - 1) right--;
  
    int top = index - width;
    int bottom = index + width;
    if (y == 0) top += width;
    if (y == height - 1) bottom -= width;
  
    if (index < N) {
        dst[index] = src[index] + k * 
            (src[top] + src[bottom] + src[left] + src[right] - src[index] * 4);
    }
}

int main (void) {
    DeviceData data;

    // set all value
    bool dim2D = true; // true as 2D, false as 3D, default as 2D
    float k, startTemp, *fix;
    int timeSteps, N, width, height, depth = 1;
    
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

    float previous[N];
    float current[N] = {0};
    fill_n(previous, N, startTemp);

    cudaMalloc((void**)&data.d_cur, N * sizeof(float));
    cudaMalloc((void**)&data.d_pre, N * sizeof(float));
    cudaMalloc((void**)&data.d_fix, N * sizeof(float));

    //dim3 blocks((width + DIM - 1) / DIM, (height + DIM - 1) / DIM, (depth + DIM - 1) / DIM);
    dim3 blocks(1, 1, 1);
    dim3 threads(4, 5, DIM);

    cudaMemcpy(data.d_fix, fix, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data.d_pre, previous, N*sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i <= timeSteps; i++) {
        if (i % 2) {
            data.in = data.d_cur;
            data.out = data.d_pre;
        } else {
            data.in = data.d_pre;
            data.out = data.d_cur;
        }
        
        update_2D_kernel<<<blocks, threads>>>(data.out, data.in, width, height, k, N);
        copy_const_kernel<<<blocks, threads>>>(data.out, data.d_fix, N);
    }

    cudaMemcpy(current, data.out, N*sizeof(int), cudaMemcpyDeviceToHost);

    // cout << dim2D << endl;
    cout << k << endl;
    cout << timeSteps << endl;
    // cout << "w = " << width << " h = " << height << " d = " << depth << endl;
    cout << startTemp << endl;

    for (int i = 0; i < N; i++) {
        if (i % width != width - 1) {
            cout << current[i] << ", ";
        } else {
            cout << current[i] << endl;
        }
    }

    cleanup(&data);

    return 1;
}