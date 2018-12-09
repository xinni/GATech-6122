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
};

void cleanup(DeviceData *d ) {
    cudaFree(d->d_fix);
    cudaFree(d->d_cur);
    cudaFree(d->d_pre);
    cudaFree(d->in);
    cudaFree(d->out);
}


void setInitHeatMap(float *dst, int width, int height, int location_x, int location_y, int widthFix, int heightFix, float fixedTemp) {
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


void setInitHeatMap(float *dst, int width, int height, int depth, int location_x, int location_y, int location_z, int widthFix, int heightFix, int depthFix, float fixedTemp) {
    for (int k = 0; k < depth; k++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if ((k >= location_z) && (k < location_z + depthFix)) {
                    if ((i >= location_y) && (i < location_y + heightFix)) {
                        if ((j >= location_x) && (j < location_x + widthFix)){
                            dst[k * width * height + i * width + j] = fixedTemp;
                        }
                    }
                }
            }
        }
    }
}


__global__ void copy_const_kernel (float *dst, const float *src, int width, int height, int depth) {
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    int index = x + y * width + z * width * height;

    if ((x < width && y < height && z < depth) && (src[index] != 0)) dst[index] = src[index];
}


__global__ void update_kernel (float *dst, float *src, int width, int height, int depth, float k) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    int index = x + y * width + z * width * height;

    int left = index - 1;  
    int right = index + 1;  
    if (x == 0) left++;  
    if (x == width - 1) right--;
  
    int top = index - width;
    int bottom = index + width;
    if (y == 0) top += width;
    if (y == height - 1) bottom -= width;

    int front = index - width * height;
    int back = index + width * height;
    if (z == 0) front = front + width * height;
    if (z == depth - 1) back = back - width * height;
  
    if (x < width && y < height && z < depth) {
        dst[index] = src[index] + k * 
            (src[top] + src[bottom] + src[left] + src[right] + src[front] + src[back] - src[index] * 6);
    }
}

int main (int argc, char *argv[]) {
    DeviceData data;

    
    bool dim2D = true; 
    float k, startTemp, *fix;
    int timeSteps, N, width, height, depth = 1;
    
    
    ifstream infile(argv[1]);
    string line;
    int index = 0;
    while (getline(infile, line)) {

        
        int found= line.find_first_not_of(" \t");
        if(found != string::npos) {
            if(line[found] == '#') continue;
        } else {
            continue;
        }
        istringstream iss(line);
        char comma;
        
        
        if (index == 0) {
            
            if (line == "3D") dim2D = false; 

        } else if (index == 1) {
            
            if (!(iss >> k)) break; 

        } else if (index == 2) {
            
            if (!(iss >> timeSteps)) break;

        } else if (index == 3) {
            
            if (dim2D) {
                if (!(iss >> width >> comma >> height) || (comma != ',')) break;
                
            } else {
                if (!(iss >> width >> comma >> height >> comma >> depth) || (comma != ',')) break;
            }
            N = width * height * depth;
            fix = new float[N];
            fill_n(fix, N, 0);

        } else if (index == 4) {
            
            if (!(iss >> startTemp)) break;

        } else {
            
            if (dim2D) {
                int _x, _y, wf, hf;
                float tf;
                if (!(iss >> _x >> comma >> _y >> comma >> wf >> comma >> hf >> 
                    comma >> tf) || (comma != ',')) break;
                setInitHeatMap(fix, width, height, _x, _y, wf, hf, tf);
            } else {
                int _x, _y, _z, wf, hf, df;
                float tf;
                if (!(iss >> _x >> comma >> _y >> comma >> _z >> comma >> wf >> comma >> hf >> 
                    comma >> df >> comma >> tf) || (comma != ',')) break;
                setInitHeatMap(fix, width, height, depth, _x, _y, _z, wf, hf, df, tf);
            }
        }
        index++;
    }
    

    float previous[N];
    float current[N] = {0};
    fill_n(previous, N, startTemp);

    cudaMalloc((void**)&data.d_cur, N * sizeof(float));
    cudaMalloc((void**)&data.d_pre, N * sizeof(float));
    cudaMalloc((void**)&data.d_fix, N * sizeof(float));

    dim3 blocks((width + DIM - 1) / DIM, (height + DIM - 1) / DIM, (depth + DIM - 1) / DIM);
    dim3 threads(DIM, DIM, DIM);

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
        
        update_kernel<<<blocks, threads>>>(data.out, data.in, width, height, depth, k);
        copy_const_kernel<<<blocks, threads>>>(data.out, data.d_fix, width, height, depth);
    }

    cudaMemcpy(current, data.out, N*sizeof(int), cudaMemcpyDeviceToHost);

    
	ofstream outFile;
	outFile.open("heatOutput.csv", ios::out);
    for (int i = 0; i < N; i++) {
        if (i % (width * height) != width * height - 1) {
            if (i % width != width - 1) {
                outFile << current[i] << ", ";
            } else {
                outFile << current[i] << endl;
            }
        } else {
            if (i == N - 1) {
                outFile << current[i] << endl;
            } else {
                outFile << current[i] << endl << endl;
            }
        }
    }
    
    cleanup(&data);

    return 0;
}