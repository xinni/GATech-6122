#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <string>
#include <sstream>
#include <fstream>

#define threadNum 8

using namespace std;

int head[threadNum];
int tail[threadNum];
float *previous;
float *current;
float *fix;

struct DeviceData {
    float *d_fix;
    float *d_cur;
    float *d_pre;
    float *in;
    float *out;
};

/***************** Set fix value to fix matrix *********************/
void setInitHeatMap(float *dst, int width, int height, int location_x, int location_y, int widthFix, int heightFix, int fixedTemp) {
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

// Overload for 3D
void setInitHeatMap(float *dst, int width, int height, int depth, int location_x, int location_y, int location_z, int widthFix, int heightFix, int depthFix, int fixedTemp) {
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

/*************** copy src mat to dst mat **********************/
void copy_const_kernel (int tid) {
    for (int i = head[tid]; i <= tail[tid]; i++) {
        if (fix[i] != 0) previous[i] = fix[i];
    }
}

/****************** temp update function **********************/
void update_kernel (int width, int height, int depth, float k, int tid) {
    for (int index = head[tid]; index <= tail[tid]; index++) {
        int left = index - 1;
        int right = index + 1;
        if (index % width == 0) left++;
        if (index % width == width - 1) right--;

        int top = index - width;
        int bottom = index + width;
        if (index % (width * height) < width) top += width;
        if ((width * height) - index % (width * height) < width) bottom -= width;
    
        // int front = index - width * height;
        // int back = index + width * height;
        // if (z == 0) front = front + width * height;
        // if (z == depth - 1) back = back - width * height;
        
        current[index] = previous[index] + k * 
            (previous[top] + previous[bottom] + previous[left] + previous[right] - previous[index] * 4);
    
    }
}


int main(int argc, const char *argv[])
{
    DeviceData data;

    // set all value
    bool dim2D = true; // true as 2D, false as 3D, default as 2D
    float k, startTemp;
    int timeSteps, N, width, height, depth = 1;
    
    /*********************** read the config file *************************/
    ifstream infile(argv[1]);
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
                
            } else {
                if (!(iss >> width >> comma >> height >> comma >> depth) || (comma != ',')) break;
            }
            N = width * height * depth;
            fix = new float[N];
            fill_n(fix, N, 0);

        } else if (index == 4) {
            // read start temp
            if (!(iss >> startTemp)) break;

        } else {
            // set fix mat
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
    /*****************************************************************/

    // Define number of points (eachProcs) each procs should calculate
	int eachThread = N / threadNum;
	int lastThread = eachThread + (N % threadNum);

	// Define two array to store the head and tail info for each rank
	for (int i = 0; i < threadNum; i++) {
		head[i] = i * eachThread;
		if (i != threadNum - 1) {
			tail[i] = head[i] + eachThread - 1;
		} else {
			tail[i] = head[i] + lastThread - 1;
		}
	}

    previous = new float[N];
    current = new float[N];
    fill_n(current, N, 0);
    fill_n(previous, N, startTemp);

    thread t[threadNum - 1];
    for (int i = 0; i < threadNum - 1; i++) {
        t[i] = thread(copy_const_kernel, i);
    }

    copy_const_kernel(threadNum - 1);

    for (int i = 0; i < threadNum - 1; i++) {
        t[i].join();
    }

    for (int i = 0; i < threadNum - 1; i++) {
        t[i] = thread(update_kernel, width, height, depth, k, i);
    }

    update_kernel(width, height, depth, k, threadNum - 1);

    for (int i = 0; i < threadNum - 1; i++) {
        t[i].join();
    }

    cout << width << height << depth << k << threadNum << endl;

    for (int i = 0; i < N; i++) {
        if (i % (width * height) != width * height - 1) {
            if (i % width != width - 1) {
                cout << current[i] << ", ";
            } else {
                cout << current[i] << endl;
            }
        } else {
            if (i == N - 1) {
                cout << current[i] << endl;
            } else {
                cout << current[i] << endl << endl;
            }
        }
    }

    return EXIT_SUCCESS;
}