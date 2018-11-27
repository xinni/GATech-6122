#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <sstream>
#include <fstream>

using namespace std;

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

int main (int argc, char *argv[]) {
    
    // set all value
    bool dim2D = true; // true as 2D, false as 3D, default as 2D
    float k, startTemp, *fix;
    int timeSteps, width, height, depth = 0;
    
    /*****************************************************************/
    // read the config file
    ifstream infile("sample.conf");
    string line;
    int index = 0;
    while (getline(infile, line)) {

        // read non # and empty line set index as line number.
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
                fix = new float[width * height];
            } else {
                if (!(iss >> width >> comma >> height >> comma >> depth) || (comma != ',')) break;
                fix = new float[width * height * depth];
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

    return 1;
}