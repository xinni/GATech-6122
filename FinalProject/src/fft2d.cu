#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include "complex.h"
#include "input_image.h"

#define DIM 8

using namespace std;

int main (int argc, char *argv[]) {
    string inFile("Tower256.txt");
    string outFile("Mytest.txt");

    if (argc > 1) {
        inFile = string(argv[1]);
        outFile = string(argv[2]);
    }

    return 0;
}
