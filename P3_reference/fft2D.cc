// Distributed two-dimensional Discrete FFT transform
// Zeyu Chen

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <chrono>
#include <future>
#include <thread>

#include "Complex.h"
#include "InputImage.h"

#define TEST 0

constexpr unsigned int NUMTHREADS = 4;

using namespace std;

//undergrad students can assume NUMTHREADS will evenly divide the number of rows in tested images
//graduate students should assume NUMTHREADS will not always evenly divide the number of rows in tested images.
// I will test with a different image than the one given
//////////////////////////////////////////////////Declare///////////////////////////////////////////////////////////////
void Transform2D(const char* inputFN);
void ITransform2D(InputImage* image, Complex* data);
void Transform1D(Complex* h, int w, Complex* H);
void ITransform1D(Complex* h, int w, Complex* H);
void Transpose(Complex* m_t, Complex*m, int m_width, int m_height);  // Transpose the metrics
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Transform2D(const char* inputFN) {
    // Do the 2D transform here.
    // 1) Use the InputImage object to read in the Tower.txt file and
    //    find the width/height of the input image.
    // 2) Create a vector of complex objects of size width * height to hold
    //    values calculated
    // 3) Do the individual 1D transforms on the rows assigned to each thread
    // 4) Force each thread to wait until all threads have completed their row calculations
    //    prior to starting column calculations
    // 5) Perform column calculations
    // 6) Wait for all column calculations to complete
    // 7) Use SaveImageData() to output the final results
    InputImage image(inputFN);  // Create the helper object for reading the image
    // Your code here, steps 2-7

    Complex* data;
    Complex* res;
    int height, width, load;
    std::future<void> foo[NUMTHREADS];

    height = image.GetHeight();
    width = image.GetWidth();
    data = image.GetImageData();  // height*width

    res = new Complex[height*width];

    // First by row
    // Assign work to each thread
    load = height/NUMTHREADS;
    for (int i = 0;i < NUMTHREADS - 1; ++i) {
        foo[i] = std::async(std::launch::async, [i, load, height, width, &data, &res] {
            for (int j = 0; j < load; ++j)
                Transform1D(data + width*j + load*width*i, width, res+ width*j + load*width*i);
        });
    }
    foo[NUMTHREADS - 1] = std::async(std::launch::async, [load, height, width, &data, &res] {
        int lload = load + height % NUMTHREADS;
        for (int j = 0; j < lload; ++j)
            Transform1D(data + width*j + width*load*(NUMTHREADS - 1), width, res+ width*j + width*load*(NUMTHREADS - 1));
    });

     for (int i = 0; i < NUMTHREADS - 1; ++i)
         foo[i].get();

    //image.SaveImageData("MyAfter1D.txt", res, width, height);

    // Then by column
    // Do transpose
    Complex* res_t = new Complex[width*height];
    Transpose(res_t, res, width, height);

    delete[] res;
    res = NULL;
    res = new Complex[height*width];

    load = width/NUMTHREADS;
    for (int i = 0;i < NUMTHREADS - 1; ++i) {
        foo[i] =std::async(std::launch::async, [i, load, height, width, &res_t, &res] {
            for (int j = 0; j < load; ++j)
                Transform1D(res_t + height*j + load*height*i, height, res + height*j + load*height*i);
        });
    }
    foo[NUMTHREADS - 1] = std::async(std::launch::async, [load, height, width, &res_t, &res] {
        int lload = load + height % NUMTHREADS;
        for (int j = 0; j < lload; ++j)
            Transform1D(res_t + height*j + height*load*(NUMTHREADS - 1), height, res + height*j + height*load*(NUMTHREADS - 1));
    });

    for (int i = 0; i < NUMTHREADS; ++i)
        foo[i].get();

    Transpose(res_t, res, height, width);

    delete[] res;
    res = res_t;

    image.SaveImageData("MyAfter2D.txt", res, width, height);
    ITransform2D(&image, res);

    delete[] res;
    res = res_t = NULL;
#if TEST
    int num = 10;
    Complex* H = new Complex[num];
    std::cout << "Original:";
    for (int i = 0; i < num; ++i) {
        std::cout << *(data + i) << ' ';
    }
    std::cout << std::endl;

    std::cout << "DFT:";
    Transform1D(data, num, H);
    for (int i = 0; i < num; ++i) {
        std::cout << *(H + i) << ' ';
    }
    std::cout << std::endl;

    Complex* h = new Complex[num];
    ITransform1D(H, num, h);
    std::cout << "IDFT:";
    for (int i = 0; i < num; ++i) {
        std::cout << *(h + i) << ' ';
    }
    std::cout << std::endl;
#endif

}

void ITransform2D(InputImage* image, Complex* data) {
    Complex* res;
    int height, width, load;
    std::future<void> foo[NUMTHREADS];

    height = image->GetHeight();
    width = image->GetWidth();

    res = new Complex[height*width];

    // First by row
    // Assign work to each thread
    load = height/NUMTHREADS;
    for (int i = 0;i < NUMTHREADS - 1; ++i) {
        foo[i] = std::async(std::launch::async, [i, load, height, width, &data, &res] {
            for (int j = 0; j < load; ++j)
                ITransform1D(data + width*j + load*width*i, width, res + width*j + load*width*i);
        });
    }
    foo[NUMTHREADS - 1] = std::async(std::launch::async, [load, height, width, &data, &res] {
        int lload = load + height % NUMTHREADS;
        for (int j = 0; j < lload; ++j)
            ITransform1D(data + width*j + width*load*(NUMTHREADS - 1), width, res + width*j + width*load*(NUMTHREADS - 1));
    });

    for (int i = 0; i < NUMTHREADS - 1; ++i)
        foo[i].get();

    // image->SaveImageDataReal("IMyAfter1D.txt", res, width, height);

    // Then by column
    // Do transpose
    Complex* res_t = new Complex[width*height];
    Transpose(res_t, res, width, height);

    delete[] res;
    res = NULL;
    res = new Complex[height*width];

    load = width/NUMTHREADS;
    for (int i = 0;i < NUMTHREADS - 1; ++i) {
        foo[i] =std::async(std::launch::async, [i, load, height, width, &res_t, &res] {
            for (int j = 0; j < load; ++j)
                ITransform1D(res_t + height*j + load*height*i, height, res + height*j + load*height*i);
        });
    }
    foo[NUMTHREADS - 1] = std::async(std::launch::async, [load, height, width, &res_t, &res] {
        int lload = load + height % NUMTHREADS;
        for (int j = 0; j < lload; ++j)
            ITransform1D(res_t + height*j + height*load*(NUMTHREADS - 1), height, res + height*j + height*load*(NUMTHREADS - 1));
    });

    for (int i = 0; i < NUMTHREADS; ++i)
        foo[i].get();

    Transpose(res_t, res, height, width);

    delete[] res;
    res = res_t;

    image->SaveImageDataReal("MyAfterInverse.txt", res, width, height);

    delete[] res;
    res = res_t = NULL;

}

void Transform1D(Complex* h, int w, Complex* H) {
    // Implement a simple 1-d DFT using the double summation equation
    // given in the assignment handout.  h is the time-domain input
    // data, w is the width (N), and H is the output array.

    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < w; ++j) {
            double real = (h + j)->real;
            double imag = (h + j)->imag;
            (H + i)->real += real*cos(2*M_PI*i*j/w) + imag*sin(2*M_PI*i*j/w);
            (H + i)->imag += -real*sin(2*M_PI*i*j/w) + imag*cos(2*M_PI*i*j/w);
        }
    }
}

void ITransform1D(Complex* h, int w, Complex* H) {
    // Implement a simple 1-d IDFT using the double summation equation
    // given in the assignment handout.  h is the time-domain input
    // data, w is the width (N), and H is the output array.
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < w; ++j) {
            double real = (h + j)->real;
            double imag = (h + j)->imag;
            (H + i)->real += (real*cos(2*M_PI*i*j/w) - imag*sin(2*M_PI*i*j/w))/w;
            (H + i)->imag += (real*sin(2*M_PI*i*j/w) + imag*cos(2*M_PI*i*j/w))/w;
        }
    }
}

void Transpose(Complex* m_t, Complex*m, int m_width, int m_height) {
    for (int i = 0; i < m_width; i++){
        for (int j = 0; j < m_width; j++)
            m_t[j*m_width + i] = m[i*m_width + j];
    }
}

int main(int argc, char** argv) {
    string fn("Tower.txt"); // default file name
    string fn0("Mytest2.txt");
    if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
    Transform2D(fn.c_str()); // Perform the transform.
}  