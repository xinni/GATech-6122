#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cmath>
#include <fstream>
#include <sstream>

#define DIM 32

const float PI = 3.14159265358979f;

using namespace std;

/*************************************************/
class Complex {
public:
    __host__ __device__ Complex() : real(0.0f), imag(0.0f) {}
    
    __host__ __device__ Complex(float r) : real(r), imag(0.0f) {}
    
    __host__ __device__ Complex(float r, float i) : real(r), imag(i) {}
    
    __host__ __device__ Complex operator+(const Complex &b) const {
        float newReal = real + b.real;
        float newImag = imag + b.imag;
        Complex newComplex(newReal, newImag);
        return newComplex;
    }
    
    __host__ __device__ Complex operator-(const Complex &b) const {
        float newReal = real - b.real;
        float newImag = imag - b.imag;
        Complex newComplex(newReal, newImag);
        return newComplex;
    }
    
    __host__ __device__ Complex operator*(const Complex &b) const {
        float newReal = real * b.real - imag * b.imag;
        float newImag = real * b.imag + imag * b.real;
        Complex newComplex(newReal, newImag);
        return newComplex;
    }
    
    __host__ __device__ Complex mag() const {
        float magNum = sqrt(real * real + imag * imag);
        Complex magComplex(magNum);
        return magComplex;
    }
    
    __host__ __device__ Complex angle() const {
        float angle = atan(1.0 * imag / real)*180/PI;
        Complex angleComplex(angle);
        return angleComplex;
    }
    
    __host__ __device__ Complex conj() const {
        Complex newComplex(real, -1.0 * imag);
        return newComplex;
    }

    float real;
    float imag;
};

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}

class InputImage {
public:

    InputImage(const char* filename){
        std::ifstream ifs(filename);
        if(!ifs) {
            std::cout << "Can't open image file " << filename << std::endl;
            exit(1);
        }
    
        ifs >> w >> h;
        data = new Complex[w * h];
        for(int r = 0; r < h; ++r) {
            for(int c = 0; c < w; ++c) {
                float real;
                ifs >> real;
                data[r * w + c] = Complex(real);
            }
        }
    }
    int get_width() const{
        return w;
    }
    int get_height() const{
        return h;
    }

    //returns a pointer to the image data.  Note the return is a 1D
    //array which represents a 2D image.  The data for row 1 is
    //immediately following the data for row 0 in the 1D array
    Complex* get_image_data() const{
        return data;
    }
    //use this to save output from forward DFT
    void save_image_data(const char* filename, Complex* d, int w, int h){
        std::ofstream ofs(filename);
        if(!ofs) {
            std::cout << "Can't create output image " << filename << std::endl;
            return;
        }
    
        ofs << w << " " << h << std::endl;
    
        for(int r = 0; r < h; ++r) {
            for(int c = 0; c < w; ++c) {
                ofs << d[r * w + c] << " ";
            }
            ofs << std::endl;
        }
    }
    //use this to save output from reverse DFT
    void save_image_data_real(const char* filename, Complex* d, int w, int h){
        std::ofstream ofs(filename);
        if(!ofs) {
            std::cout << "Can't create output image " << filename << std::endl;
            return;
        }
    
        ofs << w << " " << h << std::endl;
    
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                ofs << d[r * w + c].real << " ";
            }
            ofs << std::endl;
        }
    }

private:
    int w;
    int h;
    Complex* data;
};
/*************************************************/

struct DeviceData {
    Complex *d_data;
    Complex *d_temp;
    Complex *d_res;
};

void cleanup(DeviceData *d) {
    cudaFree(d->d_data);
    cudaFree(d->d_temp);
    cudaFree(d->d_res);
}

/*************** transform by row **********************/
__global__ void transByRow (Complex* dst, Complex* src, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * width;

    if (x < width && y < height) {
        for (int i = 0; i < width; i++) {
            float re = (src + y*width + i)->real;
            float im = (src + y*width + i)->imag;
            // (dst + index)->real += re * cos((2*PI*x*i)/width) + im*sin(2*PI*i*x/width);
            // (dst + index)->imag += -re * sin((2*PI*x*i)/width) + im*cos(2*PI*i*x/width);
            Complex w = Complex(cos(2*PI*i*x/width), -sin(2*PI*i*x/width));
            (dst + index)->real += re * w.real - im*w.imag;
            (dst + index)->imag += re * w.imag + im*w.real;
        }
    }
}

/*************** transform by column **********************/
__global__ void transByCol (Complex* dst, Complex* src, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * width;

    if (x < width && y < height) {
        for (int i = 0; i < height; i++) {
            float re = (src + x + i*width)->real;
            float im = (src + x + i*width)->imag;
            Complex w = Complex(cos(2*PI*i*y/height), -sin(2*PI*i*y/height));
            (dst + index)->real += re * w.real - im*w.imag;
            (dst + index)->imag += re * w.imag + im*w.real;
        }
    }
}

int main (int argc, char* argv[]) {
    DeviceData devs;

    std::string str = "test.txt";
    const char* inFile = str.c_str();
    str = "res.txt";
    const char* outFile = str.c_str();

    if (argc > 1) {
        inFile = argv[1];
        outFile = argv[2];
    }

    InputImage image(inFile);
    int height = image.get_height();
    int width = image.get_width();
    int N = height * width;

    Complex res[N];
    fill_n(res, N, 1);

    Complex* data = image.get_image_data();

    cudaMalloc((void**)&devs.d_data, N * sizeof(Complex));
    cudaMalloc((void**)&devs.d_res, N * sizeof(Complex));
    cudaMalloc((void**)&devs.d_temp, N * sizeof(Complex));

    cudaMemcpy(devs.d_data, data, N * sizeof(Complex), cudaMemcpyHostToDevice);

    dim3 blocks((width + DIM - 1) / DIM, (height + DIM - 1) / DIM);
    dim3 threads(DIM, DIM);

    cout << width << ", " << height << endl;

    transByRow<<<blocks, threads>>>(devs.d_temp, devs.d_data, width, height);
    transByCol<<<blocks, threads>>>(devs.d_res, devs.d_temp, width, height);

    cudaMemcpy(res, devs.d_res, N*sizeof(Complex), cudaMemcpyDeviceToHost);
    image.save_image_data(outFile, res, width, height);

    cleanup(&devs);

    return 0;
}
