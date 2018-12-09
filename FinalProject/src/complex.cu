//
// Created by brian on 11/20/18.
//

#include "complex.h"
#include <cuda.h>

#include <cmath>

const float PI = 3.14159265358979f;

__host__ __device__ Complex::Complex() : real(0.0f), imag(0.0f) {}

__host__ __device__ Complex::Complex(float r) : real(r), imag(0.0f) {}

__host__ __device__ Complex::Complex(float r, float i) : real(r), imag(i) {}

__host__ __device__ Complex Complex::operator+(const Complex &b) const {
    float newReal = real + b.real;
    float newImag = imag + b.imag;
    Complex newComplex(newReal, newImag);
    return newComplex;
}

__host__ __device__ Complex Complex::operator-(const Complex &b) const {
    float newReal = real - b.real;
    float newImag = imag - b.imag;
    Complex newComplex(newReal, newImag);
    return newComplex;
}

__host__ __device__ Complex Complex::operator*(const Complex &b) const {
    float newReal = real * b.real - imag * b.imag;
    float newImag = real * b.imag + imag * b.real;
    Complex newComplex(newReal, newImag);
    return newComplex;
}

__host__ __device__ Complex Complex::mag() const {
    float magNum = sqrt(real * real + imag * imag);
    Complex magComplex(magNum);
    return magComplex;
}

__host__ __device__ Complex Complex::angle() const {
    float angle = atan(1.0 * imag / real)*180/PI;
    Complex angleComplex(angle);
    return angleComplex;
}

__host__ __device__ Complex Complex::conj() const {
    Complex newComplex(real, -1.0 * imag);
    return newComplex;
}

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