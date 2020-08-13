/*
 * Author:  Vijay Gopal Chilkuri
 * Email:   vijay.gopal.c@gmail.com
 * Date:    12-08-2020
 */
#ifndef LINEAR_ALGEBRA_HELPERS_H
#define LINEAR_ALGEBRA_HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>
#include <cuda.h>
#include <curand.h>

#define WIDTH 16

/* *****************************
 * Vector Type
 ********************************/
typedef struct {
    int length;
    float* elements;
} Vector;

/* *****************************
 * Vector Functions
 ********************************/
// Vector multiplication kernel ? thread specification
__global__ void VectorMulKernel(Vector M, Vector N, Vector P);

// Allocate a device vector of same size as V.
Vector AllocateDeviceVector(const Vector V);

// Free a device vector.
void FreeDeviceVector(Vector V);

void FreeVector(Vector V);

// Copy one vector to another
void CopyVector(Vector Vout, Vector Vinp);

// Copy a host vector to a device vector.
void CopyToDeviceVector(Vector Vdevice, const Vector Vhost);

// Copy a device vector to a host vector.
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice);

// Vector multiplication on the device
void VectorMulOnDevice(const Vector M, const Vector N, Vector P);

Vector AllocateZeroVector(int length);

Vector AllocateVector(int length);


void PrintVector(float* ma, int X);

// Serialize Vector
void SerializeVector(Vector V, const char *filename);


/* *****************************
 * Matrix Type
 ********************************/
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

/* *****************************
 * Matrix Functions
 ********************************/
// Matrix multiplication kernel ? thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P);

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M);

// Free a device matrix.
void FreeDeviceMatrix(Matrix M);

void FreeMatrix(Matrix M);

// Copy one matrix to another
void CopyMatrix(Matrix Mout, Matrix Minp);

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);

// Matrix multiplication on the device
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);

Matrix AllocateMatrix(int height, int width);


void PrintMatrix(float* ma, int X, int Y);

// Serialize Matrix
void SerializeMatrix(Matrix M, const char *filename);

// Do paralle reduction
int parallel_reduction(void);


#endif // LINEAR_ALGEBRA_HELPERS_H
