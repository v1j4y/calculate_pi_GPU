/*
 * Author:  Vijay Gopal Chilkuri
 * Email:   vijay.gopal.c@gmail.com
 * Date:    12-08-2020
 */

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
__global__ void VectorMulKernel(Vector M, Vector N, Vector P)
{
    // 1D Thread ID
    int tx = threadIdx.x;

    // Pvalue is used to store the element of the vector
    // that is computed by the thread
    float Pvalue = 0;

    float Melement = M.elements[tx];
    float Nelement = N.elements[tx];
    Pvalue += Melement * Nelement;

    // Write the matrix to device memory;
    // each thread writes one element
    P.elements[tx] = Pvalue;
}

// Allocate a device vector of same size as V.
Vector AllocateDeviceVector(const Vector V)
{
    Vector Vdevice = V;
    int size = V.length * sizeof(float);
    cudaMalloc((void**)&Vdevice.elements, size);
    return Vdevice;
}

// Free a device vector.
void FreeDeviceVector(Vector V) {
    cudaFree(V.elements);
}

void FreeVector(Vector V) {
    free(V.elements);
}

// Copy one vector to another
void CopyVector(Vector Vout, Vector Vinp)
{
    int size = Vinp.length * sizeof(float);
    memcpy(Vout.elements, Vinp.elements, size);
}

// Copy a host vector to a device vector.
void CopyToDeviceVector(Vector Vdevice, const Vector Vhost)
{
    int size = Vhost.length * sizeof(float);
    cudaMemcpy(Vdevice.elements, Vhost.elements, size, 
    cudaMemcpyHostToDevice);
}

// Copy a device vector to a host vector.
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice)
{
    int size = Vdevice.length * sizeof(float);
    cudaMemcpy(Vhost.elements, Vdevice.elements, size, 
    cudaMemcpyDeviceToHost);
}

// Vector multiplication on the device
void VectorMulOnDevice(const Vector M, const Vector N, Vector P)
{
    // Load M and N to the device
    Vector Md = AllocateDeviceVector(M);
    CopyToDeviceVector(Md, M);
    Vector Nd = AllocateDeviceVector(N);
    CopyToDeviceVector(Nd, N);

    // Allocate P on the device
    Vector Pd = AllocateDeviceVector(P);
    CopyToDeviceVector(Pd, P); // Clear memory
    
     // Setup the execution configuration
    dim3 dimBlock(WIDTH);
    dim3 dimGrid(1);

    // Launch the device computation threads!
    VectorMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

    // Read P from the device
    CopyFromDeviceVector(P, Pd); 

    // Free device vectors
    FreeDeviceVector(Md);
    FreeDeviceVector(Nd);
    FreeDeviceVector(Pd);
} 

Vector AllocateZeroVector(int length)
{
    Vector M;
    M.length = length;
    int size = M.length;
    M.elements = NULL;

    M.elements = (float*) malloc(size*sizeof(float));

    for(unsigned int i = 0; i < M.length; i++)
    {
        M.elements[i] = 0.0;
    }
    return M;
}

Vector AllocateVector(int length)
{
    Vector M;
    M.length = length;
    int size = M.length;
    M.elements = NULL;

    M.elements = (float*) malloc(size*sizeof(float));

    for(unsigned int i = 0; i < M.length; i++)
    {
        M.elements[i] = (rand() / (float)RAND_MAX);
        if(rand() % 2)
            M.elements[i] = - M.elements[i];
    }
    return M;
}


void PrintVector(float* ma, int X)
{
  int i;
  for (i=0;i<X;i++) {
      printf("%4f ",ma[i]);
  }
  printf("\n");
}

// Serialize Vector
void SerializeVector(Vector V, const char *filename)
{
    std::ofstream f(filename);
    for(unsigned int i = 0; i < V.length; i++) {
       f << V.elements[i] << '\n';
    } 
}


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
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    // 2D Thread ID
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;
 
    for (int k = 0; k < M.width; ++k)
    { 
         float Melement = M.elements[ty * M.width + k];
         float Nelement = N.elements[k * N.width + tx];
         Pvalue += Melement * Nelement;
    } 
    // Write the matrix to device memory;
    // each thread writes one element
    P.elements[ty * P.width + tx] = Pvalue;
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix M) {
    cudaFree(M.elements);
}

void FreeMatrix(Matrix M) {
    free(M.elements);
}

// Copy one matrix to another
void CopyMatrix(Matrix Mout, Matrix Minp)
{
    int size = Minp.width * Minp.height * sizeof(float);
    memcpy(Mout.elements, Minp.elements, size);
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
  cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
  cudaMemcpyDeviceToHost);
}

// Matrix multiplication on the device
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory
    
     // Setup the execution configuration
    dim3 dimBlock(WIDTH, WIDTH);
    dim3 dimGrid(1, 1);

    // Launch the device computation threads!
    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 

    // Free device matrices
    FreeDeviceMatrix(Md);
    FreeDeviceMatrix(Nd);
    FreeDeviceMatrix(Pd);
} 

Matrix AllocateMatrix(int height, int width)
{
    Matrix M;
    M.width = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

    M.elements = (float*) malloc(size*sizeof(float));

    for(unsigned int i = 0; i < M.height * M.width; i++)
    {
        M.elements[i] = (rand() / (float)RAND_MAX);
        if(rand() % 2)
            M.elements[i] = - M.elements[i];
    }
    return M;
}


void PrintMatrix(float* ma, int X, int Y)
{
  int i,j;
  for (j=0;j<Y;j++) {
    for (i=0;i<X;i++) {
      printf("%4f ",ma[i+j*X]);
    }
    printf("\n");
  }
  printf("\n");
}


// Serialize Matrix
void SerializeMatrix(Matrix M, const char *filename)
{
    std::ofstream f(filename);
    for(unsigned int i = 0; i < M.width; i++) 
    {
        for(unsigned int j = 0; j < M.height; j++)
        {
            f << M.elements[i * M.height + j] << '\n';
        }

    } 
}

