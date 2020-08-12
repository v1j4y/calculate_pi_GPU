/*
 * Author:  Vijay Gopal Chilkuri
 * Email:   vijay.gopal.c@gmail.com
 * Date:    12-08-2020
 */

#include "linear_algebra_helpers.h"

// 2^13
#define NMC  8192 

// MCMC Kernel
__global__  void my_add(int *a, int *b, int *c){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    c[index] = a[index] + b[index];
}

int main(void) 
{
	  int i,j;
    // Allocate and initialize the matrices
    Matrix  M  = AllocateMatrix(WIDTH, WIDTH);
 
    // Initialize Matrix of grid points
    for(unsigned int i = 0; i < M.height; i++)
    {
        for(unsigned int j=0;j < M.width; j++)
        {
            M.elements[i * M.width + j] = (WIDTH/2 - i)*(WIDTH/2 - i) + (WIDTH/2 - j)*(WIDTH/2 - j);
        }
    }
 
	  PrintMatrix(M.elements,M.width,M.height);
  	printf("\n");
 
    // X and Y vectors
    Vector IdxI = AllocateVector(WIDTH);
    Vector IdxJ = AllocateVector(WIDTH);
 
    SerializeVector(IdxI, "random_numbers.dat");
    SerializeMatrix(M, "DistanceMatrix.dat");

    // Vector of measurements
    Vector PiValues = AllocateVector(NMC);
    

    // Free matrices
    FreeMatrix(M);
    FreeVector(IdxI);
    FreeVector(IdxJ);
    FreeVector(PiValues);

    return 0;
}
