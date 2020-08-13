/*
 * Author:  Vijay Gopal Chilkuri
 * Email:   vijay.gopal.c@gmail.com
 * Date:    12-08-2020
 */

#include "../lib/linear_algebra_helpers.h"

// 2^13
#define NMC  8192 
#define LenVec 256
#define NBdim  64

// ----------------------------------------------------
// Parallel reduction on GPU based on presentation
// by Mark Harris, NVIDIA.
// 
// 
// reduction kernel level 0
//
// ----------------------------------------------------
__global__  void vectorReduction0(Vector g_idata, Vector g_odata){

    // Size automatically determined using third execution control parameter
    // when kernel is invoked.
    extern __shared__ float sdata[];

    int tid     = threadIdx.x;
    int index   = blockIdx.x * blockDim.x + threadIdx.x;

    // This instruction copies data from 
    // global to shared memory of each block.
    // Only threads of a block can access this shared memory.
    sdata[tid]  = g_idata.elements[index];

    // Synchronize threads, basically a barrier.
    __syncthreads();
    
    // Do the reduction in shared memory buffer
    // Thread Id:  0 - 1 - 2 - 3 - 4 - 5
    //             |  /    |  /    |  /
    //             0       2       4
    for(unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if(tid % (2*s) == 0)
        {
           sdata[tid] += sdata[tid + s];
        }
    }

    __syncthreads();

    // Write back result to global memory
    if(tid == 0) g_odata.elements[blockIdx.x] = sdata[0];
}

int main(void) 
{
    int i,j;
    // Allocate and initialize the matrices
    Matrix  M     = AllocateMatrix(WIDTH, WIDTH);
    Vector  V     = AllocateVector(WIDTH * WIDTH);

    // Timing stuff
    struct timeval t1, t2;
    double time = 0.0;
 
    // Initialize Matrix of grid points
    for(unsigned int i = 0; i < M.height; i++)
    {
        for(unsigned int j=0;j < M.width; j++)
        {
            M.elements[i * M.width + j] = (WIDTH/2 - i)*(WIDTH/2 - i) + (WIDTH/2 - j)*(WIDTH/2 - j);
            V.elements[i * M.width + j] = M.elements[i * M.width + j];
        }
    }
 
    PrintVector(V.elements,V.length);
    printf("\n");
 
//  // X and Y vectors
//  Vector IdxI = AllocateVector(WIDTH);
//  Vector IdxJ = AllocateVector(WIDTH);
 
//  SerializeVector(IdxI, "random_numbers.dat");
//  SerializeMatrix(M, "DistanceMatrix.dat");

    // Serial Reduction of Vector elements
    float sum = 0;
    
    gettimeofday(&t1, 0);

    for(unsigned int i=0; i < V.length; i++)
    {
        sum += V.elements[i];
    }
    printf("Serial Sum=%5.1f\n",sum);
    
    gettimeofday(&t2, 0);
    
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    
    printf("Time serial sum:  %3.1f ms \n", time);


    gettimeofday(&t1, 0);

    // Parallel reduction level 0
    int NBlocks           = WIDTH*WIDTH/NBdim;
    int NThreadsPerBlock  = NBdim;
    dim3 dimBlock(NThreadsPerBlock);
    dim3 dimGrid(NBlocks);
    Vector Vout     = AllocateZeroVector(NBlocks);

    // Create device vectors
    Vector Vinp_d     = AllocateDeviceVector(V);
    Vector Vout_d     = AllocateDeviceVector(Vout);

    // Copy data to device vector
    CopyToDeviceVector(Vinp_d, V);

    // Copy vectors to device

    printf("NBlocks = %d NThreadsPerBlock=%d \n",NBlocks,NThreadsPerBlock);

    vectorReduction0<<<dimGrid, dimBlock, NBlocks>>>(Vinp_d, Vout_d);
//  VectorMulKernel<<<dimGrid, dimBlock>>>(Vinp_d, Vinp_d, Vout_d);

    // Copy data from device
    CopyFromDeviceVector(Vout, Vout_d);

    printf("Output Vector\n");
    PrintVector(Vout.elements,Vout.length);
    sum = 0;
    for(unsigned int i=0; i < Vout.length; i++)
    {
        sum += Vout.elements[i];
    }
    printf("parallel Sum=%5.1f\n",sum);

    // Parallel reduction level 1
    int WIDTH_2 = WIDTH/2;
    NBlocks           = WIDTH_2*WIDTH_2/NBdim;
    NThreadsPerBlock  = NBdim;
    dim3 dimBlock1(NThreadsPerBlock);
    dim3 dimGrid1(NBlocks);

    // Create device vectors
    Vinp_d     = AllocateDeviceVector(Vout);
    // Copy data to device vector
    CopyToDeviceVector(Vinp_d, Vout);

    Vout     = AllocateZeroVector(NBlocks);
    Vout_d     = AllocateDeviceVector(Vout);

    // Copy vectors to device

    printf("NBlocks = %d NThreadsPerBlock=%d \n",NBlocks,NThreadsPerBlock);

    vectorReduction0<<<dimGrid1, dimBlock1, NBlocks>>>(Vinp_d, Vout_d);
//  VectorMulKernel<<<dimGrid, dimBlock>>>(Vinp_d, Vinp_d, Vout_d);

    // Copy data from device
    CopyFromDeviceVector(Vout, Vout_d);

    printf("Output Vector\n");
    PrintVector(Vout.elements,Vout.length);
    sum = 0;
    for(unsigned int i=0; i < Vout.length; i++)
    {
        sum += Vout.elements[i];
    }
    printf("parallel Sum=%5.1f\n",sum);

    cudaThreadSynchronize();

    gettimeofday(&t2, 0);
    
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    
    printf("Time parallel sum:  %3.1f ms \n", time);

    // Free matrices
    FreeMatrix(M);
    FreeVector(V);
    FreeVector(Vout);
    FreeDeviceVector(Vinp_d);
    FreeDeviceVector(Vout_d);
//  FreeVector(IdxI);
//  FreeVector(IdxJ);
//  FreeVector(PiValues);

    return 0;
}
