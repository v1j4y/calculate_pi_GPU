/*
 * Author:  Vijay Gopal Chilkuri
 * Email:   vijay.gopal.c@gmail.com
 * Date:    12-08-2020
 */

#include "parallel_reduction.h"

// 2^13
#define NMC  8192 
#define LenVec 2048
#define NBdim 64

// ----------------------------------------------------
// Parallel reduction on GPU based on presentation
// by Mark Harris, NVIDIA.
// 
// 
// reduction kernel level 0
//
// ----------------------------------------------------
__global__  void vectorReduction(Vector g_idata, Vector g_odata){

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

int parallel_reduction(void) 
{
    int i,j;
    // Allocate and initialize the matrices
    int nParts = 4096;
    Vector  V     = AllocateVector(nParts * LenVec);
    printf("----------------------\n");
    printf("Total Vector Size = %d\n",nParts * LenVec);
    printf("----------------------\n");

    // Timing stuff
    struct timeval t1, t2;
    double time = 0.0;
 
    // Initialize Matrix of grid points
    for(unsigned int i = 0; i < V.length; i++)
    {
        V.elements[i] = 1.0;//(LenVec/2 - i)*(LenVec/2 - i);
    }
 
    gettimeofday(&t1, 0);
    
    // Serial Reduction of Vector elements
    float sum = 0;

    for(unsigned int j=1; j <= nParts; j++)
    {
      for(unsigned int i=0; i < LenVec; i++)
      {
          sum += V.elements[i + (j - 1) * LenVec];
      }
    }
    printf("Serial Sum=%5.1f\n",j,sum);

    gettimeofday(&t2, 0);
    
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    
    printf("Time serial sum:  %3.1f ms \n", time);


    gettimeofday(&t1, 0);

    // Parallel reduction 
    int NBlocks;
    int NThreadsPerBlock;
    Vector Vout, Vout1;
    Vector Vinp_d, Vout_d;
    Vector Vinp1_d, Vout1_d;
    sum = 0;

    for(int idxParts = 1; idxParts <= nParts; idxParts++)
    {

      int dimVec  = LenVec;
      int dimOutVec = dimVec/NBdim;
      Vout = AllocateZeroVector(dimOutVec);

      //--------------------------------------------------------
      // First  level 0
      //--------------------------------------------------------


      NBlocks           = std::min(dimVec/NBdim,64);
      NThreadsPerBlock  = NBdim;

//    printf("Level0 : Nblocks=%d NThreads=%d\n",NBlocks,NThreadsPerBlock);

      // Number of block grids
      dim3 dimGrid(NBlocks);
      // Dimension of each block
      dim3 dimBlock(NThreadsPerBlock);

      // Create device vectors
      Vinp_d     = AllocateDeviceVector(V, LenVec);
      Vout_d     = AllocateDeviceVector(Vout);

      // Copy data to device vector
      CopyToDeviceVector(Vinp_d, V, (idxParts - 1)*LenVec, LenVec);

      vectorReduction<<<dimGrid, dimBlock, NBdim>>>(Vinp_d, Vout_d);


      //--------------------------------------------------------
      // Second level 1
      //--------------------------------------------------------

      dimVec            = Vout_d.length;
      int NBdim1        = 1;
      NBlocks           = 1;
//    NThreadsPerBlock  = dimVec/NBlocks;
      NThreadsPerBlock  = std::min(dimVec/NBlocks,64);
      dimOutVec = NBlocks;
//    printf("Level0 : Nblocks=%d NThreads=%d\n",NBlocks,NThreadsPerBlock);

      // Number of block grids
      dim3 dimGrid1(NBlocks);
      // Dimension of each block
      dim3 dimBlock1(NThreadsPerBlock);

      // Create device vectors
      Vout1 = AllocateZeroVector(dimOutVec);
      Vout1_d     = AllocateDeviceVector(Vout1);

      vectorReduction<<<dimGrid1, dimBlock1, NBdim>>>(Vout_d, Vout1_d);

      // Copy data from device
      CopyFromDeviceVector(Vout1, Vout1_d);

      sum += Vout1.elements[0];
//    printf("i=%d sum=%5f\n",idxParts,sum);
    }

    // print results
    printf("parallel Sum=%5.1f\n",sum);

    cudaThreadSynchronize();
    
    gettimeofday(&t2, 0);
    
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    
    printf("Time parallel sum:  %3.1f ms \n", time);


    // Free matrices
//  FreeMatrix(M);
    FreeVector(V);
    FreeVector(Vout);
    FreeVector(Vout1);
    FreeDeviceVector(Vinp_d);
    FreeDeviceVector(Vout_d);
    FreeDeviceVector(Vinp1_d);
    FreeDeviceVector(Vout1_d);

    return 0;
}
