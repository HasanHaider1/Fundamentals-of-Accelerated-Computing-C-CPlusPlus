#include <stdio.h>
#include <assert.h>

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 20us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */
 
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void saxpy(float * a, float * b, float * c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( tid < N )
    {
        c[tid] = (2 * a[tid]) + b[tid];
    }
}

int main()
{
    float *a, *b, *c;

    int size = N * sizeof (int); // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // Initialize memory
    for( int i = 0; i < N; ++i )
    {
        a[i] = 2;
        b[i] = 1;
        c[i] = 0;
    }
    
   /*
    *Initialy set to 128 threads per block and ((N + threads_per_block - 1) / threads_per_block) blocks
    *resulted in a time of 16308857 nano seconds with 1195 memory operations on device and 4 memory operations on host
    *
    *size_t threads_per_block = 128;
    *size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
    */
    
    int deviceId;
    cudaGetDevice(&deviceId);
  
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    
    size_t threads_per_block = 256;
    size_t number_of_blocks;
    
    //Calculating the number of Blocks needed
    int BlockNum = (N + threads_per_block - 1) / threads_per_block;
  
    //Calculating the closest multiple of the number of streaming processers
    number_of_blocks = (((BlockNum - 1) / props.multiProcessorCount) + 1) * props.multiProcessorCount;
    
    //Set the number of blocks to be a multiple of the number of SMs. resulted in time of 15733043 nanoseconds
    //Prefetching a
    cudaMemPrefetchAsync(a, size, deviceId);
    //Prefetching b
    cudaMemPrefetchAsync(b, size, deviceId);
    //Prefetching c
    cudaMemPrefetchAsync(c, size, deviceId);
    //Prefetching results in a time of 68670 nanoseconds or approx 69us
    
    saxpy<<<number_of_blocks,threads_per_block>>>(a, b, c);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %f, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %f, ", i, c[i]);
    printf ("\n");

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
