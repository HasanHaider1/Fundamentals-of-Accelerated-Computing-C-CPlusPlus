#include <stdio.h>

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

/*
 * Device kernel stores into `result` the sum of each
 * same-indexed value of `a` and `b`.
 */

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  //int stride = blockDim.x * gridDim.x;
  
  result[index] = a[index] +b[index];
  //printf("%f /n",result);

/*
 *for(int i = index; i < N; i += stride)
 *{
 *  result[i] = a[i] + b[i];
 *}
 */
}

/*
 * Host function to confirm values in `vector`. This function
 * assumes all values are the same `target` value.
 */

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  /*
   * nsys should register performance changes when execution configuration
   * is updated.
   */
   
 /*
  *Our original time with 256 threads per block with ((N + threadsPerBlock - 1) / threadsPerBlock) Blocks at the end of Exercise 1 was 149813426 nanoseconds
  */
  
 /*
  *1. Changing to one thread per block. but still having enough total threads to do the operation by changing the number of threadblocks
  *The time taken for this was 182202376 nanoseconds
  */
  //threadsPerBlock = 1;
  //numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  
 /*
  *2. Changing to 1024 threads per block. with ((N + threadsPerBlock - 1) / threadsPerBlock) Blocks
  *The time taken for this was 117600895 nanoseconds
  */
  //threadsPerBlock = 1024;
  //numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  
 /*
  *3. Changing to 32 threads per block. with ((N + threadsPerBlock - 1) / threadsPerBlock) Blocks
  *The time taken for this was 157480877 nanoseconds
  */
  //threadsPerBlock = 32;
  //numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  
 /*
  *4. Changing to 128 threads per block. with ((N + threadsPerBlock - 1) / threadsPerBlock) Blocks
  *The time taken for this was 124765401 nanoseconds
  */
  //threadsPerBlock = 128;
  //numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

 /*
  *5. Changing to 128 threads per block. with ((N + threadsPerBlock - 1) / threadsPerBlock) Blocks
  *The time taken for this was 117036285 nanoseconds
  */
  threadsPerBlock = 512;
  numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  
  //It seems the most optimal configuration (out of the ones we tried) was with 512 threads per block
  
  
  
  
  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
