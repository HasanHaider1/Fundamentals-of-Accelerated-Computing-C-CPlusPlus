__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);

  /*
   * Conduct experiments to learn more about the behavior of
   * `cudaMallocManaged`.
   *
   * What happens when unified memory is accessed only by the GPU?
   * What happens when unified memory is accessed only by the CPU?
   * What happens when unified memory is accessed first by the GPU then the CPU?
   * What happens when unified memory is accessed first by the CPU then the GPU?
   *
   * Hypothesize about UM behavior, page faulting specificially, before each
   * experiment, and then verify by running `nsys`.
   */
   
   //What happens when unified memory is accessed only by the GPU?
   //ANS: No page faults
   //hostFunction(a,N);
   
   //What happens when unified memory is accessed only by the CPU?
   //ANS: No page faults
   //size_t threadsPerBlock;
   //size_t numberOfBlocks;
   //threadsPerBlock = 256;
   //numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
   //deviceKernel<<<numberOfBlocks, threadsPerBlock>>>(a, N);
   //cudaDeviceSynchronize();
   
   //What happens when unified memory is accessed first by the GPU then the CPU?
   //ANS: 786 memory operations on host, 3 on device. 
   //size_t threadsPerBlock;
   //size_t numberOfBlocks;
   //threadsPerBlock = 256;
   //numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
   //deviceKernel<<<numberOfBlocks, threadsPerBlock>>>(a, N);
   //hostFunction(a,N);
   //cudaDeviceSynchronize();
   
   //What happens when unified memory is accessed first by the CPU then the GPU?
   //ANS: 3504 operations on device.
   size_t threadsPerBlock;
   size_t numberOfBlocks;
   threadsPerBlock = 256;
   numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
   hostFunction(a,N);
   deviceKernel<<<numberOfBlocks, threadsPerBlock>>>(a, N);
   cudaDeviceSynchronize();
   
   
   
   
   
   
   
   
   
   
   
   

  cudaFree(a);
}
