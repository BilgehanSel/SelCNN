#include "Helper.cuh"

template <typename T>
__global__ void ReadArray(T* Arr, unsigned n)
{
	//printf("arr is: ");
	for (unsigned i = 0; i != n; i++)
	{
		printf("%f ", float(Arr[i]));
	}
	printf("\n");
}

// use this in .cpp files
void ReadArrayAPI(float* Arr, unsigned n)
{
	ReadArray << < 1, 1 >> > (Arr, n);
	cudaDeviceSynchronize(); // this line may be not needed.
}

// 
__global__ void SoftmaxLossBackprop(
	const float* result, float* error,
	unsigned const classCount, unsigned const batch_size,
	const unsigned* label_ptr)
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("\nidx is: %u", blockIdx.x);


	if (threadIdx.x == label_ptr[blockIdx.x])
	{
		error[idx] = 1.0f - result[idx];
	}
	else {
		error[idx] = 0.0f - result[idx];
	}

}

// use this from .cpp files
void SoftmaxLossBackpropAPI(
	const float* result, float* error,
	unsigned const classCount, unsigned const batch_size,
	const unsigned* label_ptr)
{
	SoftmaxLossBackprop << < batch_size, classCount >> > (result, error, classCount, batch_size, label_ptr);
	cudaDeviceSynchronize(); // this line may be not needed.
}





