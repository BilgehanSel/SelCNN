#pragma once
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>

extern cudnnHandle_t cudnn;
extern cublasHandle_t cublas;

extern float learning_rate;

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "CUDNN Error on file " << __FILE__ << " on line:  "      \
                << __LINE__ << ' ' \
				<< cudnnGetErrorString(status) << std::endl; \
      /*std::exit(EXIT_FAILURE);*/                               \
    }                                                        \
  }



// macro for checking cuda errors
#define checkCuda(expression)										\
 {																	\
	cudaError_t err = (expression);									\
	if (err != cudaSuccess) {										\
		std::cerr << "Cuda Error on file " << __FILE__				\
				  << " on line: " << __LINE__ << ' '				\
				  << cudaGetErrorString(err) << '\n';				\
	}																\
 }


// Randomizing 'n' number of floats, starting from 'data'
inline void Randomize(float* data, unsigned const n, const float stddev = 0.07f)
{
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
	curandSetPseudoRandomGeneratorSeed(gen, clock());
	curandGenerateNormal(gen, data, n, 0.0f, stddev);
	// For testing purposes
	/*float* h_data = new float[n];
	cudaMemcpy(h_data, data, n * sizeof(float), cudaMemcpyDeviceToHost);
	for (unsigned i = 0; i != n; i++)
	{
	std::cout << h_data[i] << ' ';
	}
	std::cout << '\n';*/
}

// 1d vector cout
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T> vec)
{
	for (auto &i : vec)
	{
		stream << i << ' ';
	}
	return (stream);
}

// to cout an array which resides in gpu ram
extern void ReadArrayAPI(float* Arr, unsigned const n);


// uses a .txt file to adjust learning_rate, the file should only contain a double or floating point number
// for example
// 0.001
inline void UpdateLR(float* learning_rate, unsigned const batch_size)
{
	
	std::ifstream file("learning_rate.txt");
	while (file >> *learning_rate)
	{
		*learning_rate /= batch_size;
	}
}