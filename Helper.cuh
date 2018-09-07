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
#include "HelperHost.h"



template <typename T>
__global__ void ReadArray(T* Arr, unsigned  n);



template <typename T>
void ReadArrayAPI(T* Arr, unsigned const n);


__global__ void SoftmaxLossBackprop(
	const float* result, float* error,
	unsigned const classCount, unsigned const batch_size,
	const unsigned* label_ptr);

void SoftmaxLossBackpropAPI(
	const float* result, float* error,
	unsigned const classCount, unsigned const batch_size,
	const unsigned* label_ptr);
