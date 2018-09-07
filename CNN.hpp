#pragma once
#include "ConvolutionalLayer.hpp"
#include <cuda.h>
#include "Helper.cuh"
#include "HelperHost.h"
#include <iostream>
#include <vector>
#include <cuda_device_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FullyConnected.h"
#include "SoftmaxLayer.h"


class CNN
{
public:
	CNN(std::vector<std::vector<float>>& _train_data,
		std::vector<unsigned>& _train_labels,
		std::vector<std::vector<float>>& _test_data,
		std::vector<unsigned>& _test_labels,
		const unsigned _class_count,
		const unsigned _channel_count,
		const unsigned _imageX,
		const unsigned _imageY,
		std::vector<unsigned>& _kernel_counts,
		std::vector<unsigned>& _kernel_sizes,
		std::vector<unsigned>& _strides,
		std::vector<unsigned>& _neuron_counts,
		const unsigned _batch_size
		);
	
	~CNN();

	void Train(unsigned const epoch);
	void Test();








private:
	// definining vars of the cnn
	unsigned const class_count;
	unsigned const channel_count;
	const unsigned imageX; // of the input image
	const unsigned imageY; // of the input image
	std::vector<unsigned>& kernel_counts; // a reference to kernel counts of convolutional layers
	std::vector<unsigned>& kernel_sizes; // a reference to kernel sizes of convolutional layers
	std::vector<unsigned>& strides; // a reference to kernel strides when performing convolution
	std::vector<unsigned>& neuron_counts; // a reference to neuron counts of the fully connected layer (excluding softmax layer)
	const unsigned batch_size;

	// understood varibles
	const unsigned train_data_count;
	const unsigned test_data_count;
	const unsigned input_size; 
	const unsigned convolutional_layer_count;

	
	// host data
	std::vector<std::vector<float>>& train_data;
	std::vector<unsigned>& train_labels;
	std::vector<std::vector<float>>& test_data;
	std::vector<unsigned>& test_labels;

	// device data ptrs
	float* d_train_data{ nullptr };
	unsigned* d_train_labels{ nullptr };
	float* d_test_data{ nullptr };
	unsigned* d_test_labels{ nullptr };

	// vector of the Convolutional Layers
	std::vector<ConvolutionalLayer> convLayers;

	// FullyConnected
	FullyConnected fc;

	// softmaxLayer
	SoftmaxLayer softmaxLayer;
};

