#pragma once
#include "HelperHost.h"

class FullyConnectedLayer
{
public:
	FullyConnectedLayer(
		
	);
	~FullyConnectedLayer();

	void Init(
		unsigned const _neuron_count,
		unsigned const _input_count,
		float* _x,
		float* _p_gradient,
		unsigned const _batch_size
	);

	void FeedForward();
	void Backprop();


	// public members
	unsigned neuron_count; //
	unsigned input_count; //
	float* y{ nullptr }; //
	float* gradient{ nullptr }; //
	float* p_gradient{ nullptr }; // previous layer's gradient
	float* w{ nullptr }; //

private:
	float* x{ nullptr }; //
	float* b{ nullptr }; //
	float* o{ nullptr }; //
	cudnnTensorDescriptor_t yDesc;
	cudnnTensorDescriptor_t bDesc;
	unsigned batch_size;

	// onevec
	float* onevec{ nullptr };

	cudnnActivationDescriptor_t activationDesc; // relu
};

