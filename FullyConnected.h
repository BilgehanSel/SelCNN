#pragma once
#include "HelperHost.h"
#include "FullyConnectedLayer.h"

class FullyConnected
{
public:
	FullyConnected(); // default constructor
	~FullyConnected();

	void Init(
		const std::vector<unsigned> _neuron_counts,
		const unsigned _input_count,
		float* _x,
		float* _p_gradient,
		const unsigned _batch_size
	);

	void Forward();
	void Backprop();
	
	// public members
	float* o{ nullptr };
	float* y{ nullptr }; // output of the last fullyConnectedLayer
	float* gradient{ nullptr }; // last layers gradient
	float* p_gradient{ nullptr };

private:

	// defining vars of the fully connected
	std::vector<unsigned> neuron_counts;
	unsigned input_count = 0;
	float* x{ nullptr }; // output of the convolution layers, device ptr
	unsigned batch_size = 0;

	std::vector<FullyConnectedLayer> fcLayers;







};

