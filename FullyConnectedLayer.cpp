#include "FullyConnectedLayer.h"






void FullyConnectedLayer::Init(
	unsigned const _neuron_count,
	unsigned const _input_count,
	float* _x,
	float* _p_gradient,
	unsigned const _batch_size
)
{
	neuron_count = _neuron_count;
	input_count = _input_count;
	x =_x;
	p_gradient = _p_gradient;
	batch_size =_batch_size;
	std::cout << "input_count: " << input_count << '\n';
	std::cout << "neuron_count: " << neuron_count << '\n';
	std::cout << "batch_size: " << batch_size << '\n';
	// allocating weights, 'w'
	size_t const w_bytes = input_count * neuron_count * sizeof(float);
	//std::cout << "Allocating: " << w_bytes / 1024.0f / 1024.0f << "Mb.\n";
	checkCuda(cudaMalloc(&w, w_bytes));
	Randomize(w, w_bytes / sizeof(float), 0.03f);


	// allocating biases, 'b' & creating and setting bDesc
	size_t const b_bytes = neuron_count * sizeof(float);
	checkCuda(cudaMalloc(&b, b_bytes));
	Randomize(b, neuron_count, 0.01f);
	checkCUDNN(cudnnCreateTensorDescriptor(&bDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		bDesc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		neuron_count,
		1, 1
	));


	// allocating output before relu, 'o' & relu, 'y' & creating and setting yDesc
	size_t const y_bytes = neuron_count * batch_size * sizeof(float);
	checkCuda(cudaMalloc(&o, y_bytes));
	checkCuda(cudaMalloc(&y, y_bytes));
	checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		yDesc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batch_size,
		neuron_count,
		1, 1
	));

	// creating & setting activation for the layers, relu is used
	checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
	checkCUDNN(cudnnSetActivationDescriptor(
		activationDesc,
		CUDNN_ACTIVATION_ELU,
		CUDNN_NOT_PROPAGATE_NAN,
		1.0
	));


	// allocating gradient for this layers output, 'gradient'
	checkCuda(cudaMalloc(&gradient, y_bytes));


	// allocating onevec
	checkCuda(cudaMalloc(&onevec, batch_size * sizeof(float)));
	std::vector<float> temp_onevec(batch_size, 1.0f);
	checkCuda(cudaMemcpy(onevec, temp_onevec.data(), batch_size * sizeof(float), cudaMemcpyHostToDevice));

}

FullyConnectedLayer::FullyConnectedLayer()
{
}

FullyConnectedLayer::~FullyConnectedLayer()
{

}

void FullyConnectedLayer::FeedForward()
{
	const float alpha = 1.0f, beta = 0.0f;

	
	// multiplying by weights, o = x*w;
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_N,
		neuron_count, batch_size, input_count,
		&alpha,
		w, neuron_count,
		x, input_count,
		&beta,
		o, neuron_count
	);

	// adding biases, o += b;
	checkCUDNN(cudnnAddTensor(
		cudnn,
		&alpha,
		bDesc, b,
		&alpha,
		yDesc, o
	));

	// relu
	checkCUDNN(cudnnActivationForward(
		cudnn,
		activationDesc,
		&alpha,
		yDesc, o,
		&beta,
		yDesc, y
	));







}

void FullyConnectedLayer::Backprop()
{
	const float alpha = 1.0f, beta = 0.0f;


	// taking derivative of activation func.
	checkCUDNN(cudnnActivationBackward(
		cudnn,
		activationDesc,
		&alpha,
		yDesc, y,
		yDesc, gradient,
		yDesc, o,
		&beta,
		yDesc, gradient
	));


	// passing gradient to previous layer
	cublasSgemm_v2(
		cublas, CUBLAS_OP_T, CUBLAS_OP_N,
		input_count, batch_size, neuron_count,
		&alpha,
		w, neuron_count,
		gradient, neuron_count,
		&beta,
		p_gradient, input_count
	);


	// updating weights, 'w'
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_T,
		neuron_count, input_count, batch_size,
		&learning_rate,
		gradient, neuron_count,
		x, input_count,
		&alpha,
		w, neuron_count
	);


	// updating biases, 'b'
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_N,
		neuron_count, 1, batch_size,
		&learning_rate,
		gradient, neuron_count,
		onevec, batch_size,
		&alpha,
		b, neuron_count
	);

}
