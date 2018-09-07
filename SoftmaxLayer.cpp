#include "SoftmaxLayer.h"



SoftmaxLayer::SoftmaxLayer() {} // default

void SoftmaxLayer::Init(
	unsigned const _class_count,
	unsigned const _input_count,
	float* _x,
	float* _p_gradient,
	unsigned const _batch_size)
{
	class_count = _class_count;
	input_count = _input_count;
	x = _x;
	p_gradient = _p_gradient;
	batch_size = _batch_size;

	// allocating weights, 'w'
	size_t const w_bytes = input_count * class_count * sizeof(float);
	checkCuda(cudaMalloc(&w, w_bytes));
	Randomize(w, w_bytes / sizeof(float), 0.04f);


	// allocating biases, 'b' & creating and setting bDesc
	size_t const b_bytes = class_count * sizeof(float);
	checkCuda(cudaMalloc(&b, b_bytes));
	Randomize(b, class_count, 0.01f);
	checkCUDNN(cudnnCreateTensorDescriptor(&bDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		bDesc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		class_count,
		1, 1
	));


	// allocating output before softmax, 'o' & softmax, 'y' & creating and setting yDesc
	size_t const y_bytes = class_count * batch_size * sizeof(float);
	checkCuda(cudaMalloc(&o, y_bytes));
	checkCuda(cudaMalloc(&y, y_bytes));
	checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		yDesc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batch_size,
		class_count,
		1, 1
	));

	// allocating gradient
	checkCuda(cudaMalloc(&gradient, y_bytes));

	// allocating onevec
	checkCuda(cudaMalloc(&onevec, batch_size * sizeof(float)));
	std::vector<float> temp_onevec(batch_size, 1.0f);
	checkCuda(cudaMemcpy(onevec, temp_onevec.data(), batch_size * sizeof(float), cudaMemcpyHostToDevice));
}

SoftmaxLayer::~SoftmaxLayer()
{
	
	/*cudaFree(y);
	cudaFree(w);
	cudaFree(b);
	cudaFree(o);
	cudaFree(gradient);
	checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(bDesc));*/
}

void SoftmaxLayer::FeedForward()
{
	const float alpha = 1.0f, beta = 0.0f;

	// multiplying by weights, o = x*w;
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_N,
		class_count, batch_size, input_count,
		&alpha,
		w, class_count,
		x, input_count,
		&beta,
		o, class_count
	);

	// adding biases, o += b;
	checkCUDNN(cudnnAddTensor(
		cudnn,
		&alpha,
		bDesc, b,
		&alpha,
		yDesc, o
	));

	checkCUDNN(cudnnSoftmaxForward(
		cudnn,
		CUDNN_SOFTMAX_FAST,
		CUDNN_SOFTMAX_MODE_CHANNEL,
		&alpha,
		yDesc, o,
		&beta,
		yDesc, y
	));
}

void SoftmaxLayer::Backprop()
{
	const float alpha = 1.0f, beta = 0.0f;
	//const float learning_rate_l = learning_rate / batch_size;
	const float learning_rate_l = learning_rate;

	// passing gradient to previous layer
	cublasSgemm_v2(
		cublas, CUBLAS_OP_T, CUBLAS_OP_N,
		input_count, batch_size, class_count,
		&alpha,
		w, class_count,
		gradient, class_count,
		&beta,
		p_gradient, input_count
	);

	// updating weights, 'w'
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_T,
		class_count, input_count, batch_size,
		&learning_rate_l,
		gradient, class_count,
		x, input_count,
		&alpha,
		w, class_count
	);

	// updating biases, 'b'
	cublasSgemm_v2(
		cublas, CUBLAS_OP_N, CUBLAS_OP_N,
		class_count, 1, batch_size,
		&learning_rate_l,
		gradient, class_count,
		onevec, batch_size,
		&alpha,
		b, class_count
	);

}
