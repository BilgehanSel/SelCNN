#include "ConvolutionalLayer.hpp"



ConvolutionalLayer::ConvolutionalLayer(
	unsigned const _kernel_count,
	unsigned const _kernel_size,
	unsigned const _stride,
	unsigned const _channel_count,
	unsigned const _inputX,
	unsigned const _inputY,
	float* _x,
	float* _p_gradient,
	unsigned const _batch_size
) :
	kernel_count(_kernel_count),
	kernel_size(_kernel_size),
	stride(_stride),
	channel_count(_channel_count),
	inputX(_inputX),
	inputY(_inputY),
	x(_x),
	p_gradient(_p_gradient),
	outputX((_inputX - _kernel_size) / _stride + 1),
	outputY((_inputY - _kernel_size) / _stride + 1),
	batch_size(_batch_size)
{
	
	// creating & setting data descriptor 'xDesc'
	checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		xDesc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batch_size,
		channel_count,
		inputY,
		inputX
	));


	// allocating weights 'w'
	size_t const w_bytes = kernel_count * pow(kernel_size, 2) * channel_count * sizeof(float);
	checkCuda(cudaMalloc(&w, w_bytes));
	// randomizing weights 'w'
	Randomize(w, w_bytes / sizeof(float), 0.03f);
	// creating & setting filter descriptions for weights 'wDesc'
	checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
	checkCUDNN(cudnnSetFilter4dDescriptor(
		wDesc,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
		kernel_count,
		channel_count,
		kernel_size, kernel_size
	));


	// allocating biases 'b'
	size_t const b_bytes = kernel_count * sizeof(float);
	checkCuda(cudaMalloc(&b, b_bytes), 0.01f);
	// randomizing biases 'b'
	Randomize(b, b_bytes / sizeof(float));
	// creating & setting tensor descriptions for biases 'bDesc'
	checkCUDNN(cudnnCreateTensorDescriptor(&bDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		bDesc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1, kernel_count, 1, 1
	));


	// allocating for output before the relu 'o' & after relu 'y'
	size_t const y_bytes = batch_size * kernel_count * outputY * outputX * sizeof(float);
	checkCuda(cudaMalloc(&o, y_bytes));
	checkCuda(cudaMalloc(&y, y_bytes));
	// creating & setting tensor descriptions for 'o' & 'y'
	checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(
		yDesc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batch_size,
		kernel_count,
		outputY, outputX
	));


	// allocating gradients 'gradient', it's size is the same y, and when gradient description is required, yDesc can be used
	size_t const gradient_bytes = batch_size * kernel_count * outputY * outputX * sizeof(float);
	checkCuda(cudaMalloc(&gradient, gradient_bytes));

	// creating & setting activation for the layers, relu is used
	checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
	checkCUDNN(cudnnSetActivationDescriptor(
		activationDesc,
		CUDNN_ACTIVATION_ELU,
		CUDNN_NOT_PROPAGATE_NAN,
		1.0
	));


	// creating & setting convolution descriptor 'convDesc'
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(
		convDesc,
		0, 0, /*zero padding*/
		stride, stride,
		1, 1, /*normal dilation*/
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT
	));


	// getting convolution forward algorithm 'fwd_algo'
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
		cudnn,
		xDesc,
		wDesc,
		convDesc,
		yDesc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0, /*unlimited for now, later will get needed workspace*/
		&fwd_algo
	));


	// getting forward workspace size & allocating 'forward_workspace'
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
		cudnn,
		xDesc,
		wDesc,
		convDesc,
		yDesc,
		fwd_algo,
		&forward_workspace_bytes
	));
	//std::cout << "trying to allocate: " << forward_workspace_bytes / 1024.0f / 1024.0f << "Mb.\n";
	checkCuda(cudaMalloc(&forward_workspace, forward_workspace_bytes));


	// getting convolution backward filter algorithm 'bwd_filter_algo'
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
		cudnn,
		xDesc,
		yDesc, /*same dimensions as dyDesc*/
		convDesc,
		wDesc, /*same dimensions as dwDesc*/
		CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
		0, /*unlimited for now, later will get needed workspace*/
		&bwd_filter_algo
	));


	// getting backward filter workspace size & allocating 'backward_filter_workspace'
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		cudnn,
		xDesc,
		yDesc,
		convDesc,
		wDesc,
		bwd_filter_algo,
		&backward_filter_workspace_bytes
	));
	cudaDeviceSynchronize();
	//std::cout << "trying to allocate: " << backward_filter_workspace_bytes << " bytes.\n";
	checkCuda(cudaMalloc(&backward_filter_workspace, backward_filter_workspace_bytes));


	// getting convolution backward data algorithm 'bwd_data_algo'
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
		cudnn,
		wDesc,
		yDesc,
		convDesc,
		xDesc,
		CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
		0,
		&bwd_data_algo
	));


	// getting backward data workspace size & allocating 'backward_data_workspace'
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
		cudnn,
		wDesc,
		yDesc,
		convDesc,
		xDesc,
		bwd_data_algo,
		&backward_data_workspace_bytes
	));
	//std::cout << "trying to allocate: " << backward_data_workspace_bytes / 1024.0f / 1024.0f << "Mb.\n";
	checkCuda(cudaMalloc(&backward_data_workspace, backward_data_workspace_bytes));


}


ConvolutionalLayer::~ConvolutionalLayer()
{

}

void ConvolutionalLayer::Set_x(float* new_x) { x = new_x; }

void ConvolutionalLayer::ConvolutionForward()
{
	const float alpha = 1.0f, beta = 0.0f;

	// convolution, o = x*w
	checkCUDNN(cudnnConvolutionForward(
		cudnn,
		&alpha,
		xDesc, x,
		wDesc, w,
		convDesc,
		fwd_algo,
		forward_workspace, forward_workspace_bytes,
		&beta,
		yDesc, o
	));

	// adding bias, o += b
	checkCUDNN(cudnnAddTensor(
		cudnn,
		&alpha,
		bDesc, b,
		&alpha,
		yDesc, o
	));

	// relu, y = RELU(o), 'RELU()' is a symbol not an actual function
	checkCUDNN(cudnnActivationForward(
		cudnn,
		activationDesc,
		&alpha,
		yDesc, o,
		&beta,
		yDesc, y
	));


}

void ConvolutionalLayer::ConvolutionBackward()
{
	// used SGD momentum to update weights
	// if wanted to use only sgd, change momentumB with beta, momentumG with alpha 
	// in 'cudnnConvolutionBackwardFilter' and 'cudnnConvolutionBackwardBias'

	const float alpha = 1.0f, beta = 0.0f, momentumB = 0.9f;
	const float momentumG = 1.0f - momentumB;
	//const float p_gradientMul = 1.5f;

	// not sure if it works...
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

	// updating weights?
	checkCUDNN(cudnnConvolutionBackwardFilter(
		cudnn,
		&learning_rate,
		xDesc, x,
		yDesc, gradient,
		convDesc,
		bwd_filter_algo,
		backward_filter_workspace,
		backward_filter_workspace_bytes,
		&alpha,
		wDesc, w
	));

	// updating biases?
	checkCUDNN(cudnnConvolutionBackwardBias(
		cudnn,
		&learning_rate,
		yDesc, gradient,
		&alpha,
		bDesc, b
	));

	// passing the gradient to backward layers
	checkCUDNN(cudnnConvolutionBackwardData(
		cudnn,
		&momentumG,
		wDesc, w,
		yDesc, gradient,
		convDesc,
		bwd_data_algo,
		backward_data_workspace,
		backward_data_workspace_bytes,
		&momentumB,
		xDesc, p_gradient
	));


}
