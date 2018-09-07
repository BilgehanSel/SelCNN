#include "CNN.hpp"




CNN::CNN(
	std::vector<std::vector<float>>& _train_data,
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
) :
	train_data(_train_data),
	train_labels(_train_labels),
	test_data(_test_data),
	test_labels(_test_labels),
	class_count(_class_count),
	channel_count(_channel_count),
	imageX(_imageX),
	imageY(_imageY),
	kernel_counts(_kernel_counts),
	kernel_sizes(_kernel_sizes),
	strides(_strides),
	neuron_counts(_neuron_counts),
	train_data_count(_train_data.size()),
	test_data_count(_test_data.size()),
	input_size(_channel_count * _imageX * _imageY),
	convolutional_layer_count(_kernel_counts.size()),
	batch_size(_batch_size)
{


	std::cout << "input size: " << input_size << '\n';
	std::cout << "batch_size: " << batch_size << '\n';
	std::cout << "train_data_count: " << train_data_count << '\n';
	std::cout << "kernel_size: " << kernel_sizes << '\n';
	std::cout << "kernel_count: " << kernel_counts << '\n';
	std::cout << "ImageX: " << imageX << '\n';
	std::cout << "ImageY: " << imageY << '\n';
	std::cout << "class_count: " << class_count << '\n';

	// allocating & copying to d_train_data
	size_t const train_data_bytes = train_data_count * input_size * sizeof(float);
	std::cout << "train_data_bytes: " << train_data_bytes << '\n';
	checkCuda(cudaMalloc(&d_train_data, train_data_bytes));
	for (unsigned i = 0; i != train_data_count; i++) {
		checkCuda(cudaMemcpy(d_train_data + i * input_size, train_data[i].data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
	}

	// allocating & copying to d_train_labels
	size_t const train_labels_bytes = train_data_count * sizeof(unsigned);
	checkCuda(cudaMalloc(&d_train_labels, train_labels_bytes));
	checkCuda(cudaMemcpy(d_train_labels, train_labels.data(), train_labels_bytes, cudaMemcpyHostToDevice));

	// allocating & copying to d_test_data
	size_t const test_data_bytes = test_data_count * input_size * sizeof(float);
	checkCuda(cudaMalloc(&d_test_data, test_data_bytes));
	for (unsigned i = 0; i != test_data_count; i++) {
		checkCuda(cudaMemcpy(d_test_data + i * input_size, test_data[i].data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
	}


	// allocating & copying to d_test_labels
	size_t const test_labels_bytes = test_data_count * sizeof(unsigned);
	checkCuda(cudaMalloc(&d_test_labels, test_labels_bytes));
	checkCuda(cudaMemcpy(d_test_labels, test_labels.data(), test_labels_bytes, cudaMemcpyHostToDevice));

	
	float* dummy_gradient{ nullptr };
	size_t const dummy_gradient_bytes = batch_size * input_size * sizeof(float);
	checkCuda(cudaMalloc(&dummy_gradient, dummy_gradient_bytes));
	// setting up the convolutional layers
	for (unsigned i = 0; i != convolutional_layer_count; i++)
	{
		if (i == 0)
		{
			convLayers.push_back(ConvolutionalLayer(
				kernel_counts[i],
				kernel_sizes[i],
				strides[i],
				channel_count,
				imageX, imageY,
				d_train_data,
				dummy_gradient,/*take care here*/
				batch_size
			));
			// dummy gradient is used because in this version, there is no way to specify the first layer of the convolutional layer...
			// when using ConvolutionalLayer::BackProp(), all layers propagate data back to previous layer, that's why...
		}
		else {
			convLayers.push_back(ConvolutionalLayer(
				kernel_counts[i],
				kernel_sizes[i],
				strides[i],
				kernel_counts[i - 1],
				convLayers.back().outputX,
				convLayers.back().outputY,
				convLayers.back().y,
				convLayers.back().gradient,
				batch_size
			));
		}
	}


	// Initing FullyConnected class object, which will use FullyConnectedLayer class to represent fully connected neurons in LeNET.
	fc.Init(
		neuron_counts,
		convLayers.back().outputX * convLayers.back().outputY * kernel_counts.back(),
		convLayers.back().y,
		convLayers.back().gradient,
		batch_size
	);


	// initing the last layer of the network which is softmax
	softmaxLayer.Init(
		class_count,
		neuron_counts.back(),
		fc.y,
		fc.gradient,
		batch_size
	);
}

CNN::~CNN()
{
}

// Train the network for some number of epochs...
void CNN::Train(unsigned const epoch)
{
	for (unsigned p = 0; p != epoch; p++)
	{
		std::cout << "Epoch is: " << p << '\n';
		unsigned* label_ptr{ nullptr };
		unsigned error_count = 0;
		for (unsigned i = 0; i < (train_data_count / batch_size) * batch_size - batch_size; i += batch_size)
		{


			//std::cout << "iteration is: " << i << std::endl;
			convLayers.front().Set_x(d_train_data + i * input_size); // setting x for the network's input
			label_ptr = d_train_labels + i; // adjusting pointer for the inputs' labels

			// manually forwarding ConvolutionLayer class objects
			for (auto &c : convLayers)
			{
				c.ConvolutionForward();
			}

			fc.Forward(); // manually forwarding FullyConnected class which forwards individual layers
			softmaxLayer.FeedForward(); // manually forwarding softmaxLayer to achieve network's answers...

			// can be commented out if training error rate is not wanted, lowers performance when enabled...
			for (unsigned j = 0; j != batch_size; j++)
			{
				int result = 0;
				cublasIsamax_v2(cublas, class_count, softmaxLayer.y + j * class_count, 1, &result);
				//ReadArrayAPI(softmaxLayer.o + j * class_count, class_count);

				// (result - 1) since cublasIsamax starts counting from 1.
				if (result - 1 != train_labels[i + j])
				{
					error_count++;
				}
			}


			// check if needed to see if gradient goes back. in other words to see if there is a gradient vanishing problem
			/*if (i % 1024 == 0) {
				ReadArrayAPI(convLayers.front().gradient, class_count);
				ReadArrayAPI(softmaxLayer.o, class_count);
			}*/

			// for testing purposes if train_data is big and test accuracy is wanted before a whole epoch...
			if (i % 4992 * 1 == 0 && i != 0) {
				Test();
			}

			/// backprop part...
			// softmaxbackprop
			SoftmaxLossBackpropAPI(softmaxLayer.y, softmaxLayer.gradient, class_count, batch_size, label_ptr);
			softmaxLayer.Backprop();
			fc.Backprop();
			for (int k = convLayers.size() - 1; k >= 0; k--)
			{
				convLayers[k].ConvolutionBackward();
			}
		}

		float error_percent = error_count / float(train_data_count) * 100.f;
		std::cout << "Train data Error Rate: " << error_percent << "\t Learning rate: " << learning_rate * batch_size << '\n';
		Test();
	}
}

// Tests networks performance(accuracy) on the test_data
void CNN::Test()
{


	unsigned* label_ptr{ nullptr };
	unsigned error_count = 0;

	// **i < (test_data_count / batch_size) * batch_size - batch_size** is used not to cause undefined behavior if test_data % batch_size != 0.
	for (unsigned i = 0; i < (test_data_count / batch_size) * batch_size - batch_size; i += batch_size)
	{
		//std::cout << "iteration is: " << i << std::endl;
		convLayers.front().Set_x(d_test_data + i * input_size);
		label_ptr = d_test_labels + i;

		for (auto &c : convLayers)
		{
			c.ConvolutionForward();
		}

		fc.Forward();
		softmaxLayer.FeedForward();
		for (unsigned j = 0; j != batch_size; j++)
		{
			int result = 0;
			cublasIsamax_v2(cublas, class_count, softmaxLayer.y + j * class_count, 1, &result);
			if (result - 1 != test_labels[i + j])
			{
				error_count++;
			}
		}

		// backpropagation part is not used, since we are only testing


	}
	float error_percent = error_count / float(test_data_count) * 100.f;
	std::cout << "Test data Error Rate: " << error_percent << "\t Learning rate: " << learning_rate * batch_size << '\n';

	// learning_rate adjustments
	UpdateLR(&learning_rate, batch_size);

}
