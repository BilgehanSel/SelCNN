#include "FullyConnected.h"




void FullyConnected::Init(
	const std::vector<unsigned> _neuron_counts,
	const unsigned _input_count,
	float* _x,
	float* _p_gradient,
	const unsigned _batch_size)
{
	// for initializer list
	neuron_counts = _neuron_counts;
	input_count =_input_count;
	x = _x;
	p_gradient = _p_gradient;
	batch_size = _batch_size;

	// creating fcLayers
	fcLayers = std::vector<FullyConnectedLayer>(neuron_counts.size());
	for (unsigned i = 0; i != neuron_counts.size(); i++)
	{
		if (i == 0)
		{
			fcLayers[i].Init(
				neuron_counts[i],
				input_count,
				x,
				p_gradient,
				batch_size
			);
		}
		else {
			fcLayers[i].Init(
				neuron_counts[i],
				neuron_counts[i - 1],
				fcLayers[i - 1].y,
				fcLayers[i - 1].gradient,
				batch_size
			);
		}
	}

	y = fcLayers.back().y;
	gradient = fcLayers.back().gradient;

}

// default constructor
FullyConnected::FullyConnected()
{
	neuron_counts = std::vector<unsigned>();
}


FullyConnected::~FullyConnected()
{
}

void FullyConnected::Forward()
{
	//ReadArrayAPI(fcLayers.back().w, neuron_counts.back());
	for (auto& i : fcLayers)
	{
		i.FeedForward();
	}
}

void FullyConnected::Backprop()
{
	for (int i = fcLayers.size() - 1; i >= 0; i--)
	{
		fcLayers[i].Backprop();
	}
}
