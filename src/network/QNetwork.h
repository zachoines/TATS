#pragma once
#include <torch/torch.h>

struct QNetwork : torch::nn::Module {

private:
	int num_inputs, num_actions, hidden_size;
	double init_w, learning_rate;
	torch::nn::Linear linear1{ nullptr }, linear2{ nullptr }, linear3{ nullptr };
	// torch::nn::LSTM lstm{ nullptr };
	// torch::nn::Dropout dropout{ nullptr };

public:
	torch::optim::Adam* optimizer = nullptr;

	// Constructor
	QNetwork(int num_inputs, int num_actions, int hidden_size, double init_w = 3e-3, double learning_rate = 3e-4);
	~QNetwork();
	torch::Tensor forward(torch::Tensor state, torch::Tensor actions, int batchSize, bool eval = false);

} typedef QN;
