#include "PolicyNetwork.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <regex>
#include <stack>
#include <cmath>
#include <torch/torch.h>
#include "Normal.h"

PolicyNetwork::PolicyNetwork(int num_inputs, int num_actions, int hidden_size, double init_w, double log_std_min, double log_std_max, double learning_rate, double action_max, double action_min) {
	this->num_inputs = num_inputs;
	this->num_actions = num_actions;
	this->hidden_size = hidden_size;
	this->init_w = init_w;
	this->log_std_min = log_std_min;
	this->log_std_max = log_std_max;
	this->learning_rate = learning_rate;

	_action_max = action_max;
	_action_min = action_min;
	_action_scale = (action_max - action_min) / 2.0;
	_action_bias = (action_max + action_min) / 2.0;

	// Set network structure
	linear1 = register_module("linear1", torch::nn::Linear(num_inputs, hidden_size));
	linear2 = register_module("linear2", torch::nn::Linear(hidden_size, hidden_size));
	mean_Linear = register_module("mean_Linear", torch::nn::Linear(hidden_size, num_actions));
	log_std_linear = register_module("log_std_linear", torch::nn::Linear(hidden_size, num_actions));

	// Initialize params
	torch::autograd::GradMode::set_enabled(false);
	torch::nn::init::xavier_uniform_(linear1->weight, 1.0);
	torch::nn::init::xavier_uniform_(linear2->weight, 1.0);
	torch::nn::init::xavier_uniform_(mean_Linear->weight, 1.0);
	torch::nn::init::xavier_uniform_(log_std_linear->weight, 1.0);
	torch::nn::init::constant_(linear1->bias, 0.0);
	torch::nn::init::constant_(linear2->bias, 0.0);
	torch::nn::init::constant_(mean_Linear->bias, 0.0);
	torch::nn::init::constant_(log_std_linear->bias, 0.0);
	torch::autograd::GradMode::set_enabled(true);

	linear1->weight.set_requires_grad(true);
	linear2->weight.set_requires_grad(true);
	mean_Linear->weight.set_requires_grad(true);
	log_std_linear->weight.set_requires_grad(true);
	linear1->bias.set_requires_grad(true);
	linear2->bias.set_requires_grad(true);
	mean_Linear->bias.set_requires_grad(true);
	log_std_linear->bias.set_requires_grad(true);

	optimizer = new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(learning_rate));
}

PolicyNetwork::~PolicyNetwork() {
	delete optimizer;
}

torch::Tensor PolicyNetwork::forward(torch::Tensor state, int batchSize, bool eval) {
	torch::Tensor X, mean, log_std, test;

	X = torch::relu(linear1->forward(state));
	X = torch::relu(linear2->forward(X)); 
	mean = mean_Linear->forward(X);

	log_std = torch::tanh(log_std_linear->forward(X));
	log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0);

	return torch::cat({ { mean }, { log_std } }, 0);
}


torch::Tensor PolicyNetwork::sample(torch::Tensor state, int batchSize, double epsilon, bool eval) {

	at::Tensor result = this->forward(state, batchSize, eval);
	at::Tensor reshapedResult = result.view({ 2, batchSize, num_actions });

	torch::Tensor mean = reshapedResult[0];
	torch::Tensor log_std = reshapedResult[1];
	torch::Tensor std = torch::exp(log_std);
                                                                                                                  
	Normal normal = Normal(mean, std); 
	torch::Tensor z = normal.rsample(); // Reparameterization
	torch::Tensor action = torch::tanh(z);
	torch::Tensor log_probs = normal.log_prob(z, log_std, mean);

	log_probs = log_probs - torch::log(1.0 - torch::pow(action, 2.0) + epsilon);
	return torch::cat({ { action }, { log_probs }, { torch::tanh(mean) }, { std }, { z } }, 0);
}