#include "ValueNetwork.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <torch/torch.h>


ValueNetwork::ValueNetwork(int num_inputs, int hidden_size, double init_w, double learning_rate)
{
    this->num_inputs = num_inputs;
    this->hidden_size = hidden_size;
    this->init_w = init_w;
    this->learning_rate = learning_rate;

    // construct and register your layers
    linear1 = register_module("linear1", torch::nn::Linear(num_inputs, hidden_size));
    linear2 = register_module("linear2", torch::nn::Linear(hidden_size, hidden_size));
    linear3 = register_module("linear3", torch::nn::Linear(hidden_size, 1));

    torch::autograd::GradMode::set_enabled(false);
    // torch::nn::init::xavier_uniform_(linear1->weight, 1.0);
    // torch::nn::init::xavier_uniform_(linear2->weight, 1.0);
    // torch::nn::init::xavier_uniform_(linear3->weight, 1.0);
    // torch::nn::init::constant_(linear1->bias, 0.0);
    // torch::nn::init::constant_(linear2->bias, 0.0);
    // torch::nn::init::constant_(linear3->bias, 0.0);
    torch::nn::init::kaiming_normal_(linear1->weight);
    torch::nn::init::kaiming_normal_(linear2->weight);
    torch::nn::init::kaiming_normal_(linear3->weight);
    torch::nn::init::constant_(linear1->bias, 0.0);
    torch::nn::init::constant_(linear2->bias, 0.0);
    torch::nn::init::constant_(linear3->bias, 0.0);
    torch::autograd::GradMode::set_enabled(true);

    linear1->weight.set_requires_grad(true);
    linear2->weight.set_requires_grad(true);
    linear3->weight.set_requires_grad(true);
    linear1->bias.set_requires_grad(true);
    linear2->bias.set_requires_grad(true);
    linear3->bias.set_requires_grad(true);

    optimizer = new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(learning_rate));
}

ValueNetwork::~ValueNetwork()
{
    delete optimizer;
}

torch::Tensor ValueNetwork::forward(torch::Tensor state, int batchSize, bool eval)
{
    torch::Tensor X;

    X = torch::leaky_relu(linear1->forward(state));
    X = torch::leaky_relu(linear2->forward(X));
    X = linear3->forward(X);

    return X;

}





