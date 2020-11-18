#pragma once
#include <torch/torch.h>

struct ValueNetwork : torch::nn::Module {

private:
    int num_inputs, num_actions, hidden_size;
    double init_w, learning_rate;
    torch::nn::Linear linear1{ nullptr }, linear2{ nullptr }, linear3{ nullptr };

public:
    torch::optim::Adam* optimizer = nullptr;

    // Constructor
    ValueNetwork(int num_inputs, int hidden_size, double init_w = 3e-3, double learning_rate = 3e-4);
    ~ValueNetwork();
    torch::Tensor forward(torch::Tensor state, int batchSize, bool eval = false);

} typedef VN;
