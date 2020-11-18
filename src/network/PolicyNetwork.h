#pragma once
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/nn.h>
#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>
#include <utility>

struct PolicyNetwork: torch::nn::Module 
{
private:
    int num_inputs, num_actions, hidden_size, log_std_min, log_std_max;
    double _action_max, _action_min, _action_scale, _action_bias;
    double learning_rate, init_w;
    torch::nn::Linear linear1{ nullptr }, linear2{ nullptr }, mean_Linear{ nullptr }, log_std_linear{ nullptr };
    
public:
    torch::optim::Adam* optimizer = nullptr;
    PolicyNetwork(int num_inputs, int num_actions, int hidden_size, double init_w = 3e-3, double log_std_min = -20.0, double log_std_max = 2.0, double learning_rate = 3e-4, double action_max = 1.0, double action_min = -1.0);
    ~PolicyNetwork();
    at::Tensor forward(torch::Tensor state, int batchSize, bool eval = false);
    at::Tensor sample(torch::Tensor state, int batchSize, double epsilon = 1e-6, bool eval = false);

} typedef PN;

