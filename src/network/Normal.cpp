#include "Normal.h"
#include <cmath>
#define _USE_MATH_DEFINES


Normal::Normal(torch::Tensor loc, torch::Tensor scale)
{
    this->loc = loc;
    this->scale = scale;
}


torch::Tensor Normal::sample()
{	
    torch::autograd::GradMode::set_enabled(false);
    return at::normal(loc, scale);
}

torch::Tensor Normal::rsample() {
    torch::Tensor eps = torch::empty(loc.sizes());
    torch::Tensor normal = loc + eps.normal_() * scale;
    return normal;
}

torch::Tensor Normal::log_prob(torch::Tensor value)
{
    torch::Tensor var = torch::pow(scale, 2.0);
    torch::Tensor log_scale = torch::log(scale);

    return -(torch::pow(value - loc, 2.0) / (2.0 * var)) - log_scale - torch::log(torch::sqrt(torch::tensor({ 2.0 * M_PI })));
}


torch::Tensor Normal::log_prob(torch::Tensor z, torch::Tensor log_std, torch::Tensor mean)
{
    return -0.5 * (torch::pow(((z - loc) / (torch::exp(log_std) + 1e-6)), 2.0) + 2.0 * log_std + torch::log(torch::tensor({ 2.0 * M_PI })));
}