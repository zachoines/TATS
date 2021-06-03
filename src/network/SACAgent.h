#pragma 
#include <torch/torch.h>
#include "PolicyNetwork.h"
#include "QNetwork.h"
#include "ValueNetwork.h"
#include "../util/util.h"
#include "../util/config.h"

class SACAgent
{
private:
    torch::DeviceType device;
    double _gamma, _tau, _a_lr, _q_lr, _p_lr;
    int _num_inputs, _num_actions;
    double _action_max, _action_min, _action_scale, _action_bias;
    int _current_update = 0;
    int _current_save_delay = 0;
    int _max_save_delay = 750;
    int _max_delay = 2;
    unsigned long _total_update = 0;

    bool _self_adjusting_alpha;
    bool _eval_mode;

    // log stats to file
    std::string _lossFileName, _lossPath;
    
    // For internal syncing of access
    pthread_mutex_t _policyNetLock = PTHREAD_MUTEX_INITIALIZER;

    QNetwork* _q_net1; 
    QNetwork* _q_net2;

    ValueNetwork* _target_value_network;
    ValueNetwork* _value_network;
    PolicyNetwork* _policy_net;

    torch::Tensor _log_alpha, _alpha, _version;
    c10::Scalar _target_entropy;
    torch::optim::Adam* _alpha_optimizer = nullptr;
    
    void _transfer_params_v2(torch::nn::Module& from, torch::nn::Module& to, bool param_smoothing = false);
    void _save_to_array(torch::nn::Module& from, Utility::sharedString* s);
    void _load_from_array(torch::nn::Module& to, Utility::sharedString* s);
    
public:
    SACAgent(int num_inputs, int num_hidden, int num_actions, double action_max, double action_min, bool alphaAdjuster = true, double gamma = 0.99, double tau = 5e-3, double alpha = 0.2, double q_lr = 3e-4, double policy_lr = 3e-4, double a_lr = 3e-4, torch::DeviceType device = torch::kCPU);
    ~SACAgent();

    void eval();
    void update(int batchSize, Utility::TrainBuffer* replayBuffer);
    torch::Tensor get_action(torch::Tensor state, bool trainMode = true);

    void save_checkpoint(int versionNo);
    bool load_checkpoint();
    void load_policy(); 
    void save_policy();
    void load_policy(Utility::sharedString* s);
    void save_policy(Utility::sharedString* s);
};

