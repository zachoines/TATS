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
    int _max_save_delay = 250;
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

    /***
	 * @brief Soft actor critic agent's constructor
	 * @param filepath Input file to write line too
     * @param num_inputs Number of inputs to SAC agents internal NN 
     * @param num_hidden Number of elements in the hidden layer of SAC agents internal NN 
     * @param num_actions Number of outputs of the of SAC agent's internal policy network
     * @param action_max Max output value of SAC agent's policy network 
     * @param action_min Min output value of SAC agent's policy network
     * @param alphaAdjuster Enable the auto-adjustment of alpha
     * @param gamma Discount value applied to target Q-Values 
     * @param tau Discount factor applied to policy network params
     * @param alpha Initial value applied to Alpha
     * @param q_lr Q-Network learning rate
     * @param policy_lr Policy-Network learning rate
     * @param a_lr Alpha-adjuster learning rate
     * @param device Device SAC agent trains on
	 */
    SACAgent(int num_inputs, int num_hidden, int num_actions, double action_max, double action_min, bool alphaAdjuster = true, double gamma = 0.99, double tau = 5e-3, double alpha = 0.2, double q_lr = 3e-4, double policy_lr = 3e-4, double a_lr = 3e-4, torch::DeviceType device = torch::kCPU);
    ~SACAgent();

    /***
	 * @brief Sets SAC agent's policy network to evaluation mode
	 * @return void
	 */
    void eval();

    /***
	 * @brief Updates SAC agents internal networks with collected batch of experiences
	 * @param batchSize Size of batch to update SAC agent on 
	 * @param replayBuffer buffer passed by reference with collected batch of experiences
	 * @return void
	 */
    void update(int batchSize, Utility::TrainBuffer* replayBuffer);

    /***
	 * @brief Gets predicted action(s) from the SAC agent's policy network
	 * @param state Tensor with input state
	 * @param trainMode Whether to retrieve agents in evaluation mode
	 * @return Tensor with predicted action(s)
	 */
    torch::Tensor get_action(torch::Tensor state, bool trainMode = true);

    /***
	 * @brief Save current checkpoint with provided version number
	 * @param versionNo Number assigned to save
	 * @return void
	 */
    void save_checkpoint(int versionNo);

    /***
	 * @brief Loads all network parameters from most recent save 
	 * @return boolean
	 */
    bool load_checkpoint();

    /***
	 * @brief Loads policy network from save
	 * @return void
	 */
    void load_policy();

    /***
	 * @brief Saves current policy network to file
	 * @return void
	 */ 
    void save_policy();

    /***
	 * @brief Loads policy network from string buffer (used in multi-process syncing of SAC agent)
     * @param s String buffer to write policy network params
	 * @return void
	 */
    void load_policy(Utility::sharedString* s);

    /***
	 * @brief Saves policy network to string buffer (used in multi-process syncing of SAC agent)
     * @param s String buffer to write policy network params
	 * @return void
	 */
    void save_policy(Utility::sharedString* s);
};

