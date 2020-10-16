#pragma 
#include <torch/torch.h>
#include "../util/util.h"
#include "../util/data.h"
#include "PolicyNetwork.h"
#include "QNetwork.h"
#include "ValueNetwork.h"

class SACAgent
{
private:
	double _gamma, _tau, _a_lr, _q_lr, _p_lr;
	int _num_inputs, _num_actions;
	double _action_max, _action_min, _action_scale, _action_bias;
	int _current_update = 0;
	int _current_save_delay = 0;
	int _max_save_delay = 100;
	int _max_delay = 2;
	unsigned long _total_update = 0;

	bool _self_adjusting_alpha;

	// log stats to file
	std::string _lossFileName, _lossPath;
	
	// For internal syncing of access
	pthread_mutex_t _policyNetLock = PTHREAD_MUTEX_INITIALIZER;

	QNetwork* _q_net1; 
	QNetwork* _q_net2;

	/*QNetwork* _target_q_network_1;
	QNetwork* _target_q_network_2;*/

	ValueNetwork* _target_value_network;
	ValueNetwork* _value_network;
	PolicyNetwork* _policy_net;

	torch::Tensor _log_alpha, _alpha;
	c10::Scalar _target_entropy;
	torch::optim::Adam* _alpha_optimizer = nullptr;
	void _save_to(torch::nn::Module& model, std::stringstream& fd);
	void _load_from(torch::nn::Module& model, std::stringstream& fd);
	void _transfer_params_v2(torch::nn::Module& from, torch::nn::Module& to, bool param_smoothing = false);
	int _save_to_array(torch::nn::Module& from, double* address, int index);
	int _save_to_array(torch::Tensor& from, double* address, int index);
	int _load_from_array(torch::nn::Module& to, double* address, int index);
	int _load_from_array(torch::Tensor& to, double* address, int index);

public:
	SACAgent(int num_inputs, int num_hidden, int num_actions, double action_max, double action_min, bool alphaAdjuster = true, double gamma = 0.99, double tau = 5e-3, double alpha = 0.2, double q_lr = 3e-4, double policy_lr = 3e-4, double a_lr = 3e-4);
	~SACAgent();

	void update(int batchSize, Utility::TrainBuffer* replayBuffer);
	int sync(bool parent, double* data);
	torch::Tensor get_action(torch::Tensor state, bool trainMode = true);

	void save_checkpoint();
	bool load_checkpoint();

};

