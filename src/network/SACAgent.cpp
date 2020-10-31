#include "SACAgent.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <unistd.h>
#include <string>
#include <fstream>
#include <torch/script.h>
#include <torch/torch.h>
#include "PolicyNetwork.h"
#include "QNetwork.h"
#include "Normal.h"
#include "../util/data.h"
#include "../util/util.h"

SACAgent::SACAgent(int num_inputs, int num_hidden, int num_actions, double action_max, double action_min, bool alphaAdjuster, double gamma, double tau, double alpha, double q_lr, double p_lr, double a_lr, torch::DeviceType device)
{
	
	this->device = device;
	auto optionsDouble = torch::TensorOptions().dtype(torch::kDouble).device(device);
	
	_self_adjusting_alpha = alphaAdjuster;
	_num_inputs = num_inputs;
	_num_actions = num_actions;
	_action_max = action_max;
	_action_min = action_min;
	_action_scale = (action_max - action_min) / 2.0;
	_action_bias = (action_max + action_min) / 2.0;

	_gamma = gamma;
	_tau = tau;
	_alpha = torch::tensor(alpha).to(optionsDouble);
	_a_lr = a_lr;
	_q_lr = q_lr;
	_p_lr = p_lr;

	// initialize networks
	_q_net1 = new QNetwork(num_inputs, num_actions, num_hidden);
	_q_net2 = new QNetwork(num_inputs, num_actions, num_hidden);
	_policy_net = new PolicyNetwork(num_inputs, num_actions, num_hidden, 0.003, -20, 2, 0.001, action_max, action_min);
	_value_network = new ValueNetwork(num_inputs, num_hidden);
	_target_value_network = new ValueNetwork(num_inputs, num_hidden);

	// set default types
	_q_net1->to(device, torch::kDouble, false);
	_q_net2->to(device, torch::kDouble, false);
	_policy_net->to(device, torch::kDouble, false);
	_value_network->to(device, torch::kDouble, false);
	_target_value_network->to(device, torch::kDouble, false);

	// Logging
	_lossFileName = "/stat/trainingLoss.txt";
	_lossPath = get_current_dir_name() + _lossFileName;
	
	
	/*_target_q_network_1 = new QNetwork(num_inputs, num_actions, num_hidden);
	_target_q_network_2 = new QNetwork(num_inputs, num_actions, num_hidden);*/
	_target_entropy = -1 * num_actions;
	
	// Load last checkpoint if available
	if (load_checkpoint()) {

	}
	else {
		_log_alpha = torch::log(_alpha).to(optionsDouble);
		_log_alpha.set_requires_grad(true);
	}

	if (_self_adjusting_alpha) {
		// Auto Entropy adjustment variables
		_alpha_optimizer = new torch::optim::Adam({ _log_alpha }, torch::optim::AdamOptions(a_lr));
	}
	

	/*for (auto param : _target_q_network_1->parameters()){
		param.set_requires_grad(false);
	}
	
	for (auto param : _target_q_network_2->parameters()) {
		param.set_requires_grad(false);
	}*/

	for (auto param : _target_value_network->parameters()) {
		param.set_requires_grad(false);
	}


	// Copy over network params with averaging
	/*_transfer_params_v2(*_q_net1, *_target_q_network_1, true);
	_transfer_params_v2(*_q_net2, *_target_q_network_2, true);*/
	_transfer_params_v2(*_value_network, *_target_value_network);

}

SACAgent::~SACAgent() {

	delete _q_net1;
	delete _q_net2;
	delete _policy_net;
	delete _alpha_optimizer;
	/*delete _target_q_network_1;
	delete _target_q_network_2; */
	delete _value_network;
	delete _target_value_network;
}

void SACAgent::_save_to(torch::nn::Module& module, std::stringstream& fd) {
	
	torch::autograd::GradMode::set_enabled(false);
	torch::serialize::OutputArchive archive;
	auto params = module.named_parameters(true /*recurse*/);
	auto buffers = module.named_buffers(true /*recurse*/);
	for (const auto& val : params) {
		archive.write(val.key(), val.value());
	}
	for (const auto& val : buffers) {
		archive.write(val.key(), val.value(), /*is_buffer*/ true);
	}

	archive.save_to(fd);
	torch::autograd::GradMode::set_enabled(true);
}

void SACAgent::_transfer_params_v2(torch::nn::Module& from, torch::nn::Module& to, bool param_smoothing) {
	// torch::NoGradGuard no_grad;
	auto to_params = to.named_parameters(true);
	auto from_params = from.named_parameters(true);

	try {
		for (auto& from_param : from_params) {
			/*torch::Tensor new_value = from_param.value().data.clone();

			if (param_smoothing) {
				torch::Tensor old_value = to_params[from_param.key()].data.clone();
				new_value = _tau * new_value + (1.0 - _tau) * old_value;
			} 
			
			to_params[from_param.key()].data.copy_(new_value);*/
			to_params[from_param.key()].data().mul_(1.0 - _tau);
			to_params[from_param.key()].data().add_(_tau * from_param.value().data());
		}
	} catch (const c10::Error& e) {   
		throw std::runtime_error("could not transfer params: " + e.msg());
	}
	
}

void SACAgent::_load_from(torch::nn::Module& module, std::stringstream& fd) {
	torch::autograd::GradMode::set_enabled(false);
	torch::serialize::InputArchive archive;
	archive.load_from(fd);
	torch::AutoGradMode enable_grad(false);
	auto params = module.named_parameters(true);
	auto buffers = module.named_buffers(true);
	for (auto& val : params) {
		archive.read(val.key(), val.value());
	}
	for (auto& val : buffers) {
		archive.read(val.key(), val.value(), true);
	}
	torch::autograd::GradMode::set_enabled(true);
}

void SACAgent::save_checkpoint()
{
	// Load from file if exists
	std::string path = get_current_dir_name();
	std::string QModelFile1 = path + "/models/checkpoint/Q_Net_Checkpoint1.pt";
	std::string QModelFile2 = path + "//models/checkpoint/Q_Net_Checkpoint2.pt";
	std::string PModelFile = path + "/models/checkpoint/P_Net_Checkpoint.pt";
	std::string AlphaFile = path + "/models/checkpoint/Alpha_Checkpoint.pt";
	std::string ValueFile = path + "/models/checkpoint/Value_Checkpoint.pt";
	std::string TargetValueFile = path + "/models/checkpoint/Target_Value_Checkpoint.pt";

	torch::serialize::OutputArchive QModelArchive1;
	_q_net1->save(QModelArchive1);
	QModelArchive1.save_to(QModelFile1);

	torch::serialize::OutputArchive QModelArchive2;
	_q_net2->save(QModelArchive2);
	QModelArchive2.save_to(QModelFile2);

	torch::serialize::OutputArchive ValueArchive;
	_value_network->save(ValueArchive);
	ValueArchive.save_to(ValueFile);

	torch::serialize::OutputArchive TargetValueArchive;
	_target_value_network->save(TargetValueArchive);
	ValueArchive.save_to(TargetValueFile);

	/*torch::serialize::OutputArchive ValueArchive;
	_target_q_network_1->save(ValueArchive);
	ValueArchive.save_to(ValueFile);

	torch::serialize::OutputArchive TargetValueArchive;
	_target_q_network_2->save(TargetValueArchive);
	ValueArchive.save_to(TargetValueFile);*/

	torch::serialize::OutputArchive PModelArchive;
	_policy_net->save(PModelArchive);
	PModelArchive.save_to(PModelFile);

	torch::save(_log_alpha, AlphaFile);
}

bool SACAgent::load_checkpoint()
{
	// Load from file if exists
	std::string path = get_current_dir_name();
	std::string QModelFile1 = path + "/models/checkpoint/Q_Net_Checkpoint1.pt";
	std::string QModelFile2 = path + "/models/checkpoint/Q_Net_Checkpoint2.pt";
	std::string PModelFile = path + "/models/checkpoint/P_Net_Checkpoint.pt";
	std::string AlphaFile = path + "/models/checkpoint/Alpha_Checkpoint.pt";
	std::string ValueFile = path + "/models/checkpoint/Value_Checkpoint.pt";
	std::string TargetValueFile = path + "/models/checkpoint/Target_Value_Checkpoint.pt";


	if (
			Utility::fileExists(QModelFile1) && 
			Utility::fileExists(QModelFile2) && 
			Utility::fileExists(PModelFile) &&
			Utility::fileExists(AlphaFile) &&
			Utility::fileExists(ValueFile) &&
			Utility::fileExists(TargetValueFile)
		) 
	{
		torch::serialize::InputArchive QModelArchive1;
		QModelArchive1.load_from(QModelFile1);
		_q_net1->load(QModelArchive1);

		torch::serialize::InputArchive QModelArchive2;
		QModelArchive2.load_from(QModelFile2);
		_q_net2->load(QModelArchive2);

		torch::serialize::InputArchive PModelArchive;
		PModelArchive.load_from(PModelFile);
		_policy_net->load(PModelArchive);

		torch::serialize::InputArchive ValueArchive;
		ValueArchive.load_from(ValueFile);
		_value_network->load(ValueArchive);

		torch::serialize::InputArchive TargetValueArchive;
		TargetValueArchive.load_from(TargetValueFile);
		_target_value_network->load(TargetValueArchive);

		/*torch::serialize::InputArchive ValueArchive;
		ValueArchive.load_from(ValueFile);
		_target_q_network_1->load(ValueArchive);

		torch::serialize::InputArchive TargetValueArchive;
		TargetValueArchive.load_from(TargetValueFile);
		_target_q_network_2->load(TargetValueArchive);*/

		torch::load(_log_alpha, AlphaFile);
		return true;
	}
	else {
		return false;
	}
}

torch::Tensor SACAgent::get_action(torch::Tensor state, bool trainMode)
{
	torch::Tensor next;

	if (trainMode) {
		if (pthread_mutex_lock(&_policyNetLock) == 0) {
			next = _policy_net->sample(state, 1);
			pthread_mutex_unlock(&_policyNetLock);
		}
		else {
			pthread_mutex_unlock(&_policyNetLock);
			throw std::runtime_error("could not obtain lock when getting action");
		}

		torch::Tensor reshapedResult = next.view({ 5, 1, _num_actions });
		return torch::squeeze(reshapedResult[0]);
	}
	else {
		if (pthread_mutex_lock(&_policyNetLock) == 0) {
			next = _policy_net->sample(state, 1, true); // Run in eval mode
			pthread_mutex_unlock(&_policyNetLock);
		}
		else {
			pthread_mutex_unlock(&_policyNetLock);
			throw std::runtime_error("could not obtain lock when getting action");
		}

		torch::Tensor reshapedResult = next.view({ 5, 1, _num_actions });
		return torch::squeeze(reshapedResult[2]); // Return the mean action
	}
}

void SACAgent::update(int batchSize, Utility::TrainBuffer* replayBuffer)
{
	using namespace Utility;
	
	try {
		double states[batchSize][_num_inputs];
		double next_states[batchSize][_num_inputs];
		double actions[batchSize][_num_actions];
		double rewards[batchSize];
		double dones[batchSize];
		double currentStateArray[_num_inputs];
		double nextStateArray[_num_inputs];

		for (int entry = 0; entry < batchSize; entry++) {
			TD train_data = replayBuffer->at(entry);
			train_data.currentState.getStateArray(currentStateArray);
			train_data.nextState.getStateArray(nextStateArray);

			for (int i = 0; i < _num_inputs; i++) {
				states[entry][i] = currentStateArray[i];
				next_states[entry][i] = nextStateArray[i];

				if (i < _num_actions) {
					actions[entry][i] = train_data.actions[i];
				}
			}

			rewards[entry] = train_data.reward;
			dones[entry] = static_cast<double>(train_data.done);
		}

		// Prepare Training tensors
		auto optionsDouble = torch::TensorOptions().dtype(torch::kDouble).device(device);
		torch::Tensor states_t = torch::from_blob(states, { batchSize, _num_inputs }, optionsDouble);
		torch::Tensor next_states_t = torch::from_blob(next_states, { batchSize, _num_inputs }, optionsDouble);
		torch::Tensor actions_t = torch::from_blob(actions, { batchSize, _num_actions }, optionsDouble);
		torch::Tensor rewards_t = torch::from_blob(rewards, { batchSize }, optionsDouble);
		torch::Tensor dones_t = torch::from_blob(dones, { batchSize }, optionsDouble);

		// Sample from Policy
		torch::Tensor current = _policy_net->sample(states_t, batchSize);
		torch::Tensor reshapedResult = current.view({ 5, batchSize, _num_actions });
		torch::Tensor new_actions_t = reshapedResult[0];
		torch::Tensor log_pi_t = reshapedResult[1];
		torch::Tensor mean = reshapedResult[2];
		torch::Tensor std = reshapedResult[3];
		torch::Tensor z_values = reshapedResult[4];
		log_pi_t = log_pi_t.sum(1, true);

		// Update alpha temperature
		if (_self_adjusting_alpha) {

			torch::Tensor alpha_loss = (-1.0 * _log_alpha * (log_pi_t + _target_entropy).detach()).mean();
			_alpha_optimizer->zero_grad();
			alpha_loss.backward();
			_alpha_optimizer->step();
			_alpha = torch::exp(_log_alpha);
			std::cout << "Current alpha: " << _alpha << std::endl;
		}

		// Estimated Q-Values
		torch::Tensor q_prediction_1 = _q_net1->forward(states_t, actions_t, batchSize);
		torch::Tensor q_prediction_2 = _q_net2->forward(states_t, actions_t, batchSize);

		// Training the Q-Value Function
		torch::Tensor target_values = _target_value_network->forward(next_states_t, batchSize);
		torch::Tensor target_q_values = torch::unsqueeze(rewards_t, 1) + _gamma * target_values * torch::unsqueeze(1.0 - dones_t, 1);

		torch::Tensor q_value_loss1 = 0.5 * torch::mean(torch::pow(q_prediction_1 - target_q_values.detach(), 2.0));
		torch::Tensor q_value_loss2 = 0.5 * torch::mean(torch::pow(q_prediction_2 - target_q_values.detach(), 2.0));

		// Training Value Function
		torch::Tensor qf1_pi = _q_net1->forward(states_t, new_actions_t, batchSize);
		torch::Tensor qf2_pi = _q_net2->forward(states_t, new_actions_t, batchSize);
		torch::Tensor value_predictions = _value_network->forward(states_t, batchSize);
		torch::Tensor q_value_predictions = torch::min(qf1_pi, qf2_pi);
		torch::Tensor target_value_func = q_value_predictions - _alpha * log_pi_t;
		torch::Tensor value_loss = 0.5 * torch::mean(torch::pow(value_predictions - target_value_func.detach(), 2.0));
		
		// Training the policy
		torch::Tensor policy_loss = (_alpha * log_pi_t - torch::min(qf1_pi, qf2_pi)).mean();
		
		// Update Policy Network
		if (pthread_mutex_lock(&_policyNetLock) == 0) {
			_policy_net->optimizer->zero_grad();
			policy_loss.backward();
			// torch::nn::utils::clip_grad_norm_(_policy_net->parameters(), 0.5);
			_policy_net->optimizer->step();
			pthread_mutex_unlock(&_policyNetLock);
		}
		else {
			pthread_mutex_unlock(&_policyNetLock);
			throw std::runtime_error("could not obtain lock");
		}

		/*c
		// Determine policy advantage and calc loss
		torch::Tensor advantage = torch::min(qf1_pi, qf2_pi) - value_predictions.detach();
		torch::Tensor policy_loss = (_alpha * log_pi_t - advantage).mean();

		// Policy Regularization
		torch::Tensor mean_reg = 1e-3 * torch::mean(mean.sum(1, true).pow(2.0));
		torch::Tensor std_reg = 1e-3 * torch::mean(std.sum(1, true).pow(2.0));

		torch::Tensor actor_reg = mean_reg + std_reg;
		policy_loss += actor_reg;
		*/

		// Update Q-Value networks
		_q_net1->optimizer->zero_grad();
		q_value_loss1.backward();
		_q_net1->optimizer->step();

		_q_net2->optimizer->zero_grad();
		q_value_loss2.backward();
		_q_net2->optimizer->step();

		// Update Value network
		_value_network->zero_grad();
		value_loss.backward();
		_value_network->optimizer->step();
		
		// Delay update of Target Value and Policy Networks
		if (_current_update >= _max_delay) {
			_current_update = 0;

			// Copy over network params with averaging
			_transfer_params_v2(*_value_network, *_target_value_network, true);

			if (_current_save_delay >= _max_save_delay) {
				_current_save_delay = 0;
				save_checkpoint();
			}	
		}
		else {
			_current_save_delay++;
			_current_update++;
			_total_update++;
		}

		// Write loss info to log
		std::string episodeData = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>
				(std::chrono::system_clock::now().time_since_epoch()).count()) + ','
				+ std::to_string(_total_update) + ','
				+ std::to_string(policy_loss.item().toDouble()) + ','
				+ std::to_string(value_loss.item().toDouble()) + ','
				+ std::to_string(q_value_loss1.item().toDouble()) + ','
				+ std::to_string(q_value_loss2.item().toDouble());

			Utility::appendLineToFile(_lossPath, episodeData);

	} catch (const c10::Error& e) {   
		throw std::runtime_error("could not update model: " + e.msg());
	}
}

int SACAgent::sync(bool parent, double* data)
{
	if (parent) {
		if (pthread_mutex_lock(&_policyNetLock) == 0) {
			int counter = 0;

			// counter = this->_load_from_array(*_q_net1, data, counter);
			// counter = this->_load_from_array(*_q_net2, data, counter);
			counter = this->_load_from_array(*_policy_net, data, counter);
			// counter = this->_load_from_array(*_value_network, data, counter);
			// counter = this->_load_from_array(*_target_value_network, data, counter);
			// counter = this->_load_from_array(_log_alpha, data, counter);
			
			pthread_mutex_unlock(&_policyNetLock);
			return counter;
		}  else {
			pthread_mutex_unlock(&_policyNetLock);
			throw std::runtime_error("could not obtain lock while syncing");
		}
	}
	else {
		try {
			int counter = 0;
			// counter = this->_save_to_array(*_q_net1, data, counter);
			// counter = this->_save_to_array(*_q_net2, data, counter);
			counter = this->_save_to_array(*_policy_net, data, counter);
			// counter = this->_save_to_array(*_value_network, data, counter);
			// counter = this->_save_to_array(*_target_value_network, data, counter);
			// counter = this->_save_to_array(_log_alpha, data, counter);

			return counter;
		} catch (...) {
			throw std::runtime_error("Error while syncing model params with parent");
		}
	}
}

int SACAgent::_save_to_array(torch::nn::Module& from, double* address, int index) {
	torch::autograd::GradMode::set_enabled(false);
	auto params = from.named_parameters(true);

	for (const auto& val : params) {
		
		torch::Tensor value = val.value();
		torch::Tensor layer = torch::flatten(value);
		auto layer_a = layer.accessor<double, 1>();
		int size = layer.numel();
		
		for (int i = 0; i < size; i++) {
			address[index] = layer_a[i];
			index++;
		}
	}

	torch::autograd::GradMode::set_enabled(true);
	return index;
}

int SACAgent::_save_to_array(torch::Tensor& from, double* address, int index) {
	torch::autograd::GradMode::set_enabled(false);

	address[index] = from.data().item().toDouble();
	index++;

	torch::autograd::GradMode::set_enabled(true);
	return index;
}

int SACAgent::_load_from_array(torch::nn::Module& to, double* address, int index) {
	auto optionsDouble = torch::TensorOptions().dtype(torch::kDouble).device(device);
	torch::autograd::GradMode::set_enabled(false);
	auto params = to.named_parameters(true);

	for (const auto& val : params) {

		int size = val.value().numel();
		double copy[size];

		for (int i = 0; i < size; i++) {
			copy[i] = address[index];
			index++;
		}

		params[val.key()].data().copy_(torch::from_blob(copy, val.value().data().sizes(), optionsDouble));

	}

	torch::autograd::GradMode::set_enabled(true);
	return index;
}

int SACAgent::_load_from_array(torch::Tensor& to, double* address, int index) {
	torch::autograd::GradMode::set_enabled(false);

	to.data().copy_(torch::tensor(address[index]));

	torch::autograd::GradMode::set_enabled(true);
	index++;
	return index;
}

// Simplified update function with 4 QNetworks and no regularization
/* void SACAgent::update(int batchSize, TrainBuffer* replayBuffer)
{
	double states[batchSize][_num_inputs];
	double next_states[batchSize][_num_inputs];
	double actions[batchSize][_num_actions];
	double rewards[batchSize];
	double dones[batchSize];
	double currentStateArray[_num_inputs];
	double nextStateArray[_num_inputs];

	for (int entry = 0; entry < batchSize; entry++) {
		TD train_data = replayBuffer->at(entry);
		train_data.currentState.getStateArray(currentStateArray);
		train_data.nextState.getStateArray(nextStateArray);

		for (int i = 0; i < _num_inputs; i++) {
			states[entry][i] = currentStateArray[i];
			next_states[entry][i] = nextStateArray[i];

			if (i < _num_actions) {
				actions[entry][i] = train_data.actions[i];
			}
		}

		rewards[entry] = train_data.reward;
		dones[entry] = static_cast<double>(train_data.done);
	}

	// Prepare Training tensors
	auto optionsDouble = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU, -1);
	torch::Tensor states_t = torch::from_blob(states, { batchSize, _num_inputs }, optionsDouble);
	torch::Tensor next_states_t = torch::from_blob(next_states, { batchSize, _num_inputs }, optionsDouble);
	torch::Tensor actions_t = torch::from_blob(actions, { batchSize, _num_actions }, optionsDouble);
	torch::Tensor rewards_t = torch::from_blob(rewards, { batchSize }, optionsDouble);
	torch::Tensor dones_t = torch::from_blob(dones, { batchSize }, optionsDouble);

	double randChance = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);

	// Sample from policy
	torch::Tensor current = _policy_net->sample(states_t, batchSize);
	torch::Tensor reshapedResult2 = current.view({ 5, batchSize, _num_actions });
	torch::Tensor new_actions_t = reshapedResult2[0];
	torch::Tensor log_pi_t = reshapedResult2[1];
	torch::Tensor mean_t = reshapedResult2[2];
	torch::Tensor std_t = reshapedResult2[3];
	torch::Tensor entropies_t = -log_pi_t.sum(1, true);

	torch::Tensor min_q = torch::min(_q_net1->forward(states_t, new_actions_t, batchSize), _q_net2->forward(states_t, new_actions_t, batchSize));
	torch::Tensor policy_loss = (- min_q - (_alpha * entropies_t) ).mean();

	// Update Policy Network
	if (pthread_mutex_lock(&_policyNetLock) == 0) {
		_policy_net->optimizer->zero_grad();
		policy_loss.backward();
		torch::nn::utils::clip_grad_norm_(_policy_net->parameters(), 0.5);
		_policy_net->optimizer->step();
		pthread_mutex_unlock(&_policyNetLock);
	}
	else {
		pthread_mutex_unlock(&_policyNetLock);
		throw std::runtime_error("could not obtain lock");
	}

	// Update alpha temperature
	if (_self_adjusting_alpha) {
		torch::Tensor alpha_loss = -1.0 * torch::mean(_log_alpha * (_target_entropy - entropies_t.detach()));
		_alpha_optimizer->zero_grad();
		alpha_loss.backward();
		_alpha_optimizer->step();
		_alpha = torch::exp(_log_alpha);
		std::cout << "Current alpha: " << _alpha << std::endl;
	}

	// Sample from Policy
	torch::autograd::GradMode::set_enabled(false);
	torch::Tensor next = _policy_net->sample(next_states_t, batchSize);
	torch::Tensor reshapedResult = next.view({ 5, batchSize, _num_actions });
	torch::Tensor next_actions_t = reshapedResult[0];
	torch::Tensor next_log_pi_t = reshapedResult[1];
	torch::Tensor next_entropies_t = -next_log_pi_t.sum(1, true);
	torch::autograd::GradMode::set_enabled(true);

	// Q-Value loss
	torch::Tensor curr_q1 = _q_net1->forward(states_t, actions_t, batchSize);
	torch::Tensor curr_q2 = _q_net2->forward(states_t, actions_t, batchSize);

	// Estimated Q-Values
	torch::autograd::GradMode::set_enabled(false);
	torch::Tensor next_q1 = _target_q_network_1->forward(next_states_t, next_actions_t, batchSize);
	torch::Tensor next_q2 = _target_q_network_2->forward(next_states_t, next_actions_t, batchSize);
	torch::Tensor next_q_target = torch::min(next_q1, next_q2) + _alpha * next_entropies_t.detach();
	torch::Tensor expected_q = torch::unsqueeze(rewards_t, 1) + torch::unsqueeze(1.0 - dones_t, 1) * _gamma * next_q_target;
	torch::autograd::GradMode::set_enabled(true);

	torch::Tensor q_value_loss1 = 0.5 * torch::mean(torch::pow(curr_q1 - expected_q.detach(), 2.0));
	torch::Tensor q_value_loss2 = 0.5 * torch::mean(torch::pow(curr_q2 - expected_q.detach(), 2.0));


	// Update Q-Value networks
	_q_net1->optimizer->zero_grad();
	q_value_loss1.backward();
	torch::nn::utils::clip_grad_norm_(_q_net1->parameters(), 0.5);
	_q_net1->optimizer->step();

	_q_net2->optimizer->zero_grad();
	q_value_loss2.backward();
	torch::nn::utils::clip_grad_norm_(_q_net2->parameters(), 0.5);
	_q_net2->optimizer->step();


	// Delay update of Target Value and Policy Networks
	if (_current_update >= _max_delay) {

		_current_update = 0;

		// Copy over network params with averaging
		_transfer_params_v2(*_q_net1, *_target_q_network_1, true);
		_transfer_params_v2(*_q_net2, *_target_q_network_2, true);

		if (_current_save_delay == _max_save_delay) {
			_current_save_delay = 0;
			save_checkpoint();
		}

		double randChance = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);

		if (0.05 >= randChance) {
			std::cout << "Here are the log pi: " << log_pi_t << std::endl;
			std::cout << "Here is min q: " << min_q << std::endl;
			std::cout << "Here are entropies: " << entropies_t << std::endl;
		}

		std::cout << "Policy Loss: " << policy_loss << std::endl;
		std::cout << "Q Loss 1: " << q_value_loss1 << std::endl;
		std::cout << "Q Loss 2: " << q_value_loss2 << std::endl;
	}
	else {
		_current_save_delay++;
		_current_update++;
	}
} */