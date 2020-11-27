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

    for (auto param : _target_value_network->parameters()) {
        param.set_requires_grad(false);
    }

    // Copy over network params with averaging
    _transfer_params_v2(*_value_network, *_target_value_network);

}

SACAgent::~SACAgent() {

    delete _q_net1;
    delete _q_net2;
    delete _policy_net;
    delete _alpha_optimizer;
    delete _value_network;
    delete _target_value_network;
}

void SACAgent::_transfer_params_v2(torch::nn::Module& from, torch::nn::Module& to, bool param_smoothing) {
    auto to_params = to.named_parameters(true);
    auto from_params = from.named_parameters(true);

    try {
        for (auto& from_param : from_params) {
            to_params[from_param.key()].data().mul_(1.0 - _tau);
            to_params[from_param.key()].data().add_(_tau * from_param.value().data());
        }
    } catch (const c10::Error& e) {   
        throw std::runtime_error("could not transfer params: " + e.msg());
    }
    
}

void SACAgent::save_policy() {
    std::string path = get_current_dir_name();
    std::string PModelFile = path + "/models/checkpoint/P_Net_Checkpoint.pt";

    torch::serialize::OutputArchive PModelArchive;
    _policy_net->save(PModelArchive);
    PModelArchive.save_to(PModelFile);
}

void SACAgent::load_policy() {
    if (pthread_mutex_lock(&_policyNetLock) == 0) {
        std::string path = get_current_dir_name();
        std::string PModelFile = path + "/models/checkpoint/P_Net_Checkpoint.pt";

        torch::serialize::InputArchive PModelArchive;
        PModelArchive.load_from(PModelFile);
        _policy_net->load(PModelArchive);
        pthread_mutex_unlock(&_policyNetLock);
    }
    else {
        pthread_mutex_unlock(&_policyNetLock);
        throw std::runtime_error("could not obtain lock when loading policy from disk");
    }
}

void SACAgent::save_checkpoint(int versionNo)
{
    // Load from file if exists
    std::string path = get_current_dir_name();
    std::string basePath = path + "/models/checkpoint/";
    std::string fullVersionPath = basePath + std::to_string(versionNo) + "/";
    mkdir(basePath.c_str(), 0755);
    std::string QModelFile1 = fullVersionPath + "Q_Net_Checkpoint1.pt";
    std::string QModelFile2 = fullVersionPath + "Q_Net_Checkpoint2.pt";
    std::string PModelFile = fullVersionPath + "P_Net_Checkpoint.pt";
    std::string AlphaFile = fullVersionPath + "Alpha_Checkpoint.pt";
    std::string ValueFile = fullVersionPath + "Value_Checkpoint.pt";
    std::string TargetValueFile = fullVersionPath + "Target_Value_Checkpoint.pt";
    std::string VersionFile = basePath + "Version.pt";

    _version = torch::tensor(versionNo);
    torch::save(_version, VersionFile);

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

    torch::serialize::OutputArchive PModelArchive;
    _policy_net->save(PModelArchive);
    PModelArchive.save_to(PModelFile);

    torch::save(_log_alpha, AlphaFile);
}

bool SACAgent::load_checkpoint()
{
    // Load version file if exists
    int versionNo;
    std::string path = get_current_dir_name();
    std::string basePath = path + "/models/checkpoint/";
    std::string versionFile = basePath + "Version.pt";
    
    if ( Utility::fileExists(versionFile) ) {
        torch::load(_version, versionFile);
        versionNo = _version.item().toInt();
    } else {
        return false;
    }
    
    // Load from file if they exist
    std::string fullVersionPath = basePath + std::to_string(versionNo) + "/";
    mkdir(basePath.c_str(), 0755);
    std::string QModelFile1 = fullVersionPath + "Q_Net_Checkpoint1.pt";
    std::string QModelFile2 = fullVersionPath + "Q_Net_Checkpoint2.pt";
    std::string PModelFile = fullVersionPath + "P_Net_Checkpoint.pt";
    std::string AlphaFile = fullVersionPath + "Alpha_Checkpoint.pt";
    std::string ValueFile = fullVersionPath + "Value_Checkpoint.pt";
    std::string TargetValueFile = fullVersionPath + "Target_Value_Checkpoint.pt";

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
        // torch::Tensor policy_loss = (_alpha * log_pi_t - torch::min(qf1_pi, qf2_pi)).mean();

        // Determine policy advantage and calc loss
        torch::Tensor advantage = torch::min(qf1_pi, qf2_pi) - value_predictions.detach();
        torch::Tensor policy_loss = (_alpha * log_pi_t - advantage).mean();

        // Policy Regularization
        torch::Tensor mean_reg = 1e-3 * torch::mean(mean.sum(1, true).pow(2.0));
        torch::Tensor std_reg = 1e-3 * torch::mean(std.sum(1, true).pow(2.0));

        torch::Tensor actor_reg = mean_reg + std_reg;
        policy_loss += actor_reg;
        
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
                save_checkpoint(_total_update);
            }	
        }

        // Update counters
        _current_save_delay++;
        _current_update++;
        _total_update++;

        // Write loss info to log
        std::string episodeData = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>
                (std::chrono::system_clock::now().time_since_epoch()).count()) + ','
                + std::to_string(_total_update) + ','
                + std::to_string(policy_loss.item().toDouble()) + ','
                + std::to_string(value_loss.item().toDouble()) + ','
                + std::to_string(q_value_loss1.item().toDouble()) + ','
                + std::to_string(q_value_loss2.item().toDouble()) + ','
                + std::to_string(_alpha.item().toDouble());

            Utility::appendLineToFile(_lossPath, episodeData);

    } catch (const c10::Error& e) {   
        throw std::runtime_error("could not update model: " + e.msg());
    }
}

void SACAgent::load_policy(Utility::sharedString* s) {
    if (pthread_mutex_lock(&_policyNetLock) == 0) {
        // Load from stream
        this->_load_from_array(*_policy_net, s);
    }
    else {
        pthread_mutex_unlock(&_policyNetLock);
        throw std::runtime_error("could not obtain lock when loading policy from disk");
    }
}

void SACAgent::save_policy(Utility::sharedString* s) { 
    this->_save_to_array(*_policy_net, s); 
}

void SACAgent::_save_to_array(torch::nn::Module& from, Utility::sharedString* s) {
    // Save to stream
    std::stringstream stream;
    torch::serialize::OutputArchive archive;
    from.save(archive);
    archive.save_to(stream);

    // Get model string
    s->clear();
    std::string modelParams = stream.str();

    for (char const &c : modelParams) {
        *s += c;
    }
}

void SACAgent::_load_from_array(torch::nn::Module& to, Utility::sharedString* s) {

    // Load from stream
    std::istringstream stream(std::string(s->begin(), s->end()));
    torch::serialize::InputArchive archive;
    archive.load_from(stream, device);
    to.load(archive);
}

