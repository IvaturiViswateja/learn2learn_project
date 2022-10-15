#!/usr/bin/env python3
import torch 
import torch.nn.functional as F
import traceback
from torch.autograd import grad
from torch.distributions import Normal, Categorical
from torch import optim
from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module
from learn2learn.algorithms.epg.losses import 
from learn2learn.algorithms.epg.networks import Memory 
from learn2learn.algorithms.epg.utils import 
from learn2learn.algorithms.common.policies import DiagNormalPolicy,CategoricalPolicy
## these are neural networks which gives the required policies 


#import cherry 
#import ppo 
#compute losses and the policy parameters
#feed in the base_epg_learner 
#base epg learner expects the value function and policy functions as well

#def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    #returns = ch.td.discount(gamma, rewards, dones)
    #baseline.fit(states, returns)
    #values = baseline(states)
    #next_values = baseline(next_states)
    #bootstraps = values * (1.0 - dones) + next_values * dones
    #next_value = torch.zeros(1, device=values.device)
    #return ch.pg.generalized_advantage(tau=tau,
                                       #gamma=gamma,
                                       #rewards=rewards,
                                       #dones=dones,
                                       #values=bootstraps,
                                       #next_value=next_value),gamma,rewards,dones,bootstraps


#def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    # Update policy and baseline
   #states = train_episodes.state()
    #actions = train_episodes.action()
    #rewards = train_episodes.reward()
    #dones = train_episodes.done()
    #next_states = train_episodes.next_state()
    #log_probs = learner.log_prob(states, actions)
    #advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    #dones, states, next_states)
    #advantages = ch.normalize(advantages).detach()
    #return a2c.policy_loss(log_probs, advantages),log_probs



import chainer as C
import chainer.functions as F
import numpy as np
from mpi4py import MPI

from epg.launching import logger
from epg.exploration import HashingBonusEvaluator
from epg.losses import Conv1DLoss
from epg.networks import Memory
from epg.utils import sym_mean, gamma_expand, int_to_onehot, onehot_to_int, \
    Adam, Normalizer, gaussian_kl, categorical_kl


class GenericAgent(object):
    def __init__(self,
                 env,
                 env_dim, 
                 act_dim, 
                 policy_output_params,
                 memory_out_size=None, 
                 inner_n_opt_steps=None, 
                 inner_opt_batch_size=None,
                 memory=None, 
                 buffer_size = None,
                 baselines = None,
                 policy = None,
                 inner_optim = None,
                 inner_actor_critic_model = None): #True
        assert inner_n_opt_steps is not None
        assert inner_opt_batch_size is not None
        assert baselines is not None
        assert inner_optim is not None
        
        self.pi = policy
        self._logstd = None
        self.baselines = baselines
        self._use_mem = mem
        self._buffer_size=buffer_size
        self.inner_actor_critic_model = inner_actor_critic_model 
        self.inner_n_opt_steps = inner_n_opt_steps
        self.inner_opt_batch_size = inner_opt_batch_size
        self.inner_optim = inner_optim 
        # define inner_optim = optim.Adam(net.parameters(), lr=0.001, momentum=0.9) outside e
        self._mem_out_size = memory_out_size
        self._mem = Memory(64, self._mem_out_size)

        self.lst_rew_bonus_eval = [HashingBonusEvaluator(dim_key=128, obs_processed_flat_dim=env_dim)]

        self._env_dim = observation_dim
        self._act_dim = action_dim

        # obs_dim, act_dim, rew, aux, done, pi params
        self._traj_in_dim = observation_dim + action_dim + len(
            self.lst_rew_bonus_eval) + 2 + policy_output_params * act_dim + self._mem_out_size

        self._loss = Conv1DLoss(traj_dim_in=self._traj_in_dim)
        self._traj_norm = Normalizer((observation_dim + action_dim + len(self.lst_rew_bonus_eval) + 2,))

    @property
    def backprop_params(self):
        if self._use_mem:
            if self.inner_actor_critic_model:
                return self.baselines.parameters + self._mem.parameters
            else:
                return self._mem.parameters
        else:
            if self.inner_actor_critic_model:
                return self.baselines.parameters
            else:
                return []

    def pi_f(self, x):
        raise NotImplementedError #the policy network not the actual policy 
        #DiagNormal Policy is also a neural network with 2 layer MLP
        #becomes easier to implement as the gaussian policy is already defined
        ##only use this for the distribution of normalised trajectories

    def pilogpi(self, x, y):
        raise NotImplementedError ##use DiagNormal Policies functions 

    def kl(self, params0, params1):
        raise NotImplementedError
        
    def logp(self, params, acts):
        raise NotImplementedError
  
    def critic_value(self, x):
        return self.baselines.forward(x)

    def action_sampled(self, obs):
        raise NotImplementedError

    def set_loss(self, loss):
        self._loss = loss

    def get_loss(self):
        return self._loss

    def process_trajectory(self, traj):
        proc_traj_in = torch.concat(
            [traj] + self._pi_f(traj[..., :self._env_dim]) + \
            [F.tile(self._mem.f(), (traj.shape[0], 1)).data],
            axis=1
        )
        return self._loss.process_trajectory(proc_traj_in)

    def epg_surrogate_loss(self, traj, processed_traj):
        loss_inputs = [traj, processed_traj] + \
                      self.pi_f(traj[..., :self._env_dim]) + \
                      [torch.tile(self._mem.f(), (traj.shape[0], 1))]
        loss_inputs = torch.concat(loss_inputs, axis=1)
        epg_surr_loss = self._loss.loss(loss_inputs)
        return epg_surr_loss

     def compute_ppo_loss(self, obs, acts, at, vt, old_params):
        params = self.pi_f(obs)
        critic_value = F.flatten(self.critic_value(obs))
        ratio = torch.exp(self._logp(params, acts) - self._logp(old_params, acts))
        surr1 = ratio * at
        surr2 = torch.clip(ratio, 1 - self._ppo_clipparam, 1 + self._ppo_clipparam) * at
        ppo_surr_loss = (
                -sym_mean(torch.minimum(surr1, surr2))
                + self._ppo_klcoeff * sym_mean(self.kl(old_params, params))
                + sym_mean(F.square(cv - vt))
        )
        return policy_loss


    def epg_update(self, obs, acts, rews, dones, ppo_factor, inner_opt_freq):

        epg_rews = rews
        # Want to zero out rewards to the EPG loss function?
        # epg_rews = np.zeros_like(rews)

        # Calculate auxiliary functions.
        lst_bonus = []
        for rew_bonus_eval in self.lst_rew_bonus_eval:
            lst_bonus.append(rew_bonus_eval.predict(obs).T)
        auxs = np.concatenate(lst_bonus, axis=0)

        traj_raw = np.c_[obs, acts, epg_rews, auxs, dones].astype(np.float32)
        # Update here, since we only have access to these raws at this specific spot.
        self._traj_norm.update(traj_raw)
        traj = self._traj_norm.norm(traj_raw)
        auxs_pad = np.zeros(self._buffer_size - obs.shape[0], dtype=np.float32)
        rew_pad = np.zeros(self._buffer_size - obs.shape[0], dtype=np.float32)
        done_pad = np.zeros(self._buffer_size - obs.shape[0], dtype=np.float32)
        obs_pad = np.zeros((self._buffer_size - obs.shape[0], obs.shape[1]), dtype=np.float32)
        act_pad = np.zeros((self._buffer_size - acts.shape[0], acts.shape[1]), dtype=np.float32)
        pad = np.hstack([obs_pad, act_pad, rew_pad[:, None], auxs_pad[:, None], done_pad[:, None]])
        traj = np.vstack([pad, traj])
        traj[:, obs.shape[1] + acts.shape[1]] = epg_rews
        traj[:, -1] = dones

        # Since the buffer length can be larger than the set of new samples, we truncate the
        # trajectories here for PPO.
        dones = dones[-inner_opt_freq:]
        rews = rews[-inner_opt_freq:]
        acts = acts[-inner_opt_freq:]
        obs = obs[-inner_opt_freq:]
        _obs = traj[-inner_opt_freq:, :obs.shape[1]]
        n = len(obs)
        
        
        if self.inner_actor_critic_model:
            old_params_sym = self._pi_f(_obs)
            vp = np.ravel(self._vf_f(_obs).data)
            old_params = [item.data for item in old_params_sym]
            advs = gamma_expand(rews + self._ppo_gam * (1 - dones) * np.append(vp[1:], vp[-1]) - vp,
                                self._ppo_gam * self._ppo_lam * (1 - dones))
            vt = advs + vp
            at = (advs - advs.mean()) / advs.std()


        epg_surr_loss = 0.
        pi_params_before = self.pi_f(_obs)
        for _ in range(self.inner_n_opt_steps):
            for idx in np.array_split(np.random.permutation(n), n // self.inner_opt_batch_size):
                # Clear gradients
                for v in self.backprop_params:
                    optimizer.zero_grad()

                # Forward pass through loss function.
                # Apply temporal conv to input trajectory
                processed_traj = self._process_trajectory(traj)
                # Compute epg loss value
                epg_surr_loss_sym = self._compute_loss(traj[idx], processed_traj[idx])
                epg_surr_loss += epg_surr_loss_sym.data

               
                
                # Add bootstrapping signal if needed.
                if self.inner_actor_critic_model:
                    old_params_idx = [item[idx] for item in old_params]
                    ppo_surr_loss = self._compute_ppo_loss(
                        _obs[idx], acts[idx], at[idx], vt[idx], old_params_idx)
                    total_surr_loss = epg_surr_loss_sym * (1 - ppo_factor) + ppo_surr_loss * ppo_factor
                else:
                    total_surr_loss = epg_surr_loss_sym
               

                # Backward pass through loss function
                ## find how to do backward pass in pytorch method 
                total_surr_loss.backward()
                for v ,adam in zip(self.backprop_params,self.inner_optim):
                    if np.isnan(v.grad).any() or np.isinf(v.grad).any():
                        logger.log(
                            "WARNING: gradient update nan on node {}".format(MPI.COMM_WORLD.Get_rank()))
                    else:
                        v.data += adam.step(v.grad)

        pi_params_after = self.pi_f(_obs)

        return epg_surr_loss / (n // self.inner_opt_batch_size) / self.inner_n_opt_steps, \
               np.mean(self.kl(pi_params_before, pi_params_after).data)


class ContinuousGenericAgent(GenericAgent):
    def __init__(self,policy, env_dim, act_dim, inner_lr=None, **kwargs):
        assert inner_lr is not None
        super().__init__(env_dim, act_dim, 2, **kwargs)
        self.pi = policy
        ## sigma is defined internally within DiagNormalPolicy
        ##self.lst_adam = [Adam(var.shape, stepsize=inner_lr) for var in self.backprop_params]
        ##policy = DiagNormalPolicy(env.state_size, env.action_size)
    ##meta_learner = l2l.algorithms.MetaSGD(policy, lr=meta_lr)
    ##baseline = LinearValue(env.state_size, env.action_size)
    ##opt = optim.Adam(meta_learner.parameters(), lr=meta_lr)
    ##all_rewards = []
    
    ##we will define this externally


    @property
    def backprop_params(self):
        return super(ContinuousGenericAgent, self).backprop_params + self._pi.train_vars + [self._logstd]

    def pi_f(self, x):
        mean,std = policy.prob_parameters(x)##already gives the standard deviation so no need of epsilon 
        return mean,std

    def pi_logp(self, obs, acts):
        log_prob = policy.log_prob(obs,acts)

    def logp(self, params, acts):
        mean, logstd = params
        locs = mean
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(1e-8)))
        distribution = Normal(locs = locs,scale = scale)
        return distribution.log_prob(acts).mean(dim=1, keepdim=True)

    def act(self, obs):
        obs = obs.astype(np.float32)
        # Use same normalization as traj.
        traj = np.concatenate(
            [obs, np.zeros(self._act_dim + 2 + len(self.lst_rew_bonus_eval), dtype=np.float32)])
        # Normalize!
        obs = self._traj_norm.norm(traj)[:self._env_dim]
        mean = self._pi.f(obs[np.newaxis, ...]).data
        std = np.exp(self._logstd.data)
        assert (std > 0).all(), 'std not > 0: {}'.format(std)
        return np.random.normal(loc=mean, scale=std).astype(np.float32)[0]

    def kl(self, params0, params1):
        return gaussian_kl(params0, params1)

    @staticmethod
    def act_to_env_format(act):
        if np.isnan(act).any() or np.isinf(act).any():
            logger.log("WARNING: nan or inf action {}".format(act))
            return np.zeros_like(act)
        else:
            return act


class DiscreteGenericAgent(GenericAgent):
    def __init__(self, env_dim, act_dim, policy inner_lr=None, **kwargs):
        assert inner_lr is not None
        assert policy = CategoricalPolicy
        super().__init__(env_dim, act_dim, 1, **kwargs)
        self.pi = policy ## take categorical Policy
         ## select optimizer outside for policy

 
##fina hoe to do backpropagation in pytorch format 
    @property
    def backprop_params(self):
        return super(DiscreteGenericAgent, self).backprop_params + self.pi.parameters()

    def pi_f(self, x):
        return policy.prob_parameters(x)

    def pi_logp(self, obs, acts):
        log_prob,actions = policy.forward(obs)
        return log_prob

    def logp(self, params, acts):
        loc = params[0]
        density = Categorical(logits=loc)
        log_prob = density.log_prob(action).mean().view(-1, 1).detach()
        return log_prob

    def act(self, obs):
        obs = obs.astype(np.float32)
        # Use same normalization as traj.
        traj = np.concatenate(
            [obs, np.zeros(self._act_dim + 2 + len(self.lst_rew_bonus_eval), dtype=np.float32)])

        # Normalize!
        obs = self._traj_norm.norm(traj)[:self._env_dim]
        prob = self.pi.f(obs[np.newaxis, ...]).data
        return int_to_onehot(self.cat_sample(prob)[0], self._act_dim)

    def kl(self, params0, params1):
        return categorical_kl(params0, params1)

    @staticmethod
    def act_to_env_format(act):
        return onehot_to_int(act)
