#!/usr/bin/env python3

"""
Trains a 2-layer MLP with MetaSGD-VPG.
Usage:
python examples/rl/maml_trpo.py
"""

import random

import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c
from cherry.models.robotics import LinearValue
from torch import optim
from tqdm import tqdm

import learn2learn as l2l
from l2l.algorithms.epg.evolution import ES
from policies import CategoricalPolicy







def main(
        experiment='dev',
        env_name='Particles2D-v1',
        adapt_lr=0.1,
        meta_lr=0.01,
        adapt_steps=1,
        num_iterations=200,
        meta_bsz=20,
        adapt_bsz=20,
        tau=1.00,
        gamma=0.99,
        num_workers=2,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        return gym.make(env_name)

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = DiagNormalPolicy(env.state_size, env.action_size)
    baseline = LinearValue(env.state_size, env.action_size)
    es = ES(env,inner_opt_freq=None, inner_max_n_epoch=None, inner_opt_batch_size=None,
            inner_buffer_size=None, inner_n_opt_steps=None, inner_lr=None, inner_use_ppo=None,
            plot_freq=10, gpi=None, mem=None)
    agent = es.create_agent(self, 
                 policy_output_params,
                 memory_out_size=None, 
                 inner_n_opt_steps=None, 
                 inner_opt_batch_size=None,
                 memory=None, 
                 buffer_size = None,
                 baselines = None,
                 policy = None,
                 inner_optim = None,
                 inner_actor_critic_model = None)
    
    
    outer_optim = optim.Adam(agent.parameters(), lr=meta_lr)
    all_rewards = []

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in tqdm(env.sample_tasks(meta_bsz)):  # Samples a new config
            learner = agent.act()
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(agent.act(), episodes=adapt_bsz)
                es = setup_es(seed, train_episodes,log_path, n_cpu, **agent_args)
                es.train(**agent_args, n_cpu=n_cpu)
               



if __name__ == '__main__':
    main()
