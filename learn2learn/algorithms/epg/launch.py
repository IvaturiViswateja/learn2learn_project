#!/usr/bin/env python3
import sys
sys.path.append('C:\\NSR\\learn2learn_project\\learn2learn\\common')
sys.path.append('C:\\NSR\\learn2learn_project\\learn2learn\\gym')

import random
import cherry as ch
import gym
# import learn2learn as l2l
import numpy as np
from torch import optim
import torch
from mpi4py import MPI
# from cherry.algorithms import a2c
# from cherry.models.robotics import LinearValue
# from learn2learn.algorithms.epg.evolution import ES
from evolution import ES
from async_vec_env import AsyncVectorEnv
from policies import CategoricalPolicy
from tqdm import tqdm


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

    env = AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = CategoricalPolicy(env.state_size, env.action_size)
    #baseline = LinearValue(env.state_size, env.action_size)
    es = ES(env,inner_opt_freq=None, inner_max_n_epoch=None, inner_opt_batch_size=None,
            inner_buffer_size=None, inner_n_opt_steps=None, inner_lr=None, inner_use_ppo=None,
            plot_freq=10, gpi=None, mem=None)
    # agent = es.create_agent(self, 
    #              policy_output_params,
    #              memory_out_size=None, 
    #              inner_n_opt_steps=None, 
    #              inner_opt_batch_size=None,
    #              memory=None, 
    #              buffer_size = None,
    #              baselines = None,
    #              policy = None,
    #              inner_optim = None,
    #              inner_actor_critic_model = None)
    pool_rank = 0
    agent = es.create_agent(env,pool_rank)
    
    outer_optim = optim.Adam(agent.parameters(), lr=meta_lr)
    all_rewards = []

    # def env_selector(env_id, seed=0):
    #     if 'RandomHopper' == env_id:
    #         env = RandomHopper(seed=seed)
    #     elif 'DirHopper' == env_id:
    #         env = DirHopper(seed=seed)
    #     elif 'NormalHopper' == env_id:
    #         env = NormalHopper(seed=seed)
    #     else:
    #         raise Exception('Unknown environment.')
    #     return env

    def setup_es(seed=0,episodes = 0, log_path='/tmp/out', n_cpu=1, **agent_args):
        seed = MPI.COMM_WORLD.Get_rank() * 1000
        assert agent_args is not None
        np.random.seed(seed)
        # env = env_selector(env_id, seed)
        env.seed(seed)
        # es = ES(env, env_id, **agent_args)
        es = ES(env,inner_max_n_epoch = episodes, **agent_args)
        # logger.log('Experiment configuration: {}'.format(str(locals())))
        return es

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in tqdm(env.sample_tasks(meta_bsz)):  # Samples a new config
            learner = agent.act()
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)
            ncpu = 1
            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(agent.act(), episodes=adapt_bsz)
                # es = setup_es(seed, train_episodes,log_path, n_cpu, **agent_args)
                es = setup_es(seed, train_episodes)
                es.train(n_cpu=ncpu)

if __name__ == '__main__':
    main()
