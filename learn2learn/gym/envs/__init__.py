#!/usr/bin/env python3
import sys
sys.path.append('C:\\NSR\\gym\\gym\\envs')

from registration import register
from .subproc_vec_env import SubprocVecEnv

# KG_env
# ----------------------------------------
register(
    id="KG_task-v1",
    entry_point="learn2learn.gym.envs.Knowledge_graphs.KG_env:KG_task_env",
    max_episode_steps=1000,
)
# 2D Navigation
# ----------------------------------------

register(
    'Particles2D-v1',
    entry_point='learn2learn.gym.envs.particles.particles_2d:Particles2DEnv',
    max_episode_steps=100,
)

# Mujoco
# ----------------------------------------

register(
    'HalfCheetahForwardBackward-v1',
    entry_point='learn2learn.gym.envs.mujoco.halfcheetah_forward_backward:HalfCheetahForwardBackwardEnv',
    max_episode_steps=100,
)

register(
    'AntForwardBackward-v1',
    entry_point='learn2learn.gym.envs.mujoco.ant_forward_backward:AntForwardBackwardEnv',
    max_episode_steps=100,
)

register(
    'AntDirection-v1',
    entry_point='learn2learn.gym.envs.mujoco.ant_direction:AntDirectionEnv',
    max_episode_steps=100,
)

register(
    'HumanoidForwardBackward-v1',
    entry_point='learn2learn.gym.envs.mujoco.humanoid_forward_backward:HumanoidForwardBackwardEnv',
    max_episode_steps=200,
)

register(
    'HumanoidDirection-v1',
    entry_point='learn2learn.gym.envs.mujoco.humanoid_direction:HumanoidDirectionEnv',
    max_episode_steps=200,
)
