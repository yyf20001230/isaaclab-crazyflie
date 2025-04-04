# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games and collect data using torchrl."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--collect_data", action="store_true", default=False, help="Collect data using SyncDataCollector.")
parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to collect data for.")
parser.add_argument("--max_steps", type=int, default=1000000, help="Maximum number of steps to collect data for.")
parser.add_argument("--data_output_dir", type=str, default="collected_data", help="Directory to save collected data.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import time
import torch
import numpy as np
from typing import Dict, Tuple, Any

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# Import TorchRL components
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvBase


class RLGamesEnvWrapper(EnvBase):
    """Wrapper for RL-Games environment to use with TorchRL SyncDataCollector."""
    
    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent
        self._batch_size = None
        
        if hasattr(self.env, "observation_space"):
            self._observation_spec = self.env.observation_space
        else:
            # Fallback for environments without explicit observation_space
            self._observation_spec = None
            
        if hasattr(self.env, "action_space"):
            self._action_spec = self.env.action_space
        else:
            # Fallback for environments without explicit action_space
            self._action_spec = None
        
    def _reset(self, **kwargs) -> TensorDict:
        obs = self.env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        
        # Initialize RNN states if used
        if self.agent.is_rnn:
            self.agent.init_rnn()
            
        # Required: enables the flag for batched observations
        _ = self.agent.get_batch_size(obs, 1)
            
        return TensorDict({"observation": torch.tensor(obs)}, batch_size=[])
    
    def _step(self, tensordict: TensorDict) -> TensorDict:
        # Convert observation to agent format
        obs = tensordict["observation"].cpu().numpy()
        obs = self.agent.obs_to_torch(obs)
        
        # Get action from agent
        with torch.inference_mode():
            actions = self.agent.get_action(obs, is_deterministic=self.agent.is_deterministic)
        
        # Step environment
        next_obs, reward, dones, info = self.env.step(actions)
        
        # Handle RNN states for terminated episodes
        if self.agent.is_rnn and self.agent.states is not None and len(dones) > 0:
            for s in self.agent.states:
                s[:, dones, :] = 0.0
                
        # Create TensorDict for return
        result = TensorDict({
            "next_observation": torch.tensor(next_obs),
            "reward": torch.tensor(reward).view(-1, 1),
            "done": torch.tensor(dones, dtype=torch.bool).view(-1, 1),
            "action": torch.tensor(actions),
        }, batch_size=[])
        
        return result
    
    def _set_seed(self, seed: int):
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
    
    @property
    def batch_size(self):
        if self._batch_size is None and hasattr(self.env, "num_envs"):
            self._batch_size = [self.env.num_envs]
        return self._batch_size


def save_collected_data(data, output_dir):
    """Save collected data to disk."""
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, "collected_data.pt")
    torch.save(data, data_path)
    print(f"[INFO] Saved collected data to {data_path}")


def main():
    """Play with RL-Games agent and optionally collect data."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.physics_dt

    if args_cli.collect_data:
        print(f"[INFO] Setting up SyncDataCollector to collect data for {args_cli.num_episodes} episodes (max {args_cli.max_steps} steps)")
        # Create TorchRL wrapper for environment and agent
        torchrl_env = RLGamesEnvWrapper(env, agent)
        
        # Create data collector
        collector = SyncDataCollector(
            torchrl_env,
            frames_per_batch=100,  # Collect data in batches of 100 steps
            total_frames=args_cli.max_steps,  # Maximum number of steps
            device=rl_device,
            storing_device=rl_device,
            max_frames_per_traj=float('inf'),  # No limit on trajectory length
        )
        
        # Create replay buffer to store collected data
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(args_cli.max_steps),
            sampler=RandomSampler(),
            batch_size=256,  # Batch size for when sampling from buffer
        )
        
        # Collect data
        episode_count = 0
        steps_collected = 0
        collection_start_time = time.time()
        
        for i, batch in enumerate(collector):
            steps_in_batch = batch.numel()
            steps_collected += steps_in_batch
            
            # Add data to replay buffer
            replay_buffer.extend(batch)
            
            # Count completed episodes in this batch
            episodes_in_batch = batch["done"].sum().item()
            episode_count += episodes_in_batch
            
            print(f"[DATA] Batch {i+1}: Collected {steps_in_batch} steps ({steps_collected} total), "
                  f"Completed {episodes_in_batch} episodes ({episode_count} total)")
            
            # Check if we've collected enough episodes
            if episode_count >= args_cli.num_episodes:
                print(f"[INFO] Reached target of {args_cli.num_episodes} episodes")
                break
                
            # Check if we're running out of time or hitting step limit
            if steps_collected >= args_cli.max_steps:
                print(f"[INFO] Reached maximum step limit of {args_cli.max_steps}")
                break
        
        collection_time = time.time() - collection_start_time
        print(f"[INFO] Data collection complete: {steps_collected} steps across {episode_count} episodes in {collection_time:.2f} seconds")
        
        # Save collected data
        data_output_dir = os.path.join(log_root_path, log_dir, args_cli.data_output_dir)
        save_collected_data(replay_buffer, data_output_dir)
    else:
        # Regular playback mode without data collection
        # reset environment
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        timestep = 0
        # required: enables the flag for batched observations
        _ = agent.get_batch_size(obs, 1)
        # initialize RNN states if used
        if agent.is_rnn:
            agent.init_rnn()
        # simulate environment
        # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
        #   attempt to have complete control over environment stepping. However, this removes other
        #   operations such as masking that is used for multi-agent learning by RL-Games.
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # convert obs to agent format
                obs = agent.obs_to_torch(obs)
                # agent stepping
                actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
                # env stepping
                obs, _, dones, _ = env.step(actions)

                # perform operations for terminated episodes
                if len(dones) > 0:
                    # reset rnn state for terminated episodes
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()