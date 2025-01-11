import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

from omni.isaac.lab.utils import configclass
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy

from quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg

@configclass
class TrainingConfig:
    # PPO hyperparameters
    learning_rate = 3e-4
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    clip_range_vf = None
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = 0.015
    
    # Training setup
    total_timesteps = 10_000_000
    save_interval = 100_000
    log_interval = 1
    num_envs = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"

class QuadcopterPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            activation_fn=nn.Tanh
        )
        
        # Actor network
        self.actor_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_space.shape[0] * 2)  # Mean and log_std
        )
        
        # Critic network
        self.critic_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))
    
    def forward(self, obs, deterministic=False):
        actor_features = self.actor_net(obs)
        mean, log_std = torch.chunk(actor_features, 2, dim=-1)
        
        # Get standard deviation
        std = torch.exp(log_std.clamp(-20, 2))
        
        # Sample actions
        if deterministic:
            actions = mean
        else:
            distribution = Normal(mean, std)
            actions = distribution.sample()
        
        # Get log probabilities
        log_prob = Normal(mean, std).log_prob(actions).sum(dim=-1)
        
        # Get value prediction
        values = self.critic_net(obs)
        
        return actions, values, log_prob

class PPOTrainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Initialize policy
        self.policy = QuadcopterPolicy(
            env.observation_space,
            env.action_space,
            get_schedule_fn(config.learning_rate)
        ).to(config.device)
        
        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(
            config.n_steps,
            env.observation_space,
            env.action_space,
            config.device,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            n_envs=config.num_envs
        )
        
        # Setup optimizers
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Initialize logging
        self.timesteps = 0
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_path = f"./trained_models/quadcopter_ppo_{now}"
        os.makedirs(self.save_path, exist_ok=True)
    
    def collect_rollouts(self):
        """Collect environment steps using the current policy."""
        obs = self.env.reset()
        for _ in range(self.config.n_steps):
            with torch.no_grad():
                actions, values, log_probs = self.policy(obs)
            
            next_obs, rewards, dones, infos = self.env.step(actions)
            
            self.rollout_buffer.add(obs, actions, rewards, dones, values, log_probs)
            obs = next_obs
            self.timesteps += self.config.num_envs
    
    def train_step(self):
        """Update policy using the currently gathered rollout buffer."""
        # Compute advantages and returns
        self.rollout_buffer.compute_returns_and_advantage(last_values=None, dones=None)
        
        for epoch in range(self.config.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.config.batch_size):
                actions = rollout_data.actions
                obs = rollout_data.observations
                old_values = rollout_data.values
                old_log_prob = rollout_data.log_probs
                advantages = rollout_data.advantages
                returns = rollout_data.returns
                
                # Evaluate actions and values
                new_actions, values, log_prob = self.policy(obs)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Ratio between old and new policy
                ratio = torch.exp(log_prob - old_log_prob)
                
                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                values_pred = values.flatten()
                value_loss = torch.mean((returns - values_pred) ** 2)
                
                # Overall loss
                loss = policy_loss + self.config.vf_coef * value_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
    
    def learn(self):
        """Main training loop."""
        while self.timesteps < self.config.total_timesteps:
            self.collect_rollouts()
            self.train_step()
            
            # Save periodically
            if self.timesteps % self.config.save_interval == 0:
                torch.save(
                    self.policy.state_dict(),
                    os.path.join(self.save_path, f"policy_{self.timesteps}.pth")
                )
            
            # Log progress
            if self.timesteps % self.config.log_interval == 0:
                print(f"Timestep: {self.timesteps}")

def main():
    # Create environment
    env_cfg = QuadcopterEnvCfg()
    env_cfg.scene.num_envs = TrainingConfig.num_envs
    env = QuadcopterEnv(env_cfg)
    
    # Create trainer
    trainer = PPOTrainer(env, TrainingConfig)
    
    # Start training
    trainer.learn()

if __name__ == "__main__":
    main()