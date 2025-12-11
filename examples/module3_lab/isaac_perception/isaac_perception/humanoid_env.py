#!/usr/bin/env python3

"""
Reinforcement Learning Environment for Humanoid Control in Isaac Sim
This environment provides a physics-accurate simulation for training humanoid
locomotion and control policies using reinforcement learning.
"""

import numpy as np
import torch
import gym
from gym import spaces
from pxr import Usd, UsdGeom, Gf, PhysxSchema
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.torch.maths import torch_get_euler_xyz
from omni.isaac.core.utils.torch.rotations import (
    quat_mul, quat_conjugate, quat_apply, quat_from_angle_axis
)
import carb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class HumanoidRLEnv(gym.Env):
    """
    Custom Gym environment for humanoid reinforcement learning in Isaac Sim.
    """

    def __init__(self,
                 stage_units_in_meters=1.0,
                 robot_usd_path=None,
                 num_envs=64,
                 max_episode_length=1000,
                 device="cuda:0",
                 sim_params=None):
        super().__init__()

        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=stage_units_in_meters)
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.device = device
        self.sim_params = sim_params or {}

        # Robot parameters
        self.robot_usd_path = robot_usd_path or self.get_default_robot_path()
        self.initial_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.initial_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        self.initial_orientations[:, 3] = 1.0  # w component of quaternion

        # Define action and observation spaces
        # For a humanoid, this could be joint torques or positions
        self.action_dim = 24  # Example: 12 joints (6 for each leg)
        self.observation_dim = 67  # Example: pos(3) + rot(4) + vel(3) + ang_vel(3) + joint_pos(12) + joint_vel(12) + actions(24) + commands(6)

        # Define action space (continuous torques)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        # Episode tracking
        self.episode_length = torch.zeros(self.num_envs, device=self.device)
        self.episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Initialize the simulation environment
        self.setup_simulation()

        # Humanoid-specific parameters
        self.base_init_state = {
            'pos': [0.0, 0.0, 1.0],  # Initial position (x, y, z)
            'rot': [0.0, 0.0, 0.0, 1.0],  # Initial rotation (x, y, z, w)
            'vlinear': [0.0, 0.0, 0.0],  # Initial linear velocity
            'vangular': [0.0, 0.0, 0.0]  # Initial angular velocity
        }

        # Joint information
        self.dof_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'torso_yaw', 'torso_pitch', 'torso_roll',
            'left_shoulder_pitch', 'left_shoulder_yaw', 'left_shoulder_roll',
            'left_elbow', 'left_wrist_pitch', 'left_wrist_yaw',
            'right_shoulder_pitch', 'right_shoulder_yaw', 'right_shoulder_roll',
            'right_elbow', 'right_wrist_pitch', 'right_wrist_yaw'
        ]  # Example joint names

        # Joint limits (example values, would need to be adjusted based on actual robot)
        self.dof_lower_limits = torch.tensor([
            -1.0, -0.5, -2.0, -2.5, -0.8, -0.5,  # Left leg
            -1.0, -0.5, -2.0, -2.5, -0.8, -0.5,  # Right leg
            -0.5, -0.5, -0.5,  # Torso
            -2.0, -1.5, -2.0, -1.5, -1.0, -1.0,  # Left arm
            -2.0, -1.5, -2.0, -1.5, -1.0, -1.0   # Right arm
        ], device=self.device)

        self.dof_upper_limits = torch.tensor([
            1.0, 0.5, 0.5, 0.5, 0.8, 0.5,  # Left leg
            1.0, 0.5, 0.5, 0.5, 0.8, 0.5,  # Right leg
            0.5, 0.5, 0.5,  # Torso
            2.0, 1.5, 2.0, 1.5, 1.0, 1.0,  # Left arm
            2.0, 1.5, 2.0, 1.5, 1.0, 1.0   # Right arm
        ], device=self.device)

        # Action scale (for converting normalized actions to joint torques)
        self.action_scale = 100.0  # Example torque scale

        # Reward parameters
        self.rew_scales = {
            'lin_vel': 1.0,      # Linear velocity reward
            'ang_vel': 0.1,      # Angular velocity reward
            'joint_acc': -0.0001, # Joint acceleration penalty
            'action_rate': -0.01, # Action rate penalty
            'cosmetic': -1.0,    # Cosmetic penalty for unnatural movements
            'height': 0.5,       # Height maintenance reward
            'alive': 1.0,        # Survival reward
        }

        # Command parameters
        self.commands = torch.zeros(self.num_envs, 6, device=self.device)  # [vx, vy, yaw_rate, pos_x, pos_y, heading]
        self.command_ranges = {
            'linear_vel_x': (-1.0, 1.0),
            'linear_vel_y': (-0.5, 0.5),
            'angular_vel_z': (-1.0, 1.0),
            'heading': (-np.pi, np.pi)
        }

        # Initialize tensors for storing state information
        self.base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)

        self.dof_pos = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.dof_vel = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.dof_torques = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.last_dof_vel = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.default_dof_pos = torch.zeros(self.action_dim, device=self.device)

        # Initialize the environment
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def get_default_robot_path(self):
        """
        Get a default robot USD path. In practice, this would point to a specific humanoid model.
        """
        # This is a placeholder - in a real implementation, you would use a specific humanoid model
        # For now, we'll return a generic path that would need to be set up in Isaac Sim
        return "/Isaac/Robots/Humanoid/humanoid_instanceable.usd"

    def setup_simulation(self):
        """
        Set up the Isaac Sim environment with ground plane and robots.
        """
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Create multiple robot instances in the scene
        for i in range(self.num_envs):
            # Add robot to the scene
            robot_path = f"/World/Robot_{i}"
            # In a real implementation, this would load the actual robot USD
            # For now, we'll just add a placeholder
            add_reference_to_stage(usd_path=self.robot_usd_path, prim_path=robot_path)

            # The robot would be added to the world here
            # robot = self.world.scene.add(Robot(prim_path=robot_path, name=f"robot_{i}"))

        print(f"Initialized {self.num_envs} environments")

    def reset_idx(self, env_ids):
        """
        Reset specific environments to initial state.
        """
        # Reset episode lengths and rewards
        self.episode_length[env_ids] = 0
        self.episode_rewards[env_ids] = 0

        # Randomize initial states
        num_resets = len(env_ids)

        # Randomize base positions
        self.base_pos[env_ids, 0] = torch_rand_float(-0.5, 0.5, (num_resets, 1), self.device).squeeze(1)
        self.base_pos[env_ids, 1] = torch_rand_float(-0.5, 0.5, (num_resets, 1), self.device).squeeze(1)
        self.base_pos[env_ids, 2] = 1.0  # Fixed height

        # Randomize base orientations
        self.base_quat[env_ids] = quat_from_angle_axis(
            torch_rand_float(-0.1, 0.1, (num_resets, 1), self.device).squeeze(1),
            torch.tensor([0, 0, 1], device=self.device).repeat(num_resets, 1)
        )

        # Randomize base velocities
        self.base_lin_vel[env_ids] = torch_rand_float(-0.1, 0.1, (num_resets, 3), self.device)
        self.base_ang_vel[env_ids] = torch_rand_float(-0.1, 0.1, (num_resets, 3), self.device)

        # Randomize joint positions and velocities
        self.dof_pos[env_ids] = self.default_dof_pos.unsqueeze(0) + 0.25 * torch_rand_float(
            -1.0, 1.0, (num_resets, self.action_dim), self.device
        )
        self.dof_vel[env_ids] = torch_rand_float(-0.1, 0.1, (num_resets, self.action_dim), self.device)

        # Reset last_dof_vel
        self.last_dof_vel[env_ids] = self.dof_vel[env_ids]

        # Randomize commands
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges['linear_vel_x'][0],
            self.command_ranges['linear_vel_x'][1],
            (num_resets, 1), self.device
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges['linear_vel_y'][0],
            self.command_ranges['linear_vel_y'][1],
            (num_resets, 1), self.device
        ).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges['angular_vel_z'][0],
            self.command_ranges['angular_vel_z'][1],
            (num_resets, 1), self.device
        ).squeeze(1)

        # Set DOF position targets to current positions
        # In a real implementation, this would set the robot's joint positions
        # For now, we'll just update our internal state

    def compute_observations(self):
        """
        Compute observations for all environments.
        """
        # Calculate base position, orientation, velocities
        root_pos = self.base_pos
        root_quat = self.base_quat
        root_lin_vel = self.base_lin_vel
        root_ang_vel = self.base_ang_vel

        # Calculate projected gravity
        gravity_vec = torch.zeros_like(root_quat)
        gravity_vec[:, 2] = -1.0  # Gravity direction in base frame
        self.projected_gravity = quat_apply(root_quat, gravity_vec)

        # Calculate joint positions and velocities
        dof_pos = self.dof_pos
        dof_vel = self.dof_vel

        # Calculate actions from last step
        prev_actions = self.last_actions

        # Calculate commands
        commands = self.commands

        # Assemble observation vector
        obs = torch.cat([
            root_pos[:, 2:3] - 1.0,  # height (relative to standing height)
            self.projected_gravity,
            root_lin_vel,
            root_ang_vel,
            dof_pos - self.default_dof_pos.unsqueeze(0),
            dof_vel,
            actions,
            commands
        ], dim=-1)

        return obs

    def compute_reward(self):
        """
        Compute rewards for all environments.
        """
        # Calculate various reward components
        lin_vel_reward = torch.sum(self.commands[:, :2] * self.base_lin_vel[:, :2], dim=1) * self.rew_scales['lin_vel']
        ang_vel_reward = self.base_ang_vel[:, 2] * self.commands[:, 2] * self.rew_scales['ang_vel']

        # Joint acceleration penalty
        joint_acc_penalty = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales['joint_acc']

        # Action rate penalty
        action_rate_penalty = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales['action_rate']

        # Height maintenance reward
        height_reward = torch.square(self.base_pos[:, 2] - 1.0) * self.rew_scales['height']

        # Alive reward (for staying upright)
        alive_reward = torch.ones_like(lin_vel_reward) * self.rew_scales['alive']

        # Combine rewards
        total_reward = (
            lin_vel_reward +
            ang_vel_reward +
            joint_acc_penalty +
            action_rate_penalty +
            height_reward +
            alive_reward
        )

        return total_reward

    def compute_reset(self):
        """
        Determine which environments should be reset.
        """
        # Reset if robot falls (too low or too tilted)
        terminated = torch.zeros_like(self.reset_buf)
        terminated = torch.where(self.base_pos[:, 2] < 0.5, torch.ones_like(terminated), terminated)
        terminated = torch.where(torch.abs(self.base_euler[:, 0]) > 1.0, torch.ones_like(terminated), terminated)
        terminated = torch.where(torch.abs(self.base_euler[:, 1]) > 1.0, torch.ones_like(terminated), terminated)

        # Reset if episode length exceeds max
        done = torch.where(self.episode_length >= self.max_episode_length, torch.ones_like(self.reset_buf), terminated)

        return done

    def pre_physics_step(self, actions):
        """
        Apply actions to the simulation before stepping physics.
        """
        # Store actions for reward calculation
        self.last_actions = self.actions.clone()
        self.actions = actions.clone()

        # Convert actions to torques
        torques = actions * self.action_scale

        # Apply torques to the robot joints
        # In a real implementation, this would use Isaac Sim's physics API to apply torques
        # For now, we'll just store the torques
        self.dof_torques = torques

    def post_physics_step(self):
        """
        Process simulation results after physics step.
        """
        # Update episode length
        self.episode_length += 1

        # Get current state from simulation
        # In a real implementation, this would read from Isaac Sim
        # For now, we'll simulate state updates
        self.base_pos += self.base_lin_vel * self.world.get_physics_dt()
        self.base_quat = integrate_quaternion(self.base_quat, self.base_ang_vel, self.world.get_physics_dt())
        self.base_lin_vel += torch.tensor([0, 0, -9.81], device=self.device) * self.world.get_physics_dt()  # Gravity
        self.dof_pos += self.dof_vel * self.world.get_physics_dt()

        # Calculate rewards
        rewards = self.compute_reward()

        # Update episode rewards
        self.episode_rewards += rewards

        # Determine resets
        resets = self.compute_reset()

        # Get observations
        obs = self.compute_observations()

        # Reset environments that need it
        reset_env_ids = (resets == 1).nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Update last DOF velocities
        self.last_dof_vel = self.dof_vel.clone()

        # Calculate progress
        progress_buf = self.episode_length.clone().float()

        return obs, rewards, resets, progress_buf

    def step(self, actions):
        """
        Execute one simulation step with given actions.
        """
        # Apply actions
        self.pre_physics_step(actions)

        # Step the physics simulation
        self.world.step(render=False)

        # Process simulation results
        obs, rewards, resets, progress = self.post_physics_step()

        return obs, rewards, resets, progress

    def reset(self):
        """
        Reset all environments to initial state.
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs = self.compute_observations()
        return obs

    def close(self):
        """
        Clean up the environment.
        """
        self.world.clear()
        omni.kit.App().get().shutdown()


def torch_rand_float(lower, upper, shape, device):
    """
    Generate random floats in a given range using PyTorch.
    """
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def integrate_quaternion(quat, omega, dt):
    """
    Integrate quaternion with angular velocity.
    """
    # Convert angular velocity to quaternion derivative
    omega_quat = torch.zeros_like(quat)
    omega_quat[:, 0] = 0
    omega_quat[:, 1:] = omega

    # Compute quaternion derivative
    quat_dot = 0.5 * quat_mul(omega_quat, quat)

    # Integrate
    new_quat = quat + quat_dot * dt

    # Normalize
    new_quat = new_quat / torch.norm(new_quat, dim=1, keepdim=True)

    return new_quat


class HumanoidActorCritic(nn.Module):
    """
    Actor-Critic network for humanoid control.
    """

    def __init__(self, obs_dim, action_dim, actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128], init_noise_std=1.0):
        super(HumanoidActorCritic, self).__init__()

        # Initialize standard deviations for actions
        self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))

        # Actor network
        actor_layers = []
        actor_dims = [obs_dim] + actor_hidden_dims + [action_dim]
        for i in range(len(actor_dims) - 1):
            actor_layers.append(nn.Linear(actor_dims[i], actor_dims[i+1]))
            actor_layers.append(nn.ELU())
        self.actor = nn.Sequential(*actor_layers[:-1])  # Remove last activation

        # Critic network
        critic_layers = []
        critic_dims = [obs_dim] + critic_hidden_dims + [1]
        for i in range(len(critic_dims) - 1):
            critic_layers.append(nn.Linear(critic_dims[i], critic_dims[i+1]))
            critic_layers.append(nn.ELU())
        self.critic = nn.Sequential(*critic_layers[:-1])  # Remove last activation

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize network weights.
        """
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)

        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self):
        """
        Forward pass is not implemented as we use separate actor and critic functions.
        """
        raise NotImplementedError

    def act(self, obs, **kwargs):
        """
        Compute actions given observations.
        """
        mean_actions = self.actor(obs)
        std_actions = self.std.repeat((obs.shape[0], 1))
        actions = torch.normal(mean_actions, std_actions)
        return actions

    def get_actions_log_prob(self, obs):
        """
        Get actions and their log probabilities.
        """
        mean_actions = self.actor(obs)
        std_actions = self.std.repeat((obs.shape[0], 1))
        actions = torch.normal(mean_actions, std_actions)
        log_prob = self.get_log_prob(mean_actions, std_actions, actions)
        return actions, log_prob

    def get_log_prob(self, mean, std, actions):
        """
        Calculate log probabilities of actions.
        """
        var = std.pow(2)
        log_prob = -(actions - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * torch.pi * var)
        return log_prob.sum(dim=1)

    def get_value(self, obs):
        """
        Get value estimates for observations.
        """
        return self.critic(obs)

    def evaluate(self, obs, actions):
        """
        Evaluate actions for given observations.
        """
        mean_actions = self.actor(obs)
        std_actions = self.std.repeat((obs.shape[0], 1))
        log_prob = self.get_log_prob(mean_actions, std_actions, actions)
        value = self.critic(obs)
        return log_prob, value


class HumanoidPPO:
    """
    PPO (Proximal Policy Optimization) algorithm for humanoid control.
    """

    def __init__(self, actor_critic, device='cuda:0', clip_param=0.2,
                 num_learning_epochs=1, num_mini_batches=4,
                 value_loss_coef=1.0, entropy_coef=0.0,
                 learning_rate=1e-3, max_grad_norm=1.0):

        self.device = device
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # Initialize actor-critic
        self.actor_critic = actor_critic.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

    def update(self, obs_batch, actions_batch, log_probs_batch,
               values_batch, rewards_batch, dones_batch):
        """
        Update the policy using PPO.
        """
        # Calculate advantages
        advantages = rewards_batch - values_batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate old and new values for clipping
        old_log_probs = log_probs_batch
        old_values = values_batch

        # Perform multiple epochs of PPO updates
        for epoch in range(self.num_learning_epochs):
            # Divide batch into mini-batches
            batch_size = obs_batch.shape[0]
            mini_batch_size = batch_size // self.num_mini_batches

            for i in range(self.num_mini_batches):
                start_idx = i * mini_batch_size
                end_idx = (i + 1) * mini_batch_size

                mb_obs = obs_batch[start_idx:end_idx]
                mb_actions = actions_batch[start_idx:end_idx]
                mb_old_log_probs = old_log_probs[start_idx:end_idx]
                mb_old_values = old_values[start_idx:end_idx]
                mb_advantages = advantages[start_idx:end_idx]

                # Get new values
                mb_log_probs, mb_values = self.actor_critic.evaluate(mb_obs, mb_actions)

                # Calculate ratio
                ratio = torch.exp(mb_log_probs - mb_old_log_probs)

                # Calculate surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                action_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_pred_clipped = mb_old_values + (mb_values - mb_old_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (mb_values - rewards_batch[start_idx:end_idx]).pow(2)
                value_losses_clipped = (value_pred_clipped - rewards_batch[start_idx:end_idx]).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # Total loss
                total_loss = action_loss + self.value_loss_coef * value_loss

                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return total_loss.item()


def train_humanoid_rl():
    """
    Train a humanoid robot using reinforcement learning.
    """
    # Initialize environment
    env = HumanoidRLEnv(
        num_envs=64,
        max_episode_length=1000,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Initialize actor-critic network
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor_critic = HumanoidActorCritic(obs_dim, action_dim)

    # Initialize PPO algorithm
    ppo = HumanoidPPO(actor_critic)

    # Training loop
    num_iterations = 1000
    for iteration in range(num_iterations):
        obs = env.reset()
        total_reward = 0
        total_steps = 0

        # Collect trajectories
        for step in range(env.max_episode_length):
            # Get actions from policy
            with torch.no_grad():
                actions = actor_critic.act(obs)

            # Take step in environment
            next_obs, rewards, dones, info = env.step(actions.cpu().numpy())

            # Update total reward
            total_reward += rewards.mean().item()
            total_steps += len(rewards)

            # Convert to tensors
            obs = torch.from_numpy(next_obs).float().to(actor_critic.device)
            rewards = torch.from_numpy(rewards).float().to(actor_critic.device)
            dones = torch.from_numpy(dones).float().to(actor_critic.device)

            # Break if all environments are done
            if dones.all():
                break

        print(f"Iteration {iteration}, Average Reward: {total_reward/env.max_episode_length:.2f}, Total Steps: {total_steps}")

    # Close environment
    env.close()


def main():
    """
    Main function to run the humanoid RL environment.
    """
    print("Initializing Humanoid RL Environment...")

    # Initialize Isaac Sim
    omni.kit.GlobalStartupParams().initialize()

    try:
        # Train the humanoid
        train_humanoid_rl()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Clean up
        omni.kit.App().get().shutdown()


if __name__ == "__main__":
    main()