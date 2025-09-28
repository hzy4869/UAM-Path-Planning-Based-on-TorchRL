import os
import numpy as np
import gymnasium as gym
from typing import List, Dict, Any
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import matplotlib.pyplot as plt

from env_utils.ac_env import ACEnvironment
from env_utils.ac_MultiAgent2 import ACEnvWrapper


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

class Critic(nn.Module):
    def __init__(self, global_obs_dim: int, hidden_dim: int = 128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(global_obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(global_obs))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class MAPPO:
    def __init__(self, obs_dim: int, act_dim: int, num_agents: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.num_agents = num_agents
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim * num_agents).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5

    def get_action(self, obs: Dict[str, np.ndarray], sample: bool = True) -> Dict[str, int]:
        actions = {}
        with torch.no_grad():
            for agent_id, agent_obs in obs.items():
                obs_tensor = torch.FloatTensor(agent_obs).to(self.device)
                logits = self.actor(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample() if sample else dist.probs.argmax()
                actions[agent_id] = action.item()
        return actions

    def compute_advantages(self, rewards: List[Dict[str, float]], values: List[torch.Tensor], dones: List[Dict[str, bool]], next_value: torch.Tensor) -> List[torch.Tensor]:
        advantages = []
        returns = []
        gae = torch.zeros(self.num_agents).to(self.device)
        for t in reversed(range(len(rewards))):
            delta = torch.tensor([rewards[t][agent_id] for agent_id in sorted(rewards[t].keys())]).to(self.device) + \
                    self.gamma * next_value * (1 - torch.tensor([dones[t][agent_id] for agent_id in sorted(dones[t].keys())]).to(self.device)) - values[t]
            gae = delta + self.gamma * self.lam * (1 - torch.tensor([dones[t][agent_id] for agent_id in sorted(dones[t].keys())]).to(self.device)) * gae
            advantages.insert(0, gae.clone())
            returns.insert(0, gae + values[t])
        advantages = torch.stack(advantages)
        returns = torch.stack(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, trajectory: Dict[str, List]):
        obs = trajectory["obs"]
        actions = trajectory["actions"]
        log_probs_old = trajectory["log_probs"]
        rewards = trajectory["rewards"]
        dones = trajectory["dones"]

        values = []
        for t in range(len(obs)):
            global_obs = torch.FloatTensor(np.concatenate([obs[t][aid] for aid in sorted(obs[t].keys())])).to(self.device)
            value = self.critic(global_obs)
            values.append(value.repeat(self.num_agents))

        global_obs = torch.FloatTensor(np.concatenate([obs[-1][aid] for aid in sorted(obs[-1].keys())])).to(self.device)
        next_value = self.critic(global_obs)

        advantages, returns = self.compute_advantages(rewards, values, dones, next_value)

        obs_flat = torch.FloatTensor(np.array([obs[t][aid] for t in range(len(obs)) for aid in sorted(obs[t].keys())])).to(self.device)
        actions_flat = torch.LongTensor([actions[t][aid] for t in range(len(actions)) for aid in sorted(actions[t].keys())]).to(self.device)
        log_probs_old_flat = torch.FloatTensor([log_probs_old[t][aid] for t in range(len(log_probs_old)) for aid in sorted(log_probs_old[t].keys())]).to(self.device)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)

        logits = self.actor(obs_flat)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_flat)
        ratio = torch.exp(log_probs - log_probs_old_flat)
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_flat
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy = dist.entropy().mean()
        loss = actor_loss - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        global_obs_flat = torch.FloatTensor(np.array([np.concatenate([obs[t][aid] for aid in sorted(obs[t].keys())]) for t in range(len(obs))])).to(self.device)
        values = self.critic(global_obs_flat).squeeze()
        value_loss = F.mse_loss(values, returns_flat)

        self.critic_optimizer.zero_grad()
        (self.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return actor_loss.item(), value_loss.item()

def plot_convergence_curves(episode_rewards: List[float], actor_losses: List[float], critic_losses: List[float], save_dir: str):
    """Plot and save convergence curves for episode rewards, actor loss, and critic loss."""
    plt.figure(figsize=(15, 5))

    # Plot episode rewards
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Episode Reward vs Episode")
    plt.legend()
    plt.grid(True)

    # Plot actor loss
    plt.subplot(1, 3, 2)
    plt.plot(actor_losses, label="Actor Loss", color="orange")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Actor Loss vs Update Step")
    plt.legend()
    plt.grid(True)

    # Plot critic loss
    plt.subplot(1, 3, 3)
    plt.plot(critic_losses, label="Critic Loss", color="green")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Critic Loss vs Update Step")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "convergence_curves.png"))
    plt.close()

def train_mappo():
    # ---------------- Environment Setup ----------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    drone_img_path = os.path.join(base_dir, "assets", "drone.png")
    sumo_cfg = "./sumo_envs/zeyun_UAM/two_car_stable.sumocfg"

    def custom_update_cover_radius(position: List[float], communication_range: float) -> float:
        """Custom method to update ground cover radius."""
        height = position[2]
        cover_radius = height / np.tan(math.radians(75/2))
        return cover_radius

    aircraft_inits = {
        'drone_1': {
            "aircraft_type": "drone",
            "action_type": "horizontal_movement",
            "position": (100, 200, 50),
            "speed": 10,
            "heading": (1, 1, 0),
            "communication_range": 50,
            "if_sumo_visualization": True,
            "img_file": drone_img_path,
            "custom_update_cover_radius": custom_update_cover_radius
        },
        'drone_2': {
            "aircraft_type": "drone",
            "action_type": "horizontal_movement",
            "position": (-100, 200, 50),
            "speed": 10,
            "heading": (1, 0, 0),
            "communication_range": 50,
            "if_sumo_visualization": True,
            "img_file": drone_img_path,
            "custom_update_cover_radius": custom_update_cover_radius
        },
    }

    env = ACEnvironment(
        sumo_cfg=sumo_cfg,
        tls_ids=["1", "2"],
        num_seconds=500,
        aircraft_inits=aircraft_inits,
        use_gui=False
    )
    env = ACEnvWrapper(env, aircraft_inits)

    # Infer observation and action dimensions
    obs = env.reset()
    num_agents = len(obs)
    obs_dim = len(obs[list(obs.keys())[0]])
    act_dim = env.action_space.n

    # Initialize MAPPO
    mappo = MAPPO(obs_dim, act_dim, num_agents)

    # Training parameters
    num_episodes = 1000
    max_steps = 500
    batch_size = 128
    trajectory = defaultdict(list)
    episode_rewards = []
    actor_losses = []
    critic_losses = []

    # Training loop
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        trajectory.clear()

        for step in range(max_steps):
            actions = mappo.get_action(obs)
            log_probs = {}
            for agent_id, agent_obs in obs.items():
                obs_tensor = torch.FloatTensor(agent_obs).to(mappo.device)
                logits = mappo.actor(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs[agent_id] = dist.log_prob(torch.tensor(actions[agent_id])).item()

            next_obs, rewards, truncated, dones, _ = env.step(actions)

            trajectory["obs"].append(obs)
            trajectory["actions"].append(actions)
            trajectory["log_probs"].append(log_probs)
            trajectory["rewards"].append(rewards)
            trajectory["dones"].append(dones)

            episode_reward += sum(rewards.values()) / num_agents
            obs = next_obs

            if len(trajectory["obs"]) >= batch_size:
                actor_loss, value_loss = mappo.update(trajectory)
                actor_losses.append(actor_loss)
                critic_losses.append(value_loss)
                trajectory.clear()
                print(f"Episode {episode}, Step {step}, Actor Loss: {actor_loss:.4f}, Value Loss: {value_loss:.4f}")

            if any(dones.values()):
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode}, Total Reward: {episode_reward:.2f}")

    # Plot convergence curves
    plot_convergence_curves(episode_rewards, actor_losses, critic_losses, base_dir)

    env.close()

if __name__ == "__main__":
    train_mappo()