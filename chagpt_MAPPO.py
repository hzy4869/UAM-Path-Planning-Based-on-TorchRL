# train_mappo_full.py
"""
Full MAPPO (discrete actions, single-env) implementation.
- Shared actor (parameter sharing) + centralized critic.
- Each agent has its own sampled action; actor outputs logits per agent (same network applied to each agent's obs).
- Compute per-agent returns from each agent's own rewards; centralized V(s) used as baseline.
- Save convergence curves (reward, losses) as PNG and TensorBoard logs.
"""

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# === Replace with your env modules ===
from env_utils.ac_env import ACEnvironment
from env_utils.ac_MultiAgent2 import ACEnvWrapper

# -------------------------
# Hyperparameters (tune as needed)
# -------------------------
SEED = 1
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
UPDATE_EPOCHS = 10
MINI_BATCH_SIZE = 64
ROLLOUT_STEPS = 1024   # timesteps per update (env steps)
ENTROPY_COEF = 1e-3
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
TOTAL_UPDATES = 600    # total parameter-update iterations
CHECKPOINT_DIR = "checkpoints_mappo"
LOG_DIR = "logs_mappo"
PLOT_DIR = "plots_mappo"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Networks
# -------------------------
def mlp(input_dim, hidden_sizes=(256, 256), activation=nn.Tanh):
    layers = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    return nn.Sequential(*layers)

class Actor(nn.Module):
    """
    Shared actor network: given single-agent observation, output logits for discrete actions.
    Parameter-sharing: same network applied to each agent's observation.
    """
    def __init__(self, obs_dim, action_dim, hidden=(256,256)):
        super().__init__()
        self.net = mlp(obs_dim, hidden)
        self.logit_head = nn.Linear(hidden[-1], action_dim)

    def forward(self, x):
        """
        x: [batch, obs_dim]
        returns logits [batch, action_dim]
        """
        h = self.net(x)
        logits = self.logit_head(h)
        return logits

class CentralizedCritic(nn.Module):
    """
    Centralized critic takes concatenated global observations (all agents) and outputs scalar V(s).
    """
    def __init__(self, global_obs_dim, hidden=(256,256)):
        super().__init__()
        self.net = mlp(global_obs_dim, hidden)
        self.v = nn.Linear(hidden[-1], 1)

    def forward(self, x):
        h = self.net(x)
        return self.v(h).squeeze(-1)  # [batch]

# -------------------------
# Rollout Buffer (on-policy)
# -------------------------
class RolloutBuffer:
    def __init__(self, agent_order: List[str], obs_dim:int, rollout_steps:int):
        self.agent_order = agent_order
        self.num_agents = len(agent_order)
        self.obs_dim = obs_dim
        self.rollout_steps = rollout_steps
        self.reset()

    def reset(self):
        self.obs = []            # list of dict agent->obs (np)
        self.actions = []        # list of dict agent->action (int)
        self.logps = []          # list of dict agent->logprob (float)
        self.rewards = []        # list of dict agent->reward (float)
        self.dones = []          # list of dict agent->done (bool)
        self.global_obs = []     # list of global_obs (np)
        self.values = []         # list of centralized V(s) at step

    def add_step(self, obs_dict, global_obs, actions, logps, rewards, dones, value):
        self.obs.append({k: v.copy() for k, v in obs_dict.items()})
        self.global_obs.append(global_obs.copy())
        self.actions.append({k: int(v) for k, v in actions.items()})
        self.logps.append({k: float(v) for k, v in logps.items()})
        self.rewards.append({k: float(v) for k, v in rewards.items()})
        self.dones.append({k: bool(v) for k, v in dones.items()})
        self.values.append(float(value))

    def compute_returns_and_advantages(self, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
        """
        Compute for each agent:
          - returns_agent[t] = discounted sum of agent i rewards from t
          - advantage_agent[t] = returns_agent[t] - V_global(s_t)
        Note: centralized V(s) is same for all agents at timestep t.
        Returns:
          returns: np array [T, num_agents]
          advantages: np array [T, num_agents]
        """
        T = len(self.rewards)
        num_agents = self.num_agents
        # prepare per-agent reward sequences
        rewards_agent = np.zeros((T, num_agents), dtype=np.float32)
        for t in range(T):
            for ia, aid in enumerate(self.agent_order):
                rewards_agent[t, ia] = self.rewards[t].get(aid, 0.0)

        # compute discounted returns per agent
        returns = np.zeros((T, num_agents), dtype=np.float32)
        for ia in range(num_agents):
            ret = last_value  # bootstrap using centralized last value (this is simple choice)
            for t in reversed(range(T)):
                if self.dones[t].get(self.agent_order[ia], False):
                    ret = 0.0  # episode ended for that agent -> bootstrap 0
                ret = rewards_agent[t, ia] + gamma * ret
                returns[t, ia] = ret

        # advantages = returns - V(s_t)
        values = np.array(self.values, dtype=np.float32)  # [T]
        # expand values to [T, num_agents]
        values_expand = np.repeat(values[:, None], num_agents, axis=1)
        advantages = returns - values_expand
        return returns, advantages

# -------------------------
# Utils
# -------------------------
def flatten_obs_dict(obs_dict: Dict[str, np.ndarray], agent_order: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for aid in agent_order:
        out[aid] = np.asarray(obs_dict[aid], dtype=np.float32).reshape(-1)
    return out

def build_global_obs(flat_obs_dict: Dict[str, np.ndarray], agent_order: List[str]) -> np.ndarray:
    parts = [flat_obs_dict[aid] for aid in agent_order]
    return np.concatenate(parts, axis=0)

# -------------------------
# Training Loop (MAPPO)
# -------------------------
def train_mappo(env,
                total_updates=TOTAL_UPDATES,
                rollout_steps=ROLLOUT_STEPS,
                update_epochs=UPDATE_EPOCHS,
                minibatch_size=MINI_BATCH_SIZE,
                save_every=50):
    # infer agent/order dims
    obs = env.reset()
    agent_order = sorted(list(obs.keys()))
    num_agents = len(agent_order)
    flat_sample = {aid: np.asarray(obs[aid], dtype=np.float32).reshape(-1) for aid in agent_order}
    obs_dim = flat_sample[agent_order[0]].shape[0]
    global_obs_dim = obs_dim * num_agents
    action_dim = env.action_space.n

    print(f"MAPPO: agents={agent_order}, num_agents={num_agents}, obs_dim={obs_dim}, global_obs_dim={global_obs_dim}, action_dim={action_dim}")

    # models & optimizers
    actor = Actor(obs_dim, action_dim).to(DEVICE)
    critic = CentralizedCritic(global_obs_dim).to(DEVICE)
    actor_optim = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optim = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # optionally LR schedulers
    actor_scheduler = optim.lr_scheduler.StepLR(actor_optim, step_size=200, gamma=0.9)
    critic_scheduler = optim.lr_scheduler.StepLR(critic_optim, step_size=200, gamma=0.9)

    buffer = RolloutBuffer(agent_order, obs_dim, rollout_steps)

    # logging
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    rewards_history = []   # per-episode average reward across agents
    actor_loss_history = []
    critic_loss_history = []
    entropy_history = []

    global_step = 0
    update_start_time = time.time()
    for update in range(total_updates):
        buffer.reset()
        steps = 0
        # collect rollouts for fixed number of env steps
        obs = env.reset()
        episode_rewards_accum = {aid: 0.0 for aid in agent_order}
        episode_lengths = {aid: 0 for aid in agent_order}
        episode_count = 0

        while steps < rollout_steps:
            flat_obs = flatten_obs_dict(obs, agent_order)
            global_obs = build_global_obs(flat_obs, agent_order)

            # select actions and log probs per agent
            actions = {}
            logps = {}
            entropies = []
            for aid in agent_order:
                ob = torch.from_numpy(flat_obs[aid]).float().to(DEVICE).unsqueeze(0)  # [1, obs_dim]
                logits = actor(ob)  # [1, action_dim]
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                actions[aid] = int(a.item())
                logps[aid] = float(dist.log_prob(a).item())
                entropies.append(float(dist.entropy().item()))

            # centralized value
            with torch.no_grad():
                gobs_t = torch.from_numpy(global_obs).float().to(DEVICE).unsqueeze(0)
                value = critic(gobs_t).item()  # scalar

            # step env
            next_obs, rewards, truncated, dones, infos = env.step(actions)

            # record
            buffer.add_step(flat_obs, global_obs, actions, logps, rewards, {aid: dones.get(aid, False) for aid in agent_order}, value)

            # accumulate episode rewards for logging
            for aid in agent_order:
                episode_rewards_accum[aid] += rewards.get(aid, 0.0)
                episode_lengths[aid] += 1

            # detect finished episodes (any agent done counts as an episode end for logging convenience)
            if any(dones.values()):
                # compute average reward per-agent for this episode
                avg_ep_reward = np.mean([episode_rewards_accum[aid] for aid in agent_order])
                rewards_history.append(avg_ep_reward)
                writer.add_scalar("Episode/AvgReward", avg_ep_reward, global_step)
                # reset accumulators
                episode_rewards_accum = {aid: 0.0 for aid in agent_order}
                episode_lengths = {aid: 0 for aid in agent_order}
                episode_count += 1

            obs = next_obs
            steps += 1
            global_step += 1

        # bootstrap
        flat_last = flatten_obs_dict(obs, agent_order)
        last_global = build_global_obs(flat_last, agent_order)
        with torch.no_grad():
            last_value = critic(torch.from_numpy(last_global).float().to(DEVICE).unsqueeze(0)).item()

        # prepare returns and advantages
        returns, advantages = buffer.compute_returns_and_advantages(last_value)
        # returns: [T, num_agents], advantages: [T, num_agents]

        # flatten for training: each (t, agent) becomes one sample
        T = len(buffer.rewards)
        nbatch = T * num_agents
        obs_batch = np.zeros((nbatch, obs_dim), dtype=np.float32)
        actions_batch = np.zeros((nbatch,), dtype=np.int64)
        old_logprobs_batch = np.zeros((nbatch,), dtype=np.float32)
        returns_batch = np.zeros((nbatch,), dtype=np.float32)
        adv_batch = np.zeros((nbatch,), dtype=np.float32)
        global_obs_batch = np.zeros((nbatch, global_obs_dim), dtype=np.float32)

        idx = 0
        for t in range(T):
            gobs_t = buffer.global_obs[t]
            for ia, aid in enumerate(agent_order):
                obs_batch[idx] = buffer.obs[t][aid].reshape(-1)
                actions_batch[idx] = buffer.actions[t][aid]
                old_logprobs_batch[idx] = buffer.logps[t][aid]
                returns_batch[idx] = returns[t, ia]
                adv_batch[idx] = advantages[t, ia]
                global_obs_batch[idx] = gobs_t
                idx += 1

        # normalize advantages
        adv_mean = adv_batch.mean()
        adv_std = adv_batch.std() + 1e-8
        adv_batch = (adv_batch - adv_mean) / adv_std

        # convert to torch
        obs_tensor = torch.from_numpy(obs_batch).float().to(DEVICE)
        actions_tensor = torch.from_numpy(actions_batch).long().to(DEVICE)
        old_logp_tensor = torch.from_numpy(old_logprobs_batch).float().to(DEVICE)
        returns_tensor = torch.from_numpy(returns_batch).float().to(DEVICE)
        adv_tensor = torch.from_numpy(adv_batch).float().to(DEVICE)
        global_obs_tensor = torch.from_numpy(global_obs_batch).float().to(DEVICE)

        # training epochs
        batch_inds = np.arange(nbatch)
        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_entropy = 0.0
        updates_this_iter = 0

        for epoch in range(update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, nbatch, minibatch_size):
                mb_inds = batch_inds[start:start + minibatch_size]
                mb_obs = obs_tensor[mb_inds]
                mb_actions = actions_tensor[mb_inds]
                mb_oldlog = old_logp_tensor[mb_inds]
                mb_adv = adv_tensor[mb_inds]
                mb_ret = returns_tensor[mb_inds]
                mb_gobs = global_obs_tensor[mb_inds]

                # actor forward
                logits = actor(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                newlogp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(newlogp - mb_oldlog)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

                # critic forward
                values_pred = critic(mb_gobs)
                critic_loss = VALUE_COEF * ((mb_ret - values_pred) ** 2).mean()

                # optimize actor
                actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                actor_optim.step()

                # optimize critic
                critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                critic_optim.step()

                epoch_actor_loss += float(actor_loss.item())
                epoch_critic_loss += float(critic_loss.item())
                epoch_entropy += float(entropy.item())
                updates_this_iter += 1

        # average metrics over optimization minibatches
        if updates_this_iter > 0:
            epoch_actor_loss /= updates_this_iter
            epoch_critic_loss /= updates_this_iter
            epoch_entropy /= updates_this_iter

        actor_loss_history.append(epoch_actor_loss)
        critic_loss_history.append(epoch_critic_loss)
        entropy_history.append(epoch_entropy)

        # schedulers step
        actor_scheduler.step()
        critic_scheduler.step()

        # TensorBoard logging
        step_for_log = update
        writer.add_scalar("Train/ActorLoss", epoch_actor_loss, step_for_log)
        writer.add_scalar("Train/CriticLoss", epoch_critic_loss, step_for_log)
        writer.add_scalar("Train/Entropy", epoch_entropy, step_for_log)
        if len(rewards_history) > 0:
            writer.add_scalar("Train/AvgEpisodeReward", rewards_history[-1], step_for_log)

        # Checkpointing
        if (update + 1) % save_every == 0 or update == total_updates - 1:
            ckpt = {
                "actor_state": actor.state_dict(),
                "critic_state": critic.state_dict(),
                "actor_optim": actor_optim.state_dict(),
                "critic_optim": critic_optim.state_dict(),
                "update": update,
            }
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"mappo_ckpt_{update+1}.pth"))
            print(f"[Update {update+1}] Saved checkpoint.")

        # periodic print
        if update % 5 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) > 0 else 0.0
            elapsed = time.time() - update_start_time
            print(f"[Update {update}/{total_updates}] global_steps={global_step} "
                  f"avg_recent_reward(50)={avg_reward:.3f} actor_loss={epoch_actor_loss:.4f} critic_loss={epoch_critic_loss:.4f} entropy={epoch_entropy:.4f} elapsed={elapsed:.1f}s")

    # end training
    writer.close()

    # Save final models
    torch.save(actor.state_dict(), os.path.join(CHECKPOINT_DIR, "actor_final.pth"))
    torch.save(critic.state_dict(), os.path.join(CHECKPOINT_DIR, "critic_final.pth"))
    print("Training complete. Models saved.")

    # Plot convergence curves
    plot_and_save(rewards_history, actor_loss_history, critic_loss_history, entropy_history)

    return actor, critic, {
        "rewards_history": rewards_history,
        "actor_loss": actor_loss_history,
        "critic_loss": critic_loss_history,
        "entropy": entropy_history
    }

# -------------------------
# Plotting utility
# -------------------------
def plot_and_save(rewards, actor_losses, critic_losses, entropies, window=50):
    os.makedirs(PLOT_DIR, exist_ok=True)
    # rewards and moving average
    plt.figure(figsize=(10,6))
    plt.plot(rewards, label="Episode Avg Reward")
    if len(rewards) >= 1:
        ma = moving_average(rewards, window)
        plt.plot(ma, label=f"MA({window})")
    plt.xlabel("Episode count")
    plt.ylabel("Avg reward per episode (across agents)")
    plt.legend()
    plt.title("Training Reward Curve")
    rfile = os.path.join(PLOT_DIR, "reward_curve.png")
    plt.savefig(rfile)
    plt.close()
    print(f"Saved reward curve to {rfile}")

    # losses
    plt.figure(figsize=(10,6))
    plt.plot(actor_losses, label="Actor Loss")
    plt.plot(critic_losses, label="Critic Loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Actor / Critic Loss")
    lfile = os.path.join(PLOT_DIR, "loss_curve.png")
    plt.savefig(lfile)
    plt.close()
    print(f"Saved loss curve to {lfile}")

    # entropy
    plt.figure(figsize=(10,6))
    plt.plot(entropies, label="Entropy")
    plt.xlabel("Update")
    plt.ylabel("Entropy")
    plt.legend()
    plt.title("Policy Entropy")
    efile = os.path.join(PLOT_DIR, "entropy_curve.png")
    plt.savefig(efile)
    plt.close()
    print(f"Saved entropy curve to {efile}")

def moving_average(x, w):
    if len(x) < 1:
        return []
    w = min(w, len(x))
    return np.convolve(x, np.ones(w), 'valid') / w

# -------------------------
# Entrypoint / Example usage
# -------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    drone_img_path = os.path.join(base_dir, "assets", "drone.png")
    sumo_cfg = "./sumo_envs/zeyun_UAM/two_car_stable.sumocfg"

    def custom_update_cover_radius(position, communication_range):
        height = position[2]
        return height / np.tan(np.radians(75/2))

    aircraft_inits = {
        'drone_1': {
            "aircraft_type": "drone",
            "action_type": "horizontal_movement",
            "position": (100, 200, 50),
            "speed": 10,
            "heading": (1, 1, 0),
            "communication_range": 50,
            "if_sumo_visualization": False,
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
            "if_sumo_visualization": False,
            "img_file": drone_img_path,
            "custom_update_cover_radius": custom_update_cover_radius
        }
    }

    env = ACEnvironment(sumo_cfg=sumo_cfg, tls_ids=["1","2"], num_seconds=500, aircraft_inits=aircraft_inits, use_gui=False)
    env = ACEnvWrapper(env, aircraft_inits)

    train_mappo(env, total_updates=TOTAL_UPDATES, rollout_steps=ROLLOUT_STEPS, update_epochs=UPDATE_EPOCHS)

if __name__ == "__main__":
    main()
