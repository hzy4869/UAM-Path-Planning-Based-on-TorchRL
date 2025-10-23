import torch
import numpy as np
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
from tensordict import TensorDict
from collections import defaultdict, deque
from typing import Dict, Any, Tuple
import math
import os

os.environ['OMP_NUM_THREADS'] = '1'

class ACEnvWrapper(EnvBase):
    """TorchRL-compatible Aircraft Environment Wrapper for MAPPO/IPPO"""
    def __init__(self, env, aircraft_inits, max_states: int = 3, max_rel_veh: int = 5, device="cpu"):
        super().__init__(device=device)
        self.env = env
        self.aircraft_inits = aircraft_inits
        self.max_states = max_states
        self.max_rel_veh = max_rel_veh
        self.x_range, self.y_range = 800, 800

        self._pos_set = defaultdict(lambda: deque(maxlen=self.max_states))
        self.latest_ac_pos = {}
        self.latest_veh_pos = {}
        self.veh_covered = {}

        self.air_actions = {
            0: (self.speed, 0),  # → right
            1: (self.speed, 1),  # ↗ right-up
            2: (self.speed, 2),  # ↑ up
            3: (self.speed, 3),  # ↖ left-up
            4: (self.speed, 4),  # ← left
            5: (self.speed, 5),  # ↙ left-down
            6: (self.speed, 6),  # ↓ down
            7: (self.speed, 7),  # ↘ right-down
            8: (0, 0),          # pause
        }

    @property
    def n_agents(self) -> int:
        """Number of agents in the environment."""
        return len(self.aircraft_inits)

    @property
    def agents(self) -> list:
        """List of agent IDs."""
        return list(self.aircraft_inits.keys())

    @property
    def speed(self) -> float:
        """Speed of the first drone."""
        if "drone_1" not in self.aircraft_inits:
            raise ValueError("aircraft_inits must contain 'drone_1' with a 'speed' key")
        return self.aircraft_inits["drone_1"]["speed"]

    @property
    def full_action_spec_unbatched(self) -> CompositeSpec:
        """
        Returns the full action spec for all agents, unbatched.
        Each agent has a Discrete action space with n=len(self.air_actions) choices.
        """
        # 单 agent 动作 spec (标量)
        action_spec = DiscreteTensorSpec(
            n=len(self.air_actions),
            shape=(),  # 单 agent
            device=self.device
        )

        # 外层 Composite 表示 n_agents 个 agent
        return CompositeSpec(
            {"agents": {"action": action_spec}},
            shape=(self.n_agents,)  # 表示有 n_agents 个 agent
        )



    
    @property
    def observation_spec(self) -> CompositeSpec:
        """Observation specification for the environment (flat style for tuple indexing)."""
        return CompositeSpec({
            ("agents", "observation"): CompositeSpec({
                "ac_attr": BoundedTensorSpec(
                    low=-float('inf'), high=float('inf'),
                    shape=(self.n_agents, self.max_states * 2),
                    dtype=torch.float32, device=self.device
                ),
                "relative_vecs": BoundedTensorSpec(
                    low=-float('inf'), high=float('inf'),
                    shape=(self.n_agents, self.max_rel_veh * 2),
                    dtype=torch.float32, device=self.device
                )
            }, shape=(self.n_agents,))
        }, shape=())


    @property
    def reward_spec(self) -> CompositeSpec:
        """Reward specification for the environment."""
        return CompositeSpec({
            "agents": {
                "reward": BoundedTensorSpec(
                    low=-float('inf'), high=float('inf'),
                    shape=(self.n_agents,), dtype=torch.float32, device=self.device
                )
            }
        }, shape=())

    @property
    def done_spec(self) -> CompositeSpec:
        """Done specification for the environment."""
        return CompositeSpec({
            "done": BoundedTensorSpec(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
            "terminated": BoundedTensorSpec(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
            "agents": {
                "done": BoundedTensorSpec(low=0, high=1, shape=(self.n_agents,), dtype=torch.bool, device=self.device)
            }
        }, shape=())


    def _set_seed(self, seed: int) -> int:
        """Set the random seed for the environment."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            self.env.seed(seed)
        except AttributeError:
            pass
        self.env.reset()
        return seed

    def prune_old_vehicles(self, current_veh_ids):
        """Remove disappeared vehicles from tracking"""
        self.latest_veh_pos = {vid: pos for vid, pos in self.latest_veh_pos.items() if vid in current_veh_ids}

    def state_wrapper(self, state):
        """Process raw state into feature dictionary for each agent"""
        print(f"Raw state: {state}")  # Debug: Check raw state structure
        feature_dict = {}
        vehicles = state.get('vehicle', {})
        self.prune_old_vehicles(set(vehicles.keys()))

        for ac_id in self.agents:  # Iterate over expected agents
            ac_info = state.get('aircraft', {}).get(ac_id, None)
            if ac_info is None or ac_info.get('aircraft_type') != 'drone':
                print(f"Warning: No valid data for agent {ac_id}, using default values")
                hist_tensor = torch.zeros(self.max_states * 2, dtype=torch.float32, device=self.device)
                rel_vecs = torch.zeros((self.max_rel_veh, 2), dtype=torch.float32, device=self.device)
                teammate_vecs = torch.zeros(2, dtype=torch.float32, device=self.device)
            else:
                ac_pos = torch.tensor(ac_info['position'][:2], dtype=torch.float32, device=self.device)
                self.latest_ac_pos[ac_id] = ac_pos
                self._pos_set[ac_id].append(ac_pos.cpu().numpy())

                hist = list(self._pos_set[ac_id])
                if len(hist) < self.max_states:
                    pad = [[0.0, 0.0]] * (self.max_states - len(hist))
                    hist = pad + hist
                hist_tensor = torch.tensor(hist, dtype=torch.float32, device=self.device).reshape(-1)

                rel_vecs = []
                for vid, veh_info in vehicles.items():
                    veh_pos = torch.tensor(veh_info['position'], dtype=torch.float32, device=self.device)
                    rel_vecs.append(veh_pos[:2] - ac_pos)
                    self.latest_veh_pos[vid] = veh_pos
                    self.veh_covered[vid] = False

                rel_vecs = torch.stack(rel_vecs[:self.max_rel_veh]) if rel_vecs else torch.zeros((self.max_rel_veh, 2), device=self.device)
                if rel_vecs.shape[0] < self.max_rel_veh:
                    pad = torch.zeros((self.max_rel_veh - rel_vecs.shape[0], 2), dtype=torch.float32, device=self.device)
                    rel_vecs = torch.cat((rel_vecs, pad), dim=0)

                teammate_vecs = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
                for mate_id, mate_pos in self.latest_ac_pos.items():
                    if mate_id != ac_id:
                        teammate_vecs[0] = mate_pos[:2] - ac_pos[:2]

            feature_dict[ac_id] = {
                "ac_attr": hist_tensor,
                "relative_vecs": rel_vecs,
                # "teammate_pos": teammate_vecs.flatten()
            }

        print(f"Feature dict: {feature_dict}")  # Debug: Check feature dict
        return feature_dict
    
    def feature_set_to_drone_obs(self, feature_dict):
        """Normalize features and convert to tensor observations"""
        obs_dict = {}
        for ac_id, feats in feature_dict.items():
            ac_attr = (feats['ac_attr'] - feats['ac_attr'].mean()) / (feats['ac_attr'].std() + 1e-8)
            relative_vecs = (feats['relative_vecs'] - feats['relative_vecs'].mean()) / (feats['relative_vecs'].std() + 1e-8)
            # teammate_pos = (feats['teammate_pos'] - feats['teammate_pos'].mean()) / (feats['teammate_pos'].std() + 1e-8)
            obs_dict[ac_id] = torch.cat([ac_attr, relative_vecs.flatten()])
        return obs_dict

    def reward_wrapper(self, dones):
        """Calculate rewards and done signals for each agent"""
        rewards = torch.zeros(self.n_agents, device=self.device)
        done_dict = torch.tensor([bool(dones) for _ in self.agents], dtype=torch.bool, device=self.device)

        for idx, ac_id in enumerate(self.agents):
            total_reward = 0.0
            _x, _y = self.latest_ac_pos.get(ac_id, torch.tensor([0.0, 0.0], device=self.device))[:2]

            min_distance = float('inf')
            min_veh_id = None
            for vid, veh_pos in self.latest_veh_pos.items():
                distance = torch.linalg.norm(torch.tensor([_x, _y], device=self.device) - veh_pos[:2])
                if distance < min_distance:
                    min_distance = distance
                    min_veh_id = vid

            if min_veh_id is not None and self.veh_covered.get(min_veh_id, False):
                min_distance = float('inf')
                min_veh_id = None
                for vid, veh_pos in self.latest_veh_pos.items():
                    if not self.veh_covered.get(vid, False):
                        distance = torch.linalg.norm(torch.tensor([_x, _y], device=self.device) - veh_pos[:2])
                        if distance < min_distance:
                            min_distance = distance
                            min_veh_id = vid
                if min_veh_id is not None:
                    if min_distance <= 150:
                        total_reward += 10.0
                    else:
                        intervals = math.floor((min_distance - 50) / 50)
                        reward = max(0, 10 - 1 * intervals)
                        total_reward += reward
            elif min_veh_id is not None:
                if min_distance <= 50:
                    self.veh_covered[min_veh_id] = True
                    total_reward += 15.0
                elif min_distance <= 150:
                    total_reward += 10.0
                else:
                    intervals = math.floor((min_distance - 50) / 50)
                    reward = max(0, 10 - 1 * intervals)
                    total_reward += reward

            if abs(_x) > self.x_range or abs(_y) > self.y_range:
                total_reward -= 20.0
                done_dict[idx] = True
            elif abs(_x) > (self.x_range - 50) or abs(_y) > (self.y_range - 50):
                total_reward -= 5.0

            rewards[idx] = total_reward

        return rewards, done_dict

    def _reset(self, tensordict=None, **kwargs):
        """Reset environment and return initial observations"""
        print(f"Resetting ACEnvWrapper on device: {self.device}")
        self.x_range, self.y_range = 800, 800
        raw_state = self.env.reset()
        feature_set = self.state_wrapper(raw_state)
        if not feature_set:
            raise RuntimeError("feature_set is empty, check ACEnvironment.reset or state_wrapper")
        obs_dict = self.feature_set_to_drone_obs(feature_set)
        if not obs_dict:
            raise RuntimeError("obs_dict is empty, check state_wrapper or feature_set_to_drone_obs")

        # Initialize TensorDict for single environment
        tensordict_out = TensorDict(
            {
                "agents": {
                    "observation": {
                        "ac_attr": torch.stack([
                            obs_dict[ac_id][:self.max_states * 2] for ac_id in self.agents
                        ], dim=0),
                        "relative_vecs": torch.stack([
                            obs_dict[ac_id][self.max_states * 2:]
                            for ac_id in self.agents
                        ], dim=0)
                        # "teammate_pos": torch.stack([
                        #     obs_dict[ac_id][-2:] for ac_id in self.agents
                        # ], dim=0)
                    }
                }
            },
            batch_size=[],
            device=self.device
        )
        print(f"Reset output: {tensordict_out}")
        return tensordict_out

    def _step(self, tensordict):
        """Step the environment with actions and return next state, rewards, and dones"""
        print(f"Step input: {tensordict}")
        action_dict = {aid: self.air_actions[int(act)] for aid, act in zip(self.agents, tensordict["agents"]["action"][0])}
        states, _, truncated, dones, infos = self.env.step(action_dict)
        feature_set = self.state_wrapper(states)
        if not feature_set:
            raise RuntimeError("feature_set is empty, check ACEnvironment.step or state_wrapper")
        rewards, agent_dones = self.reward_wrapper(dones)
        obs_dict = self.feature_set_to_drone_obs(feature_set)
        if not obs_dict:
            raise RuntimeError("obs_dict is empty, check state_wrapper or feature_set_to_drone_obs")

        # Initialize TensorDict for single environment
        tensordict_out = TensorDict(
            {
                "agents": {
                    "observation": {
                        "ac_attr": torch.stack([
                            obs_dict[ac_id][:self.max_states * 2] for ac_id in self.agents
                        ], dim=0),
                        "relative_vecs": torch.stack([
                            obs_dict[ac_id][self.max_states * 2:]
                            for ac_id in self.agents
                        ], dim=0),
                        # "teammate_pos": torch.stack([
                        #     obs_dict[ac_id][-2:] for ac_id in self.agents
                        # ], dim=0)
                    },
                    "reward": rewards,
                    "done": agent_dones
                },
                "done": torch.tensor([agent_dones.any()], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([truncated], dtype=torch.bool, device=self.device)
            },
            batch_size=[],
            device=self.device
        )
        print(f"Step output: {tensordict_out}")
        return tensordict_out

    def close(self):
        """Close the underlying environment"""
        return self.env.close()