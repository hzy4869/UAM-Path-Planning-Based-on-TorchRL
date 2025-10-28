import os
import numpy as np
import gymnasium as gym
from gymnasium.core import Env
from torchrl.envs import GymWrapper, EnvBase
from collections import defaultdict, deque
from typing import Dict, Tuple
import math

import torch
from torchrl.data import Composite, Unbounded, DiscreteTensorSpec, StackedComposite, Categorical
from tensordict import TensorDict
from torchrl.data.tensor_specs import CompositeSpec, BoundedTensorSpec


os.environ['OMP_NUM_THREADS'] = '1'

class ACEnvWrapper(GymWrapper):
    """Simplified Aircraft Environment Wrapper with relative_vecs"""
    def __init__(self, env: Env, aircraft_inits, max_states: int = 3, max_rel_veh: int = 5) -> None:
        super().__init__(env)
        self.max_states = max_states
        self.max_rel_veh = max_rel_veh
        self.aircraft_inits = aircraft_inits
        self.agents = list(aircraft_inits.keys())
        self.speed = aircraft_inits["drone_1"]["speed"]


        self.n_agents = 3
        # self.batch_size = [3, ]

        self._pos_set = defaultdict(lambda: deque(maxlen=self.max_states))
        self.latest_ac_pos = {}
        self.latest_veh_pos = {}
        self.veh_covered = {}
        self.x_range, self.y_range = None, None

        # 动作：8方向 + 暂停
        speed = self.speed
        self.air_actions = {
            0: (speed, 0),  # → 右
            1: (speed, 1),  # ↗ 右上
            2: (speed, 2),  # ↑ 正上
            3: (speed, 3),  # ↖ 左上
            4: (speed, 4),  # ← 左
            5: (speed, 5),  # ↙ 左下
            6: (speed, 6),  # ↓ 正下
            7: (speed, 7),  # ↘ 右下
            8: (0, 0),      # 暂停
        }

    # def _make_done_spec(self):
    #     return Composite(
    #         {
    #             "done": Categorical(
    #                 2, dtype=torch.bool, device=self.device, shape=(3, 1)
    #             ),
    #             "terminated": Categorical(
    #                 2, dtype=torch.bool, device=self.device, shape=(3, 1)
    #             ),
    #             "truncated": Categorical(
    #                 2, dtype=torch.bool, device=self.device, shape=(3, 1)
    #             ),
    #         },
    #         shape=(3, ),
    #     ) 

    def _make_specs(self, env, batch_size=[3]):
        # 调用父类方法生成基础 spec
        # print("here")
        super()._make_specs(env, batch_size=batch_size)
        print("obs_spec")
        print(self.observation_spec)
        # print("done_spec")
        # print(self.done_spec)


    def prune_old_vehicles(self, current_veh_ids):
        '''删掉消失的车辆'''
        self.latest_veh_pos = {vid: pos for vid, pos in self.latest_veh_pos.items() if vid in current_veh_ids}

    def state_wrapper(self, state):
        feature_dict = {}
        vehicles = state['vehicle']
        self.prune_old_vehicles(set(vehicles.keys()))

        # 更新无人机与车辆最新位置
        for ac_id, ac_info in state['aircraft'].items():
            if ac_info['aircraft_type'] != 'drone':
                continue

            ac_pos = np.array(ac_info['position'][:2], dtype=np.float32) ## type = [x y]
            self.latest_ac_pos[ac_id] = ac_pos
            self._pos_set[ac_id].append(ac_pos)

            # trajectory: [x1 y1 x2 y2 x3(current) y3(current)]
            hist = list(self._pos_set[ac_id])
            if len(hist) < self.max_states:
                pad = [[0.0, 0.0]] * (self.max_states - len(hist))
                hist = pad + hist

            # relative_vecs: 所有车辆位置，固定长度 max_rel_veh
            # type: [[x y][x y]...[x y]*40]
            rel_vecs = []
            for vid, veh_info in vehicles.items():
                veh_pos = np.array(veh_info['position'], dtype=np.float32)
                rel_vecs.append(veh_pos - ac_pos)  # 相对无人机位置
                self.latest_veh_pos[vid] = veh_pos # 保存车辆*绝对位置*
                self.veh_covered[vid] = False

            rel_vecs = np.array(rel_vecs[:self.max_rel_veh], dtype=np.float32)
            num_veh = rel_vecs.shape[0] if rel_vecs.size > 0 else 0
            if num_veh < self.max_rel_veh:
                pad = np.zeros((self.max_rel_veh - num_veh, 2), dtype=np.float32)
                if num_veh > 0:
                    rel_vecs = np.vstack((rel_vecs, pad))
                else:
                    rel_vecs = pad  # 直接全部填充

            # 队友相对位置
            teammate_vecs = np.zeros((1,2), dtype=np.float32)
            for mate_id, mate_pos in self.latest_ac_pos.items():
                if mate_id != ac_id:
                    teammate_vecs[0] = mate_pos[:2] - ac_pos[:2]

            feature_dict[ac_id] = {
                "ac_attr": np.array(hist, dtype=np.float32).reshape(-1),
                "relative_vecs": rel_vecs,
                "teammate_pos": teammate_vecs.flatten()
            }

        return feature_dict

    def feature_set_to_drone_obs(self, feature_dict):
        '''归一化--非常重要！'''
        obs_dict = {}
        for ac_id, feats in feature_dict.items():
            ac_attr = (feats['ac_attr'] - np.mean(feats['ac_attr'])) / (np.std(feats['ac_attr']) + 1e-8)
            relative_vecs = (feats['relative_vecs'] - np.mean(feats['relative_vecs'])) / (np.std(feats['relative_vecs']) + 1e-8)
            teammate_pos = (feats['teammate_pos'] - np.mean(feats['teammate_pos'])) / (np.std(feats['teammate_pos']) + 1e-8)
            obs_dict[ac_id] = np.concatenate([ac_attr, relative_vecs.flatten()])
        return obs_dict

    def reward_wrapper(self, dones) -> Tuple[Dict[str, float], Dict[str, bool]]:
        '''
        10/27
        '''
        rewards = {}
        done_dict = {aid: bool(dones) for aid in self.agents}  # 初始化 done_dict
                    

        for ac_id in self.agents:
            # print("------------------------")
            total_reward = 0
            _x, _y = self.latest_ac_pos.get(ac_id, [0, 0])[:2]


            # 计算与所有车辆的距离，找出最近的车辆
            min_distance = float('inf')
            min_veh_id = None
            for vid, veh_pos in self.latest_veh_pos.items():
                distance = np.linalg.norm(np.array([_x, _y]) - np.array(veh_pos[:2]))
                if distance < min_distance:
                    min_distance = distance
                    min_veh_id = vid
                
            
            # cover检测
            if self.veh_covered[min_veh_id] == True:
                # print("检测到冲突！")
                min_distance = float('inf')
                min_veh_id = None
                for vid, veh_pos in self.latest_veh_pos.items():
                    if self.veh_covered[vid] == False:
                        distance = np.linalg.norm(np.array([_x, _y]) - np.array(veh_pos[:2]))
                        if distance < min_distance:
                            min_distance = distance
                            min_veh_id = vid
                
                if min_distance <= 150:
                    total_reward += 10
                else:
                    # 每增加 100 距离，奖励减少 2
                    intervals = math.floor((min_distance - 50) / 50)
                    reward = max(0, 10 - 1 * intervals)
                    total_reward += reward
                # pass
            else:
                # 基于最近车辆的距离计算奖励-->基于最近的未被覆盖的车辆计算奖励
                if min_distance <= 50:
                    self.veh_covered[min_veh_id] = True
                    total_reward += 15
                    
                elif min_distance <= 150:
                    total_reward += 10
                else:
                    # 每增加 100 距离，奖励减少 2
                    intervals = math.floor((min_distance - 50) / 50)
                    reward = max(0, 10 - 1 * intervals)
                    total_reward += reward

            # 针对个体边界的惩罚
            if abs(_x) > self.x_range or abs(_y) > self.y_range:
                total_reward += -20
                done_dict[ac_id] = True  # 仅为超出边界的无人机设置 done=True
            elif abs(_x) > (self.x_range - 50) or abs(_y) > (self.y_range - 50):
                total_reward += -5

            rewards[ac_id] = total_reward

        # print(rewards)
        return rewards, done_dict

    def reset(self, seed=1, tensordict=None, **kwargs):
        ''''''
        print("reset")
        self.x_range, self.y_range = 800, 800
        raw_state = self._env.reset()
        feature_set = self.state_wrapper(raw_state)
        obs = self.feature_set_to_drone_obs(feature_set)

        agent_ids = list(obs.keys())
        obs_tensor = torch.stack([
            torch.tensor(obs[aid], dtype=torch.float32)
            for aid in agent_ids
        ])  # shape: [n_agents, obs_dim]


        # 构建 TorchRL 规范的 TensorDict 结构
        obs_td = TensorDict({
            "agents": TensorDict({
                "observation": obs_tensor,
                # "episode_reward": torch.tensor([0.0 for _ in agent_ids], dtype= torch.float32).unsqueeze(-1)
            }, batch_size=[3,]), 
            "done": torch.tensor([False], dtype=bool),
            "terminated": torch.tensor([False], dtype=bool),
            "truncated": torch.tensor([False], dtype=bool),
            # "next": next
        }, batch_size=[])
        return obs_td


    def step(self, action: TensorDict):

        action_tensor = action["agents"]["action"]  # shape: [n_agents, n_actions]
        agent_ids = list(self.agents)  # ['drone_1', 'drone_2', 'drone_3']
        action_dict = {aid: int(act.item()) for aid, act in zip(agent_ids, action_tensor)}
        new_actions = {aid: self.air_actions[act] for aid, act in action_dict.items()}

        states, rewards, truncated, dones, infos = self._env.step(new_actions)

        # 2️⃣ 生成 observation
        feature_set = self.state_wrapper(states)
        rewards, dones = self.reward_wrapper(dones)
        obs_dict = self.feature_set_to_drone_obs(feature_set)
        obs_tensor = torch.stack([torch.tensor(obs_dict[aid], dtype=torch.float32) for aid in agent_ids])

        # 3️⃣ reward / done 直接整合成 Tensor
        
        ## 可能错误原因：done其实是一个标量而非list，也就是说环境只有一个done！
        aid_temp = agent_ids[0]
        ## 暂时保留第一个reward,后续需要加上sum
        reward_tensor = torch.tensor([rewards[aid] for aid in agent_ids], dtype=torch.float32).unsqueeze(-1)
        done_tensor = torch.tensor([dones[aid_temp]], dtype=bool)
        terminated_tensor = torch.tensor([False for aid in agent_ids], dtype=bool)
        truncated_tensor = torch.tensor([False for aid in agent_ids], dtype=bool)
        # print(reward_tensor.shape)
        # print(done_tensor.shape)

        obs_td = TensorDict({
            "agents": TensorDict({
                "observation": obs_tensor,
                "reward": reward_tensor,
                # "episode_reward": torch.tensor([5.0 for _ in agent_ids], dtype= torch.float32).unsqueeze(-1),
            }, batch_size=[3,]),
            "done": done_tensor, ## 成功传入 
            "terminated": done_tensor, ## 成功传入 
            "truncated": done_tensor, ## 成功传入 
        }, batch_size=[])

        # 4️⃣ 构建最终返回 TensorDict
        out_td = TensorDict({
            "next": obs_td,
        }, batch_size=[])
        # print(out_td)
        # print(reward_tensor)
        return out_td


    def close(self):
        # return super().close()
        return self._env.close()
    

# TensorDict(
#     fields={
#         agents: TensorDict(
#             fields={
#                 info: TensorDict(
#                     fields={
#                         ground_rew: Tensor(shape=torch.Size([10, 3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
#                         pos_rew: Tensor(shape=torch.Size([10, 3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
#                     batch_size=torch.Size([10, 3]),
#                     device=cpu,
#                     is_shared=False),
#                 observation: Tensor(shape=torch.Size([10, 3, 16]), device=cpu, dtype=torch.float32, is_shared=False),  
#                 reward: Tensor(shape=torch.Size([10, 3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},       
#             batch_size=torch.Size([10, 3]),
#             device=cpu,
#             is_shared=False),
#         done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
#         terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
#     batch_size=torch.Size([10]),
#     device=cpu,
#     is_shared=False)