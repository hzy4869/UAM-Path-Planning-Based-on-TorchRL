import os
import numpy as np
import gymnasium as gym
from gymnasium.core import Env
from collections import defaultdict, deque
from typing import Dict, Any, Tuple, SupportsFloat
import time
import math

os.environ['OMP_NUM_THREADS'] = '1'

class ACEnvWrapper(gym.Wrapper):
    """Simplified Aircraft Environment Wrapper with relative_vecs"""
    def __init__(self, env: Env, aircraft_inits, max_states: int = 3, max_rel_veh: int = 5) -> None:
        super().__init__(env)
        self.max_states = max_states
        self.max_rel_veh = max_rel_veh
        self.aircraft_inits = aircraft_inits
        self.agents = list(aircraft_inits.keys())
        self.speed = aircraft_inits["drone_1"]["speed"]

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

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.air_actions))

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "ac_attr": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_states*3,), dtype=np.float32),
            "relative_vecs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_rel_veh, 2), dtype=np.float32),
            # "teammate_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
    
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
        2025/9/28: 一个无人机靠近车辆后（<50），会导致另一个无人机无法获取奖励（在同样也是靠近一个车辆
        的情况下，但>50），暂时不影响收敛
        '''
        rewards = {}
        done_dict = {aid: bool(dones) for aid in self.agents}  # 初始化 done_dict
        
        # 车辆是否被cover初始化
        print(self.latest_veh_pos)
        print("veh_cover is here", self.veh_covered)
                    

        for ac_id in self.agents:
            # print("------------------------")
            total_reward = 0
            _x, _y = self.latest_ac_pos.get(ac_id, [0, 0])[:2]


            # 计算与所有车辆的距离，找出最近的车辆
            min_distance = float('inf')
            min_veh_id = None
            for vid, veh_pos in self.latest_veh_pos.items():
                distance = np.linalg.norm(np.array([_x, _y]) - np.array(veh_pos[:2]))
                # print("for vid = ", vid)
                # print("ac pos: ", np.array([_x, _y]))
                # print("car pos:", np.array(veh_pos[:2]))
                # min_distance = min(min_distance, distance)
                # min_veh_id = vid
                if distance < min_distance:
                    min_distance = distance
                    min_veh_id = vid
            # print("最近的车辆：", min_veh_id)
            # print("距离为：", min_distance)
                
            
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

        print(rewards)
        return rewards, done_dict

    def reset(self, seed=1):
        self.x_range, self.y_range = 800, 800
        raw_state = self.env.reset()
        feature_set = self.state_wrapper(raw_state)
        obs = self.feature_set_to_drone_obs(feature_set)
        return obs

    def step(self, action: Dict[str,int]):
        # 转换动作
        new_actions = {aid: self.air_actions[int(act)] for aid, act in action.items()}
        states, rewards, truncated, dones, infos = super().step(new_actions)
        feature_set = self.state_wrapper(states)
        # print("after arrange the states, now state to reward func:")
        # print(feature_set)
        rewards, dones = self.reward_wrapper(dones)
        obs_dict = self.feature_set_to_drone_obs(feature_set)
        # time.sleep(0.1)
        
        return obs_dict, rewards, truncated, dones, infos

    def close(self):
        return super().close()
