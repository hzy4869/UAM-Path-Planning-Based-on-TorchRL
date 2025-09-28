'''
@Author: Hu Zeyun
@Date: 2025/9/22
@Description: 处理 ACEnvironment
+多智能体环境
'''
import os

from tensorflow.python.ops.special_math_ops import bessel_y0

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import gymnasium as gym
import math
from gymnasium.core import Env
from typing import Any, SupportsFloat, Tuple, Dict
from typing import List
from collections import defaultdict, deque, Counter

from numpy import floating
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

class ACEnvWrapper(gym.Wrapper):
    """Aircraft Env Wrapper for single junction with tls_id
    """
    def __init__(self, env: Env, aircraft_inits, max_states: int = 3) -> None:
        super().__init__(env)
        self.max_states = max_states
        self._pos_set = defaultdict(lambda: deque(maxlen=self.max_states))
        self.speed = aircraft_inits["drone_1"]["speed"]
        self.aircraft_inits = aircraft_inits
        self.agents = list(aircraft_inits.keys())
        self.x_range, self.y_range = None, None
        self.initial_points = {
            ac_id: ac_value["position"] for ac_id, ac_value in aircraft_inits.items()
        }
        self.break_spot = np.array([0, 0])
        self.latest_ac_pos = {}
        self.latest_veh_pos = {}
        self.latest_cover_radius = {}
        self.veh_trajectories = defaultdict(list)
        self.ac_trajectories = defaultdict(list)
        self.cluster_point = []
        self.total_covered = 0
        self.cover_efficiency = None
        speed = self.speed
        self.air_actions = {
            0: (speed, 0),  # -> 右
            1: (speed, 1),  # ↗ 右上
            2: (speed, 2),  # ↑ 正上
            3: (speed, 3),  # ↖ 左上
            4: (speed, 4),  # ← 左
            5: (speed, 5),  # ↙ 左下
            6: (speed, 6),  # ↓ 正下
            7: (speed, 7),  # ↘ 右下
            8: (0, 0),  # 暂停
            9: (0, 2),  # 暂停
            10: (0, 4),  # 暂停
            11: (0, 6),  # 暂停
        }
        with open("reward_log.txt", "w", encoding="utf-8") as f:
            f.write("")  # 写空内容清空

    def flatten_single_obs(self, single_obs):
        """
        输入：single_obs = raw_state['aircraft'][aid]
        输出：numpy array
        """
        pos = np.array(single_obs['position'], dtype=np.float32)
        speed = np.array([single_obs['speed']], dtype=np.float32)
        heading = np.array(single_obs['heading'], dtype=np.float32)
        comm_range = np.array([single_obs['communication_range']], dtype=np.float32)
        
        # 组合成一维向量
        obs_vec = np.concatenate([pos, speed, heading, comm_range])
        return obs_vec


    # def  get_relative_ac_pos(self, aircraft_id, pos) -> List:
    #     _init_points = self.initial_points[aircraft_id]
    #     pos_new = [pos[0] - _init_points[0], pos[1] - _init_points[1], pos[2] - _init_points[2]]
    #     return pos_new

    # def  get_relative_pos(self, aircraft_id, pos) -> List:
    #     _init_points = self.initial_points[aircraft_id]
    #     pos_new = [pos[0] - _init_points[0], pos[1] - _init_points[1]]
    #     return pos_new

    # def compute_cover_centers(self, veh_pos, radius):
    #     """
    #     返回每个无人机对应的最佳覆盖中心，最多 len(self.agents) 个中心
    #     """
    #     if len(veh_pos) == 0:
    #         return []

    #     tree = KDTree(veh_pos)
    #     max_centers = len(self.agents)
    #     centers = []

    #     remaining_idx = np.arange(len(veh_pos))

    #     for _ in range(max_centers):
    #         if len(remaining_idx) == 0:
    #             break
    #         counts = tree.query_radius(veh_pos[remaining_idx], r=radius, count_only=True)
    #         best_idx = remaining_idx[np.argmax(counts)]
    #         best_center = veh_pos[best_idx].copy()
    #         centers.append(best_center)

    #         # 删除半径内车辆避免重复
    #         neigh_idx = tree.query_radius([best_center], r=radius)[0]
    #         remaining_idx = np.setdiff1d(remaining_idx, neigh_idx)

    #     return centers
    

    # def density_peaks_clustering(self, veh_pos, radius, n_centers=1):
    #     N = len(veh_pos)
    #     dist = np.linalg.norm(veh_pos[:, None, :] - veh_pos[None, :, :], axis=2)
    #     rho = np.sum(dist < radius, axis=1) - 1
    #     delta = np.zeros(N)
    #     for i in range(N):
    #         higher = np.where(rho > rho[i])[0]
    #         if higher.size > 0:
    #             delta[i] = np.min(dist[i, higher])
    #         else:
    #             delta[i] = np.max(dist[i])
    #     gamma = rho * delta
    #     centers = np.argsort(-gamma)[:n_centers]

    #     best_center_idx = centers[0]
    #     cluster_size = rho[best_center_idx]+1
    #     if cluster_size > 3:
    #         center_point = veh_pos[centers]
    #         center_point = center_point.reshape(-1)
    #     else: return None
    #     return center_point

    # @staticmethod
    # def distance_penalty(dist, cover_radius, p=10):
    #     ratio = dist / cover_radius
    #     penalty = -np.log1p(ratio) / np.log1p(p)
    #     return penalty

    def break_spot_reward(self, dist, cover_radius):
        spot_dist = np.linalg.norm(dist)
        if spot_dist <= cover_radius:
            return 4 # 3.5 4 2
        else:
            penalty = self.distance_penalty(spot_dist - cover_radius, cover_radius, p=25)
            return penalty #* 0.45

    def prune_old_vehicles(self, current_veh_ids):
        # 删掉消失的车辆
        self.latest_veh_pos = {vid: pos for vid, pos in self.latest_veh_pos.items() if vid in current_veh_ids}

    @property
    def action_space(self):
        return gym.spaces.Discrete(12)

    @property
    def observation_space(self):
        spaces = {
            "ac_attr": gym.spaces.Box(low=np.zeros((9,)), high=np.ones((9,)), shape=(9,)), # (1,9)
            "relative_vecs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(40,2), dtype=np.float32), # (1,20,2)
            "cover_counts": gym.spaces.Box(low=0, high=np.inf, shape=(1,)), # (1,1)
            "break_spot": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)), # (1,2)
        }
        dict_space = gym.spaces.Dict(spaces)
        return dict_space

    def _get_initial_state(self) -> List[int]:
        return [0, 0, 0]  # x, y, z

    # Wrapper
    def state_wrapper(self, state):
        """
        max count??? may should be changed.
        多智能体状态处理，每架无人机只看到覆盖范围内车辆
        返回:
            feature_dict: {aircraft_id: feature_set}  每架无人机的特征向量
            new_state: {aircraft_id: vehicle_state}  每架无人机覆盖到的车辆状态
        """
        new_state = dict()
        feature_dict = dict()
        
        veh = state['vehicle']
        aircraft = state['aircraft']
        self.prune_old_vehicles(set(veh.keys()))

        for aircraft_id, aircraft_info in aircraft.items():
            if aircraft_info['aircraft_type'] != 'drone':
                continue

            cover_radius = aircraft_info['cover_radius']
            aircraft_pos = aircraft_info['position']
            # ac_pos = self.get_relative_ac_pos(aircraft_id, aircraft_pos)
            ac_pos = aircraft_pos

            # 更新无人机状态与轨迹
            self.latest_cover_radius[aircraft_id] = cover_radius
            self.latest_ac_pos[aircraft_id] = ac_pos
            self._pos_set[aircraft_id].append(ac_pos)
            self.ac_trajectories[aircraft_id].append(ac_pos)

            hist = list(self._pos_set[aircraft_id])
            # 如果数量不足，前面补零
            if len(hist) < self.max_states:
                pad = [[0.0, 0.0, 0.0]] * (self.max_states - len(hist))
                hist = pad + hist

            vehicle_state = {}
            relative_vecs = []

            # 遍历所有车辆，保留覆盖范围内的
            for vehicle_id, vehicle_info in veh.items():
                veh_pos = vehicle_info['position']
                dx = veh_pos[0] - ac_pos[0]
                dy = veh_pos[1] - ac_pos[1]
                dist = math.hypot(dx, dy)

                self.veh_trajectories[vehicle_id].append(veh_pos)
                self.latest_veh_pos[vehicle_id] = veh_pos

                # if dist <= cover_radius:
                    # vehicle_state[vehicle_id] = vehicle_info.copy()
                relative_vecs.append(veh_pos)  # 只加入覆盖范围内车辆->所有车辆都加入

            # 更新每架无人机的覆盖车辆数量
            # cover_count = len(vehicle_state)
            new_state[aircraft_id] = vehicle_state

            # 处理 relative_vecs，固定长度 40，不足补零
            if relative_vecs:
                relative_vecs = np.array(relative_vecs[:2])
                if relative_vecs.shape[0] < 2:
                    pad = np.zeros((2 - relative_vecs.shape[0], 2))
                    relative_vecs = np.vstack((relative_vecs, pad))
            else:
                relative_vecs = np.zeros((2, 2))

            # === 队友相对位置 ===
            teammate_vecs = [[0.0, 0.0]]
            for mate_id, mate_pos in self.latest_ac_pos.items():
                if mate_id == aircraft_id:
                    continue
                dx = mate_pos[0] - ac_pos[0]
                dy = mate_pos[1] - ac_pos[1]
                # dx = mate_pos[0]
                # dy = mate_pos[1]
                teammate_vecs[0] = ([dx, dy])
            print("****队友位置：")
            print(teammate_vecs)

            # 构建无人机特征向量
            feature_dict[aircraft_id] = {
                "ac_attr": np.array(hist, dtype=np.float32).reshape(-1),
                "relative_vecs": relative_vecs,           # 覆盖范围内车辆的相对位置
                # "cover_count": cover_count,               # 覆盖车辆数量
                # "break_spot": np.array(self.break_spot).reshape(-1),  # 休息点
                # "cover_efficiency": cover_count / max_count if max_count > 0 else 0
                "teammate_pos": np.array(teammate_vecs).reshape(-1)
            }
        # print("call state:")
        # print(feature_dict)

        return feature_dict, new_state
    

    def reward_wrapper(self, states, dones) -> tuple:
        """奖励：两无人机靠近两辆车"""
        total_reward = 0
        log_lines = []

        if not hasattr(self, 'agents'):
            self.agents = list(states.keys())  # 初始化无人机列表

        # 如果环境里至少有两辆车
        if self.latest_veh_pos and len(self.latest_veh_pos) >= 2:
            veh_positions = np.array(list(self.latest_veh_pos.values()))[:2]  # 只取两辆目标车
            ac_positions = [np.array(self.latest_ac_pos[aid][:2]) for aid in self.agents]
            print("车辆位置：")
            print(veh_positions)
            print("无人机位置：")
            print(ac_positions)

            # ========== 分配匹配 (Hungarian assignment) ==========
            from scipy.optimize import linear_sum_assignment
            cost_matrix = np.zeros((len(ac_positions), len(veh_positions)))
            for i, ac_pos in enumerate(ac_positions):
                for j, v_pos in enumerate(veh_positions):
                    cost_matrix[i, j] = np.linalg.norm(ac_pos - v_pos)  # 距离越小越好

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            match_reward = 0
            for i, j in zip(row_ind, col_ind):
                dist = cost_matrix[i, j]
                if dist < 50:
                    r = 10  # 固定奖励
                    match_reward += r
                    log_lines.append(f"[Proximity Reward] {self.agents[i]} → Car{j}: dist={dist:.2f}, +{r}")
                else:
                    log_lines.append(f"[Proximity Reward] {self.agents[i]} → Car{j}: dist={dist:.2f}, +0")

            total_reward += match_reward

        else:
            log_lines.append("[No Vehicles] No proximity reward")

        # ===== 边界惩罚 =====
        for ac_id in self.agents:
            _x, _y = np.array(self.latest_ac_pos[ac_id][:2])
            if abs(_y) > (self.y_range - 50):
                total_reward += -5
                log_lines.append(f"[Boundary Penalty] {ac_id}: -5 (near Y boundary)")
            if abs(_x) > (self.x_range - 50):
                total_reward += -5
                log_lines.append(f"[Boundary Penalty] {ac_id}: -5 (near X boundary)")

            if abs(_x) > self.x_range or abs(_y) > self.y_range:
                total_reward += -100
                log_lines.append(f"[Boundary Exceed] {ac_id}: -100")
                self._write_log(log_lines)
                rewards = {aid: total_reward for aid in self.agents}
                done_dict = {aid: bool(dones) for aid in self.agents}
                return rewards, done_dict

        # 写入日志
        self._write_log(log_lines)
        rewards = {aid: total_reward for aid in self.agents}
        done_dict = {aid: bool(dones) for aid in self.agents}

        return rewards, done_dict



    def _write_log(self, log_lines):
        """把日志写入 reward_log.txt"""
        with open("reward_log.txt", "a", encoding="utf-8") as f:
            for line in log_lines:
                f.write(line + "\n")
    


    def feature_set_to_drone_obs(self, feature_dict):
        """
        将每架无人机的 feature_dict flatten，生成多智能体 obs dict。
        
        Args:
            feature_dict (dict): {aircraft_id: feature_set} 
                feature_set 包含 'ac_attr', 'relative_vecs', 'cover_count', 'break_spot', 'cover_efficiency'
                
        Returns:
            obs_dict (dict): key=drone_x, value=一维 numpy array (flatten 后)
        """
        obs_dict = {}
        
        for i, (ac_id, feats) in enumerate(feature_dict.items()):
            ac_attr = np.array(feats['ac_attr']).flatten()
            relative_vecs = np.array(feats['relative_vecs']).flatten()
            temmate_pos = np.array(feats['teammate_pos']).flatten()
            
            # flatten 并拼接成一维向量
            full_vec = np.concatenate([ac_attr, relative_vecs, temmate_pos])
            
            obs_dict[f'drone_{i+1}'] = full_vec

        return obs_dict



    def reset(self, seed=1):
        self.x_range = 800
        self.y_range = 800
        raw_state = self.env.reset()

        state, _ = self.state_wrapper(state=raw_state)
        obs = self.feature_set_to_drone_obs(state)
        return obs



    def step(self, action: Dict[str, int]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        new_actions = {}
        for aid, act in action.items():
            if isinstance(act, (int, np.int64)):
                new_actions[aid] = self.air_actions[int(act)]
            elif isinstance(act, tuple):
                new_actions[aid] = act
            else:
                raise TypeError(f"Unrecognized action type: {act} (type {type(act)})")

        states, rewards, truncated, dones, infos = super().step(new_actions) # 与环境交互
        feature_set, veh_states = self.state_wrapper(state=states) # 处理 state
        rewards, dones = self.reward_wrapper(states=veh_states,dones=dones) # 处理 reward

        # 打印无人机信息
        for drone_id in veh_states.keys():  # 遍历无人机ID
            ac_pos = self.latest_ac_pos[drone_id]
            print(f"[Step] {drone_id} Position: {ac_pos}, Reward: {rewards}")

        obs_dict = self.feature_set_to_drone_obs(feature_set)
        # print("----------feature set----------")
        # print(feature_set)
        # print("---------obs flatten---------")
        # print(obs_dict)
        
        return obs_dict, rewards, truncated, dones, infos
    
    def close(self) -> None:
        return super().close()
