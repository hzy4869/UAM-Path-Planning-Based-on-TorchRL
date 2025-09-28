    # def reward_wrapper(self, states, dones) -> tuple:
    #     """多中心共享奖励"""
    #     total_reward = 0
    #     log_lines = []

    #     if not hasattr(self, 'agents'):
    #         self.agents = list(states.keys())  # 初始化无人机列表

    #     if self.latest_veh_pos:
    #         veh_positions = np.array(list(self.latest_veh_pos.values()))
    #         centers = self.compute_cover_centers(veh_positions, radius=max(self.latest_cover_radius.values()))
    #         print(centers)
    #     else:
    #         centers = []

    #     for ac_id, vehicle_info in states.items():
    #         ac_pos = np.array(self.latest_ac_pos[ac_id][:2])
    #         cover_radius = self.latest_cover_radius[ac_id]

    #         if centers:
    #             # 选离无人机最近的中心
    #             distances = [np.linalg.norm(center - ac_pos) for center in centers]
    #             nearest_center = centers[np.argmin(distances)]
    #             # 计算覆盖车辆
    #             covered = [v for v in self.latest_veh_pos.values()
    #                     if np.linalg.norm(np.array(v) - nearest_center) <= cover_radius]
    #             reward = len(covered)
    #             total_reward += reward
    #             log_lines.append(f"[Reward] {ac_id}: +{reward} (covered {len(vehicle_info)} vehicles near nearest center)")
    #         else:
    #             # 无车团情况--不给奖励
    #             reward = self.break_spot_reward(ac_pos, cover_radius)
    #             total_reward += 0
    #             log_lines.append(f"[No Vehicle/Cluster] {ac_id}: +{0}")

    #         # ===== 边界惩罚 =====
    #         _x, _y = ac_pos
    #         if abs(_y) > (self.y_range - 50):
    #             total_reward += -5
    #             log_lines.append(f"[Boundary Penalty] {ac_id}: -5 (near Y boundary)")
    #         if abs(_x) > (self.x_range - 50):
    #             total_reward += -5
    #             log_lines.append(f"[Boundary Penalty] {ac_id}: -5 (near X boundary)")

    #         if abs(_x) > self.x_range or abs(_y) > self.y_range:
    #             # dones = True
    #             total_reward += -100
    #             log_lines.append(f"[Boundary Exceed] {ac_id}: -100")
    #             self._write_log(log_lines)
    #             rewards = {aid: total_reward for aid in self.agents}
    #             done_dict = {aid: bool(dones) for aid in self.agents}
    #             return rewards, done_dict

    #     # 写入日志
    #     self._write_log(log_lines)
    #     # 共享奖励
    #     rewards = {aid: total_reward for aid in self.agents}
    #     done_dict = {aid: bool(dones) for aid in self.agents}

    #     return rewards, done_dict


    # def reward_wrapper(self, states, dones) -> tuple:
    #     """奖励：两无人机靠近两辆车"""
    #     total_reward = 0
    #     log_lines = []

    #     if not hasattr(self, 'agents'):
    #         self.agents = list(states.keys())  # 初始化无人机列表

    #     # 如果环境里至少有两辆车
    #     if self.latest_veh_pos and len(self.latest_veh_pos) >= 2:
    #         veh_positions = np.array(list(self.latest_veh_pos.values()))[:2]  # 只取两辆目标车
    #         ac_positions = [np.array(self.latest_ac_pos[aid][:2]) for aid in self.agents]

    #         # ========== 分配匹配 (Hungarian assignment) ==========
    #         from scipy.optimize import linear_sum_assignment
    #         cost_matrix = np.zeros((len(ac_positions), len(veh_positions)))
    #         for i, ac_pos in enumerate(ac_positions):
    #             for j, v_pos in enumerate(veh_positions):
    #                 cost_matrix[i, j] = np.linalg.norm(ac_pos - v_pos)  # 距离越小越好

    #         row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #         match_reward = 0
    #         for i, j in zip(row_ind, col_ind):
    #             dist = cost_matrix[i, j]
    #             r = max(0, 50 - dist)  # 距离越近奖励越大（最多50）
    #             match_reward += r
    #             log_lines.append(f"[Proximity Reward] {self.agents[i]} → Car{j}: dist={dist:.2f}, +{r:.2f}")

    #         total_reward += match_reward
    #     else:
    #         log_lines.append("[No Vehicles] No proximity reward")

    #     # ===== 边界惩罚 =====
    #     for ac_id in self.agents:
    #         _x, _y = np.array(self.latest_ac_pos[ac_id][:2])
    #         if abs(_y) > (self.y_range - 50):
    #             total_reward += -5
    #             log_lines.append(f"[Boundary Penalty] {ac_id}: -5 (near Y boundary)")
    #         if abs(_x) > (self.x_range - 50):
    #             total_reward += -5
    #             log_lines.append(f"[Boundary Penalty] {ac_id}: -5 (near X boundary)")

    #         if abs(_x) > self.x_range or abs(_y) > self.y_range:
    #             total_reward += -100
    #             log_lines.append(f"[Boundary Exceed] {ac_id}: -100")
    #             self._write_log(log_lines)
    #             rewards = {aid: total_reward for aid in self.agents}
    #             done_dict = {aid: bool(dones) for aid in self.agents}
    #             return rewards, done_dict

    #     # 写入日志
    #     self._write_log(log_lines)
    #     rewards = {aid: total_reward for aid in self.agents}
    #     done_dict = {aid: bool(dones) for aid in self.agents}

    #     return rewards, done_dict