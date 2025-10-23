'''
@Author: WANG Maonan
@Date: 2023-09-14 13:48:19
@Description: Aircraft Environment
@LastEditTime: 2023-09-14 16:02:21
'''
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Box, MultiDiscrete
from typing import Dict
from typing import Any, List
from tshub.tshub_env.tshub_env import TshubEnvironment

class ACEnvironment(gym.Env):
    def __init__(self, sumo_cfg: str, tls_ids: List[str], num_seconds: int, aircraft_inits: Dict[str, Any], use_gui: bool = False) -> None:
        super().__init__()

        self.tsc_env = TshubEnvironment(
            sumo_cfg=sumo_cfg,
            is_aircraft_builder_initialized=True, 
            is_vehicle_builder_initialized=True,
            is_traffic_light_builder_initialized=False,
            aircraft_inits=aircraft_inits, 
            num_seconds=num_seconds,
            use_gui=use_gui,
            is_libsumo=(not use_gui), # 如果不开界面, 就是用 libsumo
        )
        self.tls_ids = tls_ids
        self.aircraft_inits = aircraft_inits
        self.n_agents = len(aircraft_inits)


        self.action_space = GymDict({
            "agents": GymDict({
                "action": MultiDiscrete([9] * self.n_agents)
            })
        })

        n_features = 16
        self.observation_space = GymDict({
            "agents": GymDict({
                "observation": Box(low=-np.inf, high=np.inf, shape=(self.n_agents, n_features), dtype=np.float32),
                ## 这里的done需要二次修改，last point of trajectory，in torchrl/value/functional.py
                "done": Box(low=0, high=1, shape=(self.n_agents, 1), dtype=bool),
                "reward": Box(low=-np.inf, high=np.inf, shape=(self.n_agents, 1), dtype=np.float32),
                "episode_reward": Box(low=-np.inf, high=np.inf, shape=(self.n_agents, 1), dtype=np.float32)
            })
        })

    def reset(self):
        state_infos = self.tsc_env.reset()
        return state_infos
        
    def step(self, action: Dict[str, Dict[str, int]]):
        action = {'aircraft': action} # 这里只控制 aircraft 即可
        states, rewards, infos, dones = self.tsc_env.step(action)
        truncated = dones

        return states, rewards, truncated, dones, infos
    
    def close(self) -> None:
        self.tsc_env._close_simulation()