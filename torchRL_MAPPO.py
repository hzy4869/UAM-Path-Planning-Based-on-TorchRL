import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "rl\\sota-implementations\\multiagent"))
sys.path.append(os.path.join(os.path.dirname(__file__), "rl"))
from mappo_ippo import train
from omegaconf import OmegaConf
cfg = OmegaConf.load("rl\\sota-implementations\\multiagent\\mappo_ippo.yaml")

from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from loguru import logger
import argparse
import torch
# from torchrl.envs import TransformedEnv, RewardSum

from env_utils.ac_env import ACEnvironment
from env_utils.ac_MultiAgent_test import ACEnvWrapper
from env_utils.make_multi_env import make_parallel_env


path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), file_log_level="ERROR", terminal_log_level="ERROR")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters.')
    parser.add_argument('--env_name', type=str, default="zeyun_UAM", help='The name of environment, detroit_UAM, berlin_UAM')
    parser.add_argument('--passenger_len', type=int, default=5, help='The number of passengers')
    parser.add_argument('--passenger_type', type=str, default="real_time", help='fix or real time')
    parser.add_argument('--snir_min', type=int, default=-17, help='The threshold of SNIR') # 最小SNIR值，小于这个值的乘客不参与训练
    parser.add_argument('--num_envs', type=int, default=1, help='The number of environments')
    parser.add_argument('--n_steps', type=int, default=2000, help='The number of steps in each environment')
    parser.add_argument('--policy_model', type=str, default="fusion_models_v8", help='policy network: baseline_models or fusion_models_4')
    parser.add_argument('--features_dim', type=int, default=2048, help='The dimension of output features 64')
    parser.add_argument('--lr', type=float, default=5e-5, help='The learning rate of PPO')
    parser.add_argument('--batch_size', type=int, default=400, help='The batch size of PPO')
    parser.add_argument('--num_seconds', type=int, default=2500, help='exploration steps')
    parser.add_argument('--cuda_id', type=int, default=0, help='The id of cuda device')
    args = parser.parse_args()  # Parse the arguments
    device = f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu'

    # #########
    # Init
    # #########
    sumo_cfg = path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}.sumocfg")
    net_file = path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}.net.xml")
    snir_files = {
        '100': path_convert(f"./sumo_envs/{args.env_name}/{args.env_name}_SNIR_100.txt") # 高度： 文件路径
        # xxx, 这里可以添加不同高度的文件
    }

    aircraft_inits = {
        'drone_1': {
            "aircraft_type": "drone",
            "action_type": "horizontal_movement",
            "position": (100, 200, 50),
            "speed": 10,
            "heading": (1, 1, 0),
            "communication_range": 50,
            "if_sumo_visualization": True,
            "img_file": path_convert('./assets/drone.png'),
        },
        'drone_2': {
            "aircraft_type": "drone",
            "action_type": "horizontal_movement",
            "position": (300, -100, 50),
            "speed": 10,
            "heading": (1, 0, 0),
            "communication_range": 50,
            "if_sumo_visualization": True,
            "img_file": path_convert('./assets/drone.png'),
        },

        'drone_3': {
            "aircraft_type": "drone",
            "action_type": "horizontal_movement",
            "position": (-100, -100, 50),
            "speed": 10,
            "heading": (1, 0, 0),
            "communication_range": 50,
            "if_sumo_visualization": True,
            "img_file": path_convert('./assets/drone.png'),
        }
    }

    # passenger_seq = {
    # 'num_seconds': args.num_seconds + 10,
    # "passenger_seq": {
    #     "step_0": [[(2220, 1400), (1850, 1100)], [(1450, 2000), (1270, 450)], [(650, 1000), (1700, 620)]],
    #     "step_5": [[(1010, 2220), (400, 1200)]],
    # }
    # }

    passenger_pos = {
        "passenger_pos": [(-100, 0), (0, 0), (100, 0)],
    }

    env = ACEnvironment(
        sumo_cfg=sumo_cfg,
        tls_ids=["1", "2"],
        num_seconds=500,
        aircraft_inits=aircraft_inits,
        use_gui=True,
    )
    env = ACEnvWrapper(env, aircraft_inits)


    env_test = ACEnvironment(
        sumo_cfg=sumo_cfg,
        tls_ids=["1", "2"],
        num_seconds=500,
        aircraft_inits=aircraft_inits,
        use_gui=True,
    )
    env_test = ACEnvWrapper(env_test, aircraft_inits)
    # env = TransformedEnv(
    #     env,
    #     RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    # )
    # obs = env.reset()
    train(cfg, env=env, env_test=env_test)












    # env = make_parallel_env(
    #     sumo_cfg=sumo_cfg,
    #     tls_ids=["1", "2"],
    #     num_seconds=500,
    #     aircraft_inits=aircraft_inits,
    #     num_envs=1,   # 可以根据GPU/CPU能力调整
    #     use_gui=False,
    #     seed=42
    # )
    
    