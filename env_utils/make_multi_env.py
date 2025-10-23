import torch
from torchrl.envs import ParallelEnv
from functools import partial
from env_utils.ac_env import ACEnvironment
from env_utils.ac_MultiAgent import ACEnvWrapper


def make_single_env(sumo_cfg, tls_ids, num_seconds, aircraft_inits, use_gui=False, seed=0, device="cpu"):
    """
    创建单个ACEnvWrapper环境实例

    Args:
        sumo_cfg (str): SUMO配置文件路径
        tls_ids (list): 交通信号灯ID列表
        num_seconds (int): 仿真时长（秒）
        aircraft_inits (dict): 无人机初始配置
        use_gui (bool): 是否使用SUMO GUI
        seed (int): 随机种子
        device (str): 计算设备（"cpu" 或 "cuda"）
    
    Returns:
        ACEnvWrapper: 封装后的环境实例
    """
    # 构建基础环境
    base_env = ACEnvironment(
        sumo_cfg=sumo_cfg,
        tls_ids=tls_ids,
        num_seconds=num_seconds,
        aircraft_inits=aircraft_inits,
        use_gui=use_gui,
    )

    # 用自定义Wrapper封装
    wrapped_env = ACEnvWrapper(
        env=base_env,
        aircraft_inits=aircraft_inits,
        max_states=3,
        max_rel_veh=5,
        device=device,
    )

    # 设置种子
    wrapped_env._set_seed(seed)
    return wrapped_env


def make_parallel_env(
    sumo_cfg: str,
    tls_ids: list,
    num_seconds: int,
    aircraft_inits: dict,
    num_envs: int = 4,
    use_gui: bool = False,
    seed: int = 0,
    device: str = "cpu",
):
    """
    创建TorchRL兼容的并行环境 (ParallelEnv)

    Args:
        sumo_cfg (str): SUMO配置文件路径
        tls_ids (list): 交通信号灯ID列表
        num_seconds (int): 仿真时长（秒）
        aircraft_inits (dict): 无人机初始配置
        num_envs (int): 并行环境数量
        use_gui (bool): 是否使用SUMO GUI
        seed (int): 基础随机种子
        device (str): 计算设备（"cpu" 或 "cuda"）

    Returns:
        ParallelEnv: 并行环境实例
    """
    make_env_fn = partial(
        make_single_env,
        sumo_cfg=sumo_cfg,
        tls_ids=tls_ids,
        num_seconds=num_seconds,
        aircraft_inits=aircraft_inits,
        use_gui=use_gui,
        device=device,
    )

    # 为每个并行环境设置唯一种子
    create_env_kwargs = [{"seed": seed + i} for i in range(num_envs)]

    env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=make_env_fn,
        create_env_kwargs=create_env_kwargs,
        shared_memory=False,  # 防止SUMO类环境共享内存出错
    )

    return env
