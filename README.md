# UAM-Path-Planning

## Quick start:
The basic env is under env_utils/ac_env.py. The encapsulated environment is under env_utils/ac_MultiAgent_test.py.

The key problem now is probably step() and reset() not match in ac_MultiAgent_test.py.

### 1. Install TransSimHub
```bash
git clone https://github.com/Traffic-Alpha/TransSimHub.git
cd TransSimHub
pip install -e .  # Install in editable mode
```

### 2. Install UAM-Path-Planning
```bash
git clone https://github.com/Traffic-Alpha/UAM-Path-Planning.git
cd UAM-Path-Planning
pip install -r requirements.txt
```

### 3. Install required package of torchRL
The package of torchRL has been downloaded in the local repository, just need to add pre-requisite package.
```bash
pip install "torchrl[utils]"
```

### 4. Train RL
You can train the RL model with the following code:
```bash
python torchRL_MAPPO.py
```





