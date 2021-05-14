## Coordinated Hunting Experiments with MADDPG

Code for training [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) agents in a collective hunting task. 

### Command-line options

- `--num-predators`: number of predators in the environment (default: `3`)

- `--speed` speed of the prey as multiples of predator speed (default: `1.0`)

- `--cost` cost-action ratio for predators (default: `0.0`)

- `--selfish`: predator selfish index (default: `0.0`)

- `--num-traj`: number of trajectories to sample (default: `10`)

- `--visualize`: whether to generate demos for sampled trajectories (default: `1`)

- `--save-images`: whether to save demo images (default: `1`)


### Required Packages

* python 3.7.3
* tensorflow 1.13.1
* Numpy 1.16.4
* Pygame 1.9.6

### Code Structure

- `./exec/train.py`: contains code for training MADDPG agents

- `./exec/evaluate.py`: contains code for evaluating MADDPG agents

- `./src/environment/multiAgentEnv.py`, `./src/environment/reward.py`: collective hunting environment code

- `./src/functionTools/loadSaveModel.py`, `./src/functionTools/trajectory.py`: function tools used in training

- `./src/maddpg/rlTools/RLrun.py`, `./src/maddpg/rlTools/tf_util.py`: RL training functions used

- `./src/maddpg/trainer/MADDPG.py`: core code for maddpg training

- `./visualize/drawDemo.py`: visualization code used in `evaluate.py`

- `requirements.txt`: contains requirements for model training and evaluation


### Works Cited
* [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).
* [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

