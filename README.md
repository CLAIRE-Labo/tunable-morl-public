# Policy Optimization for Dynamic Multi-Objective Reinforcement Learning
This repository contains the supplementary code for the paper
> Terekhov, M., & Gulcehre, C. In Search for Architectures and Loss Functions in Multi-Objective Reinforcement Learning. In ICML 2024 Workshop: Aligning Reinforcement Learning Experimentalists and Theorists.

## Installation
For reproducible experiments, we used Docker containers for dependency management. See the [installation readme](installation/docker-amd64/README.md) for more details on building the container within or outside the EPFL ecosystem. We also provide the [`environment.yml`](installation/docker-amd64/dependencies/environment.yml) file for a conda environment which can be used without a container, but with less reproducibility guarantees.

## Overview
The implementation here provides the main algorithm as well as all actor/critic architectures described in the paper. The entry point to run our algorithm is the [`train_moe.py`](src/pymorl/train_moe.py) script. We use [Hydra](https://hydra.cc/) for configuration management. The default configuration can be found [here](src/pymorl/configs/train_moe.yaml).

## Built With
This repository is based on the [template for reproducible code](https://github.com/CLAIRE-Labo/python-ml-research-template) by Skander Moalla. The code is written with the [TorchRL](https://pytorch.org/rl/stable/index.html) library. We used [MORL-baselines](https://lucasalegre.github.io/morl-baselines/) as a source of state-of-the-art algorithms for multi-objective reinforcement learning.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
