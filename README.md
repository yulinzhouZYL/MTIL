# MTIL: Encoding Full History with Mamba for Temporal Imitation Learning

This repository contains the official implementation for the paper:

MTIL: Encoding Full History with Mamba for Temporal Imitation Learning [YulinZhou]  （Submitted to CoRL2025）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Core Idea and Principle

Standard imitation learning (IL) methods often struggle with long-horizon sequential tasks due to reliance on the Markov assumption, which limits their ability to use historical context to resolve observational ambiguity. This is particularly problematic in manipulation tasks requiring ordered steps, where similar observations at different stages can lead to incorrect actions (e.g., skipping a necessary intermediate step).

**Mamba Temporal Imitation Learning (MTIL)** addresses this challenge by leveraging the **Mamba architecture**, a type of State Space Model (SSM). The core principle is to utilize Mamba's recurrent hidden state to **encode the entire trajectory history** into a compressed representation. By conditioning action predictions on this comprehensive historical context alongside current observations, MTIL can effectively disambiguate states and enable robust execution of complex, state-dependent sequential tasks that foil traditional Markovian approaches.

## Code Overview

This codebase provides the implementation for the MTIL agent. Currently, it includes scripts and configurations primarily for:

* Training MTIL policies on tasks from the **ACT dataset (ALOHA simulated tasks)**.
* Running inference (evaluation) with trained MTIL policies on the ACT dataset tasks.


## Installation

We recommend using Conda for environment management.

1.  **Create Conda Environment:**
    ```bash
    conda create -n mtil python=3.9
    conda activate mtil
    ```

2.  **Install Dependencies:**
    *(Note: `pytorch` and `torchvision` installation might vary based on your CUDA version. Please refer to the official PyTorch website for specific commands if needed.)*
    ```bash
    # Install PyTorch and Torchvision (Example for CPU/CUDA 11.x, adjust as needed)
    # pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    pip install torch torchvision # Simpler version, might work depending on setup

    # Install other dependencies
    pip install pyquaternion pyyaml rospkg pexpect opencv-python matplotlib einops packaging h5py ipython pytorch-lightning

    # Install MuJoCo and dm_control (versions specified in requirements)
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14

    # Install DETR components (modify path if needed)
    cd act/detr 
    pip install -e .
    cd ../.. # Return to root directory

    # Install Mamba-SSM (Requires CUDA and C++ compiler, ensure pip >= 23.1)
    # The --no-build-isolation flag might be needed depending on your environment
    pip install mamba-ssm causal-conv1d --no-build-isolation 
    # OR try without the flag first if you encounter issues:
    # pip install mamba-ssm causal-conv1d
    ```
    *(Note: `mamba-ssm` installation can sometimes be tricky. Refer to the official mamba-ssm repository if you encounter compilation issues.)*

## Usage


1.**Training and Evaluation:**

```bash
# Example command for training on an ACT task
python train_mtil.py --task [task_name] --dataset_dir [path_to_dataset] ...


# Example command for evaluating a trained policy
python evaluate_mtil.py --policy_path [path_to_checkpoint] --task [task_name] ...
