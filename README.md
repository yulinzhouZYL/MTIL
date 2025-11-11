# MTIL: Encoding Full History with Mamba for Temporal Imitation Learning

[![arXiv](https://img.shields.io/badge/arXiv-2505.12410-b31b1b.svg)](https://arxiv.org/abs/2505.12410)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official implementation of our paper, which has been accepted for publication in the **IEEE Robotics and Automation Letters (RA-L)**.

- **[arXiv Preprint]**: [https://arxiv.org/abs/2505.12410](https://arxiv.org/abs/2505.12410)
- **[IEEE Xplore Publication]**: [https://ieeexplore.ieee.org/document/11184145](https://ieeexplore.ieee.org/document/11184145)

## Core Idea and Principle

Standard imitation learning (IL) methods often struggle with long-horizon sequential tasks due to reliance on the Markov assumption, which limits their ability to use historical context to resolve observational ambiguity. This is particularly problematic in manipulation tasks requiring ordered steps, where similar observations at different stages can lead to incorrect actions (e.g., skipping a necessary intermediate step).

![QQ_1747044302341](https://github.com/user-attachments/assets/d6f1d07d-326e-4d50-94fb-191bff9d6143)

**Mamba Temporal Imitation Learning (MTIL)** addresses this challenge by leveraging the **Mamba architecture**, a type of State Space Model (SSM). The core principle is to utilize Mamba's recurrent hidden state to **encode the entire trajectory history** into a compressed representation. By conditioning action predictions on this comprehensive historical context alongside current observations, MTIL can effectively disambiguate states and enable robust execution of complex, state-dependent sequential tasks that foil traditional Markovian approaches, achieving superior performance as detailed within the paper.
## Code Overview

This codebase provides the implementation for the MTIL agent. Currently, it includes scripts and configurations primarily for:

* Training MTIL policies on tasks from the **ACT dataset (ALOHA simulated tasks)**.
* Running inference (evaluation) with trained MTIL policies on the ACT dataset tasks.
* **Note:** Code for real-world experiments will be cleaned up and released in a future update.

## Installation

We recommend using Conda for environment management.

1.  **Create Conda Environment:**
    ```bash
    conda create -n mtil python=3.9
    conda activate mtil
    ```

2.  **Install Dependencies:**
    *(Note: `pytorch` and `torchvision` installation might vary based on your CUDA version. Please refer to the official PyTorch website for specific commands if needed.)*

    First, install PyTorch according to your system configuration (CUDA/CPU). For example:
    ```bash
    # Example for CUDA 11.8 (check PyTorch website for your specific CUDA version)
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    # For CPU only:
    pip3 install torch torchvision torchaudio
    ```

    Then, install Mamba-SSM (Requires CUDA and a C++ compiler for optimized CUDA kernels. CPU-only operation is also possible but slower):
    ```bash
    # The --no-build-isolation flag might be needed depending on your environment and pip version
    pip install mamba-ssm causal-conv1d --no-build-isolation
    # OR try without the flag first if you encounter issues:
    # pip install mamba-ssm causal-conv1d
    ```
    *(Note: `mamba-ssm` installation can sometimes be tricky, especially the CUDA compiled components. Refer to the [official mamba-ssm repository](https://github.com/state-spaces/mamba) if you encounter compilation issues. Ensure you have a compatible C++ compiler and CUDA toolkit installed if using GPU support.)*

    Finally, install other dependencies:
    ```bash
    pip install pyquaternion pyyaml rospkg pexpect opencv-python matplotlib einops packaging h5py ipython pytorch-lightning mujoco==2.3.7 dm_control==1.0.14
    ```

## Usage

Follow these steps to prepare the data, train, and evaluate the MTIL model:

1.  **Generate Simulation Datasets:**
    * The simulation datasets (`transfer_cube` and `insertion`) from the ACT project are required for training and evaluation with the provided scripts.
    * Please generate these datasets by following the instructions provided in the **'Simulated experiments'** section of the original ACT repository:
        **[https://github.com/tonyzhaozh/act](https://github.com/tonyzhaozh/act)**

2.  **Prepare Data and Normalization:**
    * Place the generated `transfer_cube` and `insertion` task datasets into your desired local directory.
    * Run the `M_dataset.py` script. **You may need to modify paths within the `M_dataset.py` script itself to point to your dataset locations and desired output locations for `scaler_params.pth`.**
        ```bash
        python M_dataset.py
        ```
    * This script processes the datasets and generates the necessary normalization parameter file `scaler_params.pth`. Make sure this file is accessible for the training script (it's often saved in a specific data or checkpoint directory).

3.  **Training:**
    * Once the datasets are generated and `scaler_params.pth` is created, you can start training.
    * **You may need to configure dataset paths, scaler path, checkpoint directory, and other hyperparameters directly within the `train.py` script.**
    * Execute the training script:
        ```bash
        python train.py
        ```

4.  **Evaluation:**
    * To evaluate a trained policy checkpoint, use the evaluation scripts.
    * **Ensure checkpoint paths, dataset paths, and scaler paths are correctly specified within the respective evaluation scripts**
    * command for evaluating on the insertion task
    *  ```bash
        python evaluate_model_insertion.py
        ```
    * command for evaluating on the transfer_cube task
        ```bash
        python evaluate_model_transfer.py
        ```

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{Zhou2025MTIL,
  author={Zhou, Yulin and Lin, Yuankai and Peng, Fanzhe and Chen, Jiahui and Huang, Kaiji and Yang, Hua and Yin, Zhouping},
  journal={IEEE Robotics and Automation Letters}, 
  title={MTIL: Encoding Full History with Mamba for Temporal Imitation Learning}, 
  year={2025},
  volume={10},
  number={11},
  pages={11761-11767},
  doi={10.1109/LRA.2025.3615520}
}
