import os
import sys
import torch
import torch.nn as nn
import numpy as np


from train.scaler_M import Scaler
from train.mamba_policy import MambaPolicy, MambaConfig


class MyInferenceModel(nn.Module):
    """
    包含:
      - policy_arm1 / policy_arm2
      - scaler
      - load_checkpoint
      - predict / denormalize
    """
    def __init__(self, checkpoint_path: str, config: MambaConfig, lowdim_dict: dict):
        """
        :param checkpoint_path: 已训练好的 ckpt 路径
        :param config: 与训练时相同的 MambaConfig
        :param lowdim_dict: 与训练时相同的 lowdim_shapes, 用于构造 Scaler
        """
        super().__init__()
        print("[MyInferenceModel] Initializing...")

        self.policy = MambaPolicy(
            camera_names=config.camera_names,
            backbone=config.backbone,
            pretrained_backbone=config.pretrained_backbone,
            freeze_backbone=config.freeze_backbone,
            embed_dim=config.embed_dim,
            lowdim_dim=14,
            d_model=config.d_model,
            action_dim=14,
            sum_camera_feats=config.sum_camera_feats,
            num_blocks=config.num_blocks,
            mamba_cfg={
                'd_state': config.d_state,
                'd_conv': config.d_conv,
                'expand': config.expand,
                'headdim': config.headdim,
                'activation': config.activation,
                'use_mem_eff_path': config.use_mem_eff_path,
            }
        )
        print("[MyInferenceModel]  Policy created.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = Scaler(lowdim_dict=lowdim_dict)
        self.scaler.load('scaler_params.pth')
        self.policy.to(self.device)
        print(f"[MyInferenceModel] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cuda:0')
        state_dict = ckpt['state_dict']

        # 键名替换（移除所有'policy.'前缀）
        filtered_policy = {k.replace('policy.', ''): v for k, v in state_dict.items() if k.startswith('policy.')}
        # 使用 strict=False 并打印缺失/多余的键
        # missing, unexpected = self.policy.load_state_dict(filtered_policy, strict=False)
        # print(f"Missing keys: {missing}, Unexpected keys: {unexpected}")
        # 加载权重
        self.policy.load_state_dict(filtered_policy, strict=True)

        self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)
        # 添加设备检查
        print("[MyInferenceModel] hidden devices:",
              [tensor.device for pair in self.hiddens for tensor in pair if tensor is not None])
        print("[MyInferenceModel] Weights loaded from checkpoint!")
        self.cuda()
        self.eval()

    def forward(self, lowdim, rgb):
        """
          lowdim_arm1: shape [B,7] => pose(6)+gripper(1)
          lowdim_arm2: shape [B,7]
          rgb: dict of { 'top','angle' } => [B, C, H, W]
        返回:
          [B,14] => [arm1(7), arm2(7)]
        """
        with torch.no_grad():
            # policy
            pred_action, self.hiddens = self.policy.step(lowdim, rgb, self.hiddens)

        return pred_action

    def reset_hiddens(self):
        self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)
        print("[MyInferenceModel] Hidden states reset.")

    def denormalize(self, actions):
        """
        actions: shape [B,14], 前7=arm1(6+1), 后7=arm2(6+1)
        return: shape [B,14], 反归一化
        """
        arm1_dict = {
            'agl_1_act': actions[..., 0:1],'agl_2_act': actions[..., 1:2],'agl_3_act': actions[..., 2:3],
            'agl_4_act': actions[..., 3:4],'agl_5_act': actions[..., 4:5],'agl_6_act': actions[..., 5:6],
            'gripper_act': actions[..., 6:7]
        }
        arm2_dict = {
            'agl2_1_act': actions[..., 7:8],'agl2_2_act': actions[..., 8:9],'agl2_3_act': actions[..., 9:10],
            'agl2_4_act': actions[..., 10:11],'agl2_5_act': actions[..., 11:12],'agl2_6_act': actions[..., 12:13],
            'gripper_act2': actions[..., 13:14]
        }
        arm1_denorm = self.scaler.denormalize(arm1_dict)
        arm2_denorm = self.scaler.denormalize(arm2_dict)
        out = torch.cat([
            arm1_denorm['agl_1_act'],arm1_denorm['agl_2_act'],arm1_denorm['agl_3_act'],
            arm1_denorm['agl_4_act'],arm1_denorm['agl_5_act'],arm1_denorm['agl_6_act'],
            arm1_denorm['gripper_act'],
            arm2_denorm['agl2_1_act'],arm2_denorm['agl2_2_act'],arm2_denorm['agl2_3_act'],
            arm2_denorm['agl2_4_act'],arm2_denorm['agl2_5_act'],arm2_denorm['agl2_6_act'],
            arm2_denorm['gripper_act2']
        ], dim=2)
        return out
