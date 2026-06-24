import os
import sys
import torch
import torch.nn as nn
import numpy as np
from M_dataset_par import get_image_transform # 导入归一化函数

from train.scaler_M import Scaler
from mamba_policy_par import MambaPolicy, MambaConfig

class MyInferenceModel(nn.Module):
    """
    包含:
      - policy_arm1 / policy_arm2
      - scaler
      - load_checkpoint
      - predict / denormalize
    """
    def __init__(self, checkpoint_path: str, scaler_path: str, config: MambaConfig, lowdim_dict: dict):
        """
        :param checkpoint_path: 已训练好的 ckpt 路径
        :param config: 与训练时相同的 MambaConfig
        :param lowdim_dict: 与训练时相同的 lowdim_shapes, 用于构造 Scaler
        """
        super().__init__()
        print("[MyInferenceModel] Initializing...")

        self.policy = MambaPolicy(
            camera_names=config.camera_names,
            embed_dim=config.embed_dim,
            lowdim_dim=config.lowdim_dim,
            d_model=config.d_model,
            action_dim=config.action_dim,
            sum_camera_feats=config.sum_camera_feats,
            num_blocks=config.num_blocks,
            mamba_cfg=config,
            use_backbone=True,  # <--- [关键] 推理必须为 True
            future_steps=16 # 显式传入
        )
        print("[MyInferenceModel]  Policy created.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = Scaler(lowdim_dict=lowdim_dict)
        self.scaler.load(scaler_path)
        # 3. 初始化归一化函数 (640x480)
        self.transform_func = get_image_transform(config.img_size)
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
        # [关键] strict=False
        # 因为 checkpoint 里只有 adapter+mamba 的权重，没有 shared_backbone 的权重
        # backbone 的权重已经在初始化 FrozenDinov2 时由 facebook 预训练模型加载好了
        missing, unexpected = self.policy.load_state_dict(filtered_policy, strict=False)
        # 验证一下：Missing 的应该是 shared_backbone 相关，Unexpected 应该为空
        backbone_missing = [k for k in missing if 'shared_backbone' in k]
        other_missing = [k for k in missing if 'shared_backbone' not in k]
        
        if len(other_missing) > 0:
            print(f"[WARNING] Unrelated keys missing: {other_missing}")
        print(f"[Info] Backbone weights skipped from ckpt (using pretrained): {len(backbone_missing)} keys.")
        # self.policy.load_state_dict(filtered_policy, strict=True)

        self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)
        # 添加设备检查
        print("[MyInferenceModel] hidden devices:",
              [tensor.device for pair in self.hiddens for tensor in pair if tensor is not None])
        print("[MyInferenceModel] Weights loaded from checkpoint!")
        self.cuda()
        self.eval()

    def forward(self, lowdim, rgb):
        """
          lowdim: shape [B, 14]
          rgb: dict of { 'top': [B, C, H, W] }
        返回:
          [B, 16, 14] => 生成的未来动作块
        """
        with torch.no_grad():
            # 1. 图像预处理 (Numpy -> Tensor Normalized)
            rgb_tensor_dict = {}
            for cam, img_np in rgb.items():
                # 检查是否包含 Batch 维度
                # transform_func 期望 (H, W, 3)，如果是 (1, H, W, 3) 需要 squeeze
                if img_np.ndim == 4:
                    img_np = img_np[0] # 取出第一张
                
                # img_np: (H, W, 3) uint8 BGR/RGB
                # transform_func 会处理 ToTensor 和 Normalize -> (3, H, W)
                tensor_img = self.transform_func(img_np).to(self.device)
                
                # 增加 Batch 维度: [1, 3, H, W]
                rgb_tensor_dict[cam] = tensor_img.unsqueeze(0)

            # 2. Policy 推理
            # policy.step 会自动调用 backbone -> adapter -> mamba
            # 确保 lowdim 在正确的 device
            lowdim = lowdim.to(self.device)
            pred_action, self.hiddens = self.policy.step(lowdim, rgb_tensor_dict, self.hiddens)

        return pred_action


    def reset_hiddens(self):
        self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)
        print("[MyInferenceModel] Hidden states reset.")

    def denormalize(self, actions):
        """
        actions: shape [B, 16, 14]
        return: shape [B, 16, 14], 反归一化
        """
        # 使用 ... 自动适配 [B, 16, 14] 维度
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
        ], dim=2) # 在最后一个维度拼接 (dim=2 for [B, 16, 14])
        return out
