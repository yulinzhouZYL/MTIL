import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from tqdm import trange, tqdm
from scaler_M import Scaler  # 确保导入正确的 Scaler 类
import h5py

class MambaSequenceDataset(Dataset):
    """
    将“每个 record 文件夹”视为“一条时序轨迹”。
    每个文件夹包含多帧数据，逐帧加载并返回。
    每次切换轨迹时，重置隐状态。
    """
    def __init__(self, root_dir: str, mode: str = "train", use_pose10d: bool = True,
                 resize_hw=(640,480), selected_cameras: List[str] = None,
                 scaler: Optional[Scaler] = None,
                 future_steps=16):  # <-- 未来多少步
        super().__init__()
        assert mode in ["train", "test"], "mode must be 'train' or 'test'"
        self.dataset_dir = os.path.join(root_dir, mode)
        self.use_pose10d = use_pose10d
        self.resize_hw = resize_hw
        self.future_steps = future_steps
        self.selected_cameras = selected_cameras
        if self.selected_cameras is None:
            self.selected_cameras = ['top']
        # 加载所有轨迹路径并记录每条轨迹的长度
        self.records = []
        self.lengths = []  # 记录每条轨迹的长度
        for item in os.listdir(self.dataset_dir):
            record_path = os.path.join(self.dataset_dir, item)
            self.records.append(record_path)
            with h5py.File(record_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                self.lengths.append(qpos.shape[0])

        # 累积轨迹长度，用于全局索引到轨迹索引的映射
        self.cum_lengths = np.cumsum([0] + self.lengths)

        # 定义低维数据的键和形状
        self.lowdim_keys = [
            'agl_1', 'agl_2', 'agl_3', 'agl_4', 'agl_5', 'agl_6',
            'agl2_1', 'agl2_2', 'agl2_3', 'agl2_4', 'agl2_5', 'agl2_6'
            'gripper_pos', 'gripper_pos2',
            'agl_1_act', 'agl_2_act', 'agl_3_act', 'agl_4_act', 'agl_5_act', 'agl_6_act',
            'agl2_1_act', 'agl2_2_act', 'agl2_3_act', 'agl2_4_act', 'agl2_5_act', 'agl2_6_act'
            'gripper_act', 'gripper_act2'
        ]
        self.lowdim_shapes = {
            'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
            'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
            'gripper_pos': 1,
            'gripper_pos2': 1,
            'agl_1_act': (future_steps, 1), 'agl_2_act': (future_steps, 1), 'agl_3_act': (future_steps, 1),
            'agl_4_act': (future_steps, 1), 'agl_5_act': (future_steps, 1), 'agl_6_act': (future_steps, 1),
            'agl2_1_act': (future_steps, 1), 'agl2_2_act': (future_steps, 1), 'agl2_3_act': (future_steps, 1),
            'agl2_4_act': (future_steps, 1), 'agl2_5_act': (future_steps, 1), 'agl2_6_act': (future_steps, 1),
            'gripper_act': (future_steps, 1),
            'gripper_act2': (future_steps, 1)
        }

        # 初始化 Scaler
        self.scaler = scaler
        if self.scaler is None and mode == "train":
            # 如果没有提供 Scaler 且是训练模式，则初始化一个 Scaler
            self.scaler = Scaler(lowdim_dict=self.lowdim_shapes)
            self.fitting = False  # 标志是否在拟合
        else:
            self.fitting = False

    def __len__(self):
        return self.cum_lengths[-1]

    def fit_scaler(self, batch_size=64, num_workers=4):
        """
        计算归一化参数。
        """
        if not self.scaler:
            raise ValueError("Scaler is not initialized.")

        print("Fitting scaler on dataset...")
        self.fitting = True  # 开始拟合，禁用归一化
        data_cache = {key: [] for key in self.scaler.lowdim_dict.keys()}
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
        for batch in tqdm(dataloader, desc='Fitting scaler'):
            lowdim = batch['lowdim']
            for key in self.scaler.lowdim_dict.keys():
                data_cache[key].append(lowdim[key])
        self.fitting = False  # 拟合完成，启用归一化

        # 将所有批次的数据拼接起来
        for key in data_cache.keys():
            data_cache[key] = torch.cat(data_cache[key], dim=0)
        # 计算最小值和最大值
        self.scaler.fit(data_cache)
        print("Scaler fitted.")
        return self.scaler


    def save_scaler(self, filepath: str):
        """
        保存 Scaler 的归一化参数到文件。
        """
        if self.scaler:
            self.scaler.save(filepath)
            print(f"Scaler saved to {filepath}.")
        else:
            raise ValueError("Scaler is not initialized.")

    def load_scaler(self, filepath: str):
        """
        从文件加载 Scaler 的归一化参数。
        """
        if self.scaler:
            self.scaler.load(filepath)
            print(f"Scaler loaded from {filepath}.")
        else:
            raise ValueError("Scaler is not initialized.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 根据全局idx找到对应轨迹和帧
        traj_idx = np.searchsorted(self.cum_lengths, idx, side='right') - 1
        frame_idx = idx - self.cum_lengths[traj_idx]
        record_path = self.records[traj_idx]

        # 加载低维数据
        with h5py.File(record_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            act = root['/action'][()]
            agl_1 = qpos[:, 0:1]
            agl_2 = qpos[:, 1:2]
            agl_3 = qpos[:, 2:3]
            agl_4 = qpos[:, 3:4]
            agl_5 = qpos[:, 4:5]
            agl_6 = qpos[:, 5:6]
            gripper_pos = qpos[:, 6:7]
            agl2_1 = qpos[:, 7:8]
            agl2_2 = qpos[:, 8:9]
            agl2_3 = qpos[:, 9:10]
            agl2_4 = qpos[:, 10:11]
            agl2_5 = qpos[:, 11:12]
            agl2_6 = qpos[:, 12:13]
            gripper_pos2 = qpos[:, 13:14]
            agl_1_act = act[:, 0:1]
            agl_2_act = act[:, 1:2]
            agl_3_act = act[:, 2:3]
            agl_4_act = act[:, 3:4]
            agl_5_act = act[:, 4:5]
            agl_6_act = act[:, 5:6]
            gripper_act = act[:, 6:7]
            agl2_1_act = act[:, 7:8]
            agl2_2_act = act[:, 8:9]
            agl2_3_act = act[:, 9:10]
            agl2_4_act = act[:, 10:11]
            agl2_5_act = act[:, 11:12]
            agl2_6_act = act[:, 12:13]
            gripper_act2 = act[:, 13:14]


        def shift_N_steps(arr):
            """返回 [future_steps, arr_dim] 的多步标签."""
            out_list = []
            for step in range(self.future_steps):
                future_idx = frame_idx + step
                if future_idx >= len(arr):
                    future_idx = len(arr) - 1  # 超出则用最后一帧
                out_list.append(arr[future_idx])
            return np.stack(out_list, axis=0)  # [16, arr_dim]

        agl_1_act = shift_N_steps(agl_1_act)
        agl_2_act = shift_N_steps(agl_2_act)
        agl_3_act = shift_N_steps(agl_3_act)
        agl_4_act = shift_N_steps(agl_4_act)
        agl_5_act = shift_N_steps(agl_5_act)
        agl_6_act = shift_N_steps(agl_6_act)
        agl2_1_act = shift_N_steps(agl2_1_act)
        agl2_2_act = shift_N_steps(agl2_2_act)
        agl2_3_act = shift_N_steps(agl2_3_act)
        agl2_4_act = shift_N_steps(agl2_4_act)
        agl2_5_act = shift_N_steps(agl2_5_act)
        agl2_6_act = shift_N_steps(agl2_6_act)
        gripper_act = shift_N_steps(gripper_act)
        gripper_act2 = shift_N_steps(gripper_act2)


        # 提取单步的低维数据
        agl_1 = agl_1[frame_idx]
        agl_2 = agl_2[frame_idx]
        agl_3 = agl_3[frame_idx]
        agl_4 = agl_4[frame_idx]
        agl_5 = agl_5[frame_idx]
        agl_6 = agl_6[frame_idx]
        agl2_1 = agl2_1[frame_idx]
        agl2_2 = agl2_2[frame_idx]
        agl2_3 = agl2_3[frame_idx]
        agl2_4 = agl2_4[frame_idx]
        agl2_5 = agl2_5[frame_idx]
        agl2_6 = agl2_6[frame_idx]
        gripper_pos = gripper_pos[frame_idx]
        gripper_pos2 = gripper_pos2[frame_idx]

        # 低维数据转为 PyTorch Tensor
        def ensure_1d_array(arr):
            """确保输入的数组是至少一维的"""
            if len(arr.shape) == 0:  # 如果是标量，转成一个一维张量
                arr = np.expand_dims(arr, axis=0)
            return torch.tensor(arr, dtype=torch.float32)

        agl_1 = ensure_1d_array(agl_1)
        agl_2 = ensure_1d_array(agl_2)
        agl_3 = ensure_1d_array(agl_3)
        agl_4 = ensure_1d_array(agl_4)
        agl_5 = ensure_1d_array(agl_5)
        agl_6 = ensure_1d_array(agl_6)
        agl2_1 = ensure_1d_array(agl2_1)
        agl2_2 = ensure_1d_array(agl2_2)
        agl2_3 = ensure_1d_array(agl2_3)
        agl2_4 = ensure_1d_array(agl2_4)
        agl2_5 = ensure_1d_array(agl2_5)
        agl2_6 = ensure_1d_array(agl2_6)
        gripper_pos = ensure_1d_array(gripper_pos)
        gripper_pos2 = ensure_1d_array(gripper_pos2)

        agl_1_act = ensure_1d_array(agl_1_act)
        agl_2_act = ensure_1d_array(agl_2_act)
        agl_3_act = ensure_1d_array(agl_3_act)
        agl_4_act = ensure_1d_array(agl_4_act)
        agl_5_act = ensure_1d_array(agl_5_act)
        agl_6_act = ensure_1d_array(agl_6_act)
        agl2_1_act = ensure_1d_array(agl2_1_act)
        agl2_2_act = ensure_1d_array(agl2_2_act)
        agl2_3_act = ensure_1d_array(agl2_3_act)
        agl2_4_act = ensure_1d_array(agl2_4_act)
        agl2_5_act = ensure_1d_array(agl2_5_act)
        agl2_6_act = ensure_1d_array(agl2_6_act)
        gripper_act = ensure_1d_array(gripper_act)
        gripper_act2 = ensure_1d_array(gripper_act2)

        # 确保 gripper_act 和 gripper_act2 的形状是 [16,1]
        if gripper_act.ndim == 1:
            gripper_act = gripper_act.unsqueeze(-1)  # => [16,1]
        if gripper_act2.ndim == 1:
            gripper_act2 = gripper_act2.unsqueeze(-1)  # => [16,1]


        # 低维数据转为 float
        agl_1 = agl_1.clone().detach().float()
        agl_2 = agl_2.clone().detach().float()
        agl_3 = agl_3.clone().detach().float()
        agl_4 = agl_4.clone().detach().float()
        agl_5 = agl_5.clone().detach().float()
        agl_6 = agl_6.clone().detach().float()
        agl2_1 = agl2_1.clone().detach().float()
        agl2_2 = agl2_2.clone().detach().float()
        agl2_3 = agl2_3.clone().detach().float()
        agl2_4 = agl2_4.clone().detach().float()
        agl2_5 = agl2_5.clone().detach().float()
        agl2_6 = agl2_6.clone().detach().float()
        gripper_pos = gripper_pos.clone().detach().float()
        gripper_pos2 = gripper_pos2.clone().detach().float()

        agl_1_act = agl_1_act.clone().detach().float()
        agl_2_act = agl_2_act.clone().detach().float()
        agl_3_act = agl_3_act.clone().detach().float()
        agl_4_act = agl_4_act.clone().detach().float()
        agl_5_act = agl_5_act.clone().detach().float()
        agl_6_act = agl_6_act.clone().detach().float()
        agl2_1_act = agl2_1_act.clone().detach().float()
        agl2_2_act = agl2_2_act.clone().detach().float()
        agl2_3_act = agl2_3_act.clone().detach().float()
        agl2_4_act = agl2_4_act.clone().detach().float()
        agl2_5_act = agl2_5_act.clone().detach().float()
        agl2_6_act = agl2_6_act.clone().detach().float()
        gripper_act = gripper_act.clone().detach().float()
        gripper_act2 = gripper_act2.clone().detach().float()

        lowdim_dict = {
            'agl_1': agl_1, 'agl_2': agl_2, 'agl_3': agl_3, 'agl_4': agl_4, 'agl_5': agl_5, 'agl_6': agl_6,
            'agl2_1': agl2_1, 'agl2_2': agl2_2, 'agl2_3': agl2_3, 'agl2_4': agl2_4, 'agl2_5': agl2_5, 'agl2_6': agl2_6,
            'gripper_pos': gripper_pos,
            'gripper_pos2': gripper_pos2,
            'agl_1_act': agl_1_act, 'agl_2_act': agl_2_act, 'agl_3_act': agl_3_act,
            'agl_4_act': agl_4_act, 'agl_5_act': agl_5_act, 'agl_6_act': agl_6_act,
            'agl2_1_act': agl2_1_act, 'agl2_2_act': agl2_2_act, 'agl2_3_act': agl2_3_act,
            'agl2_4_act': agl2_4_act, 'agl2_5_act': agl2_5_act, 'agl2_6_act': agl2_6_act,
            'gripper_act': gripper_act,
            'gripper_act2': gripper_act2
        }

        if getattr(self, 'fitting', False):
            # 如果正在拟合，不需要加载图像
            rgb_dict = {}
        else:
            # 加载图像数据
            def read_image_bgr_as_float(path):
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    return None
                if self.resize_hw is not None:
                    # resize e.g. (640, 480)
                    w, h = self.resize_hw
                    img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
                img_bgr = img_bgr.astype(np.float32) / 255.0
                img_bgr = np.transpose(img_bgr, (2, 0, 1))  # => (C,H,W)
                return img_bgr

            rgb_dict = {}
            # 加载选定的相机
            for cam in self.selected_cameras:
                if cam == 'angle':
                    with h5py.File(record_path, 'r') as root:
                        img = root['observations/images/angle'][()]
                        img_bgr = img[frame_idx]
                        img_bgr = img_bgr.astype(np.float32) / 255.0
                        img_bgr = np.transpose(img_bgr, (2, 0, 1))
                        rgb_dict['angle'] = img_bgr
                elif cam == 'top':
                    with h5py.File(record_path, 'r') as root:
                        img = root['observations/images/top'][()]
                        img_bgr = img[frame_idx]
                        img_bgr = img_bgr.astype(np.float32) / 255.0
                        img_bgr = np.transpose(img_bgr, (2, 0, 1))
                        rgb_dict['top'] = img_bgr

            # 图像转为PyTorch Tensor
            rgb_dict = {k: torch.tensor(v, dtype=torch.float32) if v is not None else None for k, v in rgb_dict.items()}

        data_dict = {
            'lowdim': lowdim_dict,
            'rgb': rgb_dict,
            'traj_idx': traj_idx
        }

        return data_dict


def main():
    root_dir = "insert_data200"  # 你自己的数据集目录
    dataset = MambaSequenceDataset(root_dir=root_dir, mode="train", use_pose10d=True)

    # 计算归一化参数
    scaler = dataset.fit_scaler(batch_size=64, num_workers=0)
    # 保存归一化参数
    dataset.save_scaler('scaler_params.pth')

    # 测试 __getitem__ 方法
    data_dict = dataset[0]  # 获取第一条数据

    # 打印 lowdim_dict 和 rgb_dict 中每个张量的维度
    lowdim_dict = data_dict['lowdim']
    rgb_dict = data_dict['rgb']

    print("Lowdim dict dimensions:")
    for key, value in lowdim_dict.items():
        print(f"{key}: {value.shape}")

    print("\nRGB dict dimensions:")
    for key, value in rgb_dict.items():
        if value is not None:
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: None")



if __name__ == "__main__":
    main()
