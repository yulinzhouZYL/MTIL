import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
# 确保目录里有 scaler_M.py 且内容是你发给我的那个
from scaler_M import Scaler 
import cv2
from torchvision import transforms
cv2.setNumThreads(0)  # <--- 关键！防止 OpenCV 在 Worker 进程中乱开线程导致死锁或崩溃
cv2.ocl.setUseOpenCL(False) # 禁止 OpenCL，防止显存冲突

# === 公共归一化逻辑 (必须与 DINOv2 预训练保持一致) ===
def get_image_transform(resize_hw=(640, 480)):

    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_hw[1], resize_hw[0])), # (H, W)
        transforms.ToTensor(), # [0,1] & (C,H,W)
    ])
    
    def process(img_bgr):
        # 假设输入是 numpy array (H,W,3)
        return t(img_bgr)
    return process

class MambaTrajectoryDataset(Dataset):
    def __init__(self, root_dir: str, mode: str = "train", use_pose10d: bool = True,
                 resize_hw=(640,480), selected_cameras: List[str] = None,
                 scaler: Optional[Scaler] = None,
                 future_steps=16, load_features=False):
        super().__init__()
        assert mode in ["train", "test"], "mode must be 'train' or 'test'"
        self.load_features = load_features
        self.use_pose10d = use_pose10d
        self.resize_hw = resize_hw
        self.future_steps = future_steps
        self.selected_cameras = selected_cameras if selected_cameras else ['top']
        # 路径逻辑：自动切换到特征目录
        if self.load_features:
            self.dataset_dir = os.path.join(root_dir, mode) # Fallback
            print(f"[{mode}] Loading PRECOMPUTED FEATURES from: {self.dataset_dir}")
        else:
            self.dataset_dir = os.path.join(root_dir, mode)
            self.transform_func = get_image_transform(resize_hw)
            print(f"[{mode}] Loading RAW IMAGES from: {self.dataset_dir}")
        # 扫描文件夹 (现在每个轨迹是一个文件夹)
        self.records = []
        if os.path.exists(self.dataset_dir):
            # 排序很重要，保证索引一致
            # NPY模式下，dataset_dir 下面全是 episode_x 的文件夹
            dirs = sorted([d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))])
            self.records = [os.path.join(self.dataset_dir, d) for d in dirs]
            
        print(f"[{mode}] Found {len(self.records)} trajectories.")

        # 定义需要归一化的 Key
        self.lowdim_keys = [
            'agl_1', 'agl_2', 'agl_3', 'agl_4', 'agl_5', 'agl_6',
            'agl2_1', 'agl2_2', 'agl2_3', 'agl2_4', 'agl2_5', 'agl2_6',
            'gripper_pos', 'gripper_pos2',
            'agl_1_act', 'agl_2_act', 'agl_3_act', 'agl_4_act', 'agl_5_act', 'agl_6_act',
            'agl2_1_act', 'agl2_2_act', 'agl2_3_act', 'agl2_4_act', 'agl2_5_act', 'agl2_6_act',
            'gripper_act', 'gripper_act2'
        ]
        
        # 构建 lowdim_dict 传给 Scaler 初始化
        # 注意：Scaler 初始化需要知道 shape，这里我们用 dummy shape，scaler 内部主要用 keys
        self.lowdim_shapes = {k: 1 for k in self.lowdim_keys} 
        # Action 的 key shape 其实是 (16, 1)，但 Scaler 处理时只要 dim>1 就行
        
        self.scaler = scaler
        if self.scaler is None and mode == "train":
            self.scaler = Scaler(lowdim_dict=self.lowdim_shapes)
            self.fitting = False
        else:
            self.fitting = False

    def __len__(self):
        return len(self.records)

    def fit_scaler(self, batch_size=1, num_workers=0):
        """
        修复版 fit_scaler: 收集所有数据，一次性 fit
        """
        if not self.scaler:
            raise ValueError("Scaler is not initialized.")

        print("Fitting scaler on dataset (Low-dim only)...")
        self.fitting = True # 标记为正在拟合，__getitem__ 不会返回图片，节省内存
        
        # 容器：{key: [tensor_traj1, tensor_traj2, ...]}
        cache = {key: [] for key in self.lowdim_keys}
        
        # 使用 DataLoader 加速读取
        loader = DataLoader(self, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=lambda x: x[0])
        
        for item in tqdm(loader, desc='Collecting data'):
            raw = item['lowdim_raw']
            for k in self.lowdim_keys:
                # 如果是 action (L, 16, 1)，展平为 (L*16, 1) 或者保持 (L, 16, 1) 
                # Scaler 代码里 mean(dim=0)，如果是 (L, 16, 1)，mean 出来是 (16, 1)，这是对的
                # 但通常我们希望对所有时间步和预测步统一归一化，这里保持 Scaler 逻辑
                if k in raw:
                    cache[k].append(raw[k])
        
        # 合并数据
        full_data_dict = {}
        for k, v_list in cache.items():
            if len(v_list) > 0:
                full_data_dict[k] = torch.cat(v_list, dim=0) # [Total_L, ...]
        
        # 调用 Scaler.fit
        self.scaler.fit(full_data_dict)
        self.fitting = False
        print("Scaler fitted.")
        return self.scaler

    def save_scaler(self, filepath: str):
        if self.scaler:
            self.scaler.save(filepath)

    def load_scaler(self, filepath: str):
        if self.scaler:
            self.scaler.load(filepath)

    def __getitem__(self, idx: int):
        record_path = self.records[idx]
        # ---------------- A. NPY 极速模式 (Memory Mapped) ----------------
        if self.load_features:
            # 1. 读取 Lowdim (qpos, action)
            # mmap_mode='r' 是核心：只建立映射，不全读入内存
            qpos_path = os.path.join(record_path, 'qpos.npy')
            action_path = os.path.join(record_path, 'action.npy')
            
            # 使用 np.array() 将 mmap 对象转为内存中的 numpy 数组
            # 虽然这里发生了拷贝，但只针对这一条轨迹，速度极快且不占文件锁
            qpos = np.array(np.load(qpos_path, mmap_mode='r')) 
            act = np.array(np.load(action_path, mmap_mode='r'))
            
            # 2. 读取 特征 (Feature)
            rgb_data = {}
            for cam in self.selected_cameras:
                feat_path = os.path.join(record_path, f'{cam}.npy')
                # 同样的 mmap 技巧
                # [L, 1024, Hf, Wf]
                feat_mmap = np.load(feat_path, mmap_mode='r')
                
                # 转 Tensor (触发实际 IO，但无锁)
                rgb_data[cam] = torch.from_numpy(np.array(feat_mmap)) # Float32

        # ---------------- B. H5 兼容模式 (原始图片) ----------------
        else:
            # 需要 import h5py，仅在此分支使用，避免全局依赖
            import h5py 
            with h5py.File(record_path, 'r') as root:
                qpos = root['/observations/qpos'][()] 
                act = root['/action'][()]
                
                rgb_data = {}
                for cam in self.selected_cameras:
                    imgs = root[f'observations/images/{cam}'][()]
                    processed_imgs = []
                    for i in range(len(imgs)):
                        processed_imgs.append(self.transform_func(imgs[i]))
                    rgb_data[cam] = torch.stack(processed_imgs)

        L = qpos.shape[0]
        
        # 1. 构造 Lowdim Raw Dict
        # 注意: 确保维度至少是 [L, 1]
        def to_torch(arr):
            t = torch.from_numpy(arr).float()
            if t.dim() == 1: t = t.unsqueeze(1)
            return t

        lowdim_raw = {
            'agl_1': to_torch(qpos[:, 0]), 'agl_2': to_torch(qpos[:, 1]), 'agl_3': to_torch(qpos[:, 2]),
            'agl_4': to_torch(qpos[:, 3]), 'agl_5': to_torch(qpos[:, 4]), 'agl_6': to_torch(qpos[:, 5]),
            'gripper_pos': to_torch(qpos[:, 6]),
            'agl2_1': to_torch(qpos[:, 7]), 'agl2_2': to_torch(qpos[:, 8]), 'agl2_3': to_torch(qpos[:, 9]),
            'agl2_4': to_torch(qpos[:, 10]), 'agl2_5': to_torch(qpos[:, 11]), 'agl2_6': to_torch(qpos[:, 12]),
            'gripper_pos2': to_torch(qpos[:, 13])
        }

        # 2. 构造 Future Actions [L, 16, D]
        # Vectorized padding and striding
        if self.future_steps > 0:
            act_padded = np.pad(act, ((0, self.future_steps-1), (0,0)), mode='edge')
            shape = (L, self.future_steps, act.shape[1])
            strides = (act_padded.strides[0], act_padded.strides[0], act_padded.strides[1])
            act_chunks = np.lib.stride_tricks.as_strided(act_padded, shape=shape, strides=strides)
        else:
            act_chunks = act[:, np.newaxis, :] # dummy

        # 把 Action 放入 raw dict 以便归一化
        # 拆分 act_chunks 到各个关节 key
        # act_chunks: [L, 16, 14]
        lowdim_raw.update({
            'agl_1_act': torch.from_numpy(act_chunks[:,:,0:1]).float(),
            'agl_2_act': torch.from_numpy(act_chunks[:,:,1:2]).float(),
            'agl_3_act': torch.from_numpy(act_chunks[:,:,2:3]).float(),
            'agl_4_act': torch.from_numpy(act_chunks[:,:,3:4]).float(),
            'agl_5_act': torch.from_numpy(act_chunks[:,:,4:5]).float(),
            'agl_6_act': torch.from_numpy(act_chunks[:,:,5:6]).float(),
            'gripper_act': torch.from_numpy(act_chunks[:,:,6:7]).float(),
            'agl2_1_act': torch.from_numpy(act_chunks[:,:,7:8]).float(),
            'agl2_2_act': torch.from_numpy(act_chunks[:,:,8:9]).float(),
            'agl2_3_act': torch.from_numpy(act_chunks[:,:,9:10]).float(),
            'agl2_4_act': torch.from_numpy(act_chunks[:,:,10:11]).float(),
            'agl2_5_act': torch.from_numpy(act_chunks[:,:,11:12]).float(),
            'agl2_6_act': torch.from_numpy(act_chunks[:,:,12:13]).float(),
            'gripper_act2': torch.from_numpy(act_chunks[:,:,13:14]).float(),
        })

        if self.fitting:
            return {'lowdim_raw': lowdim_raw}

        # 3. 归一化
        lowdim_norm = self.scaler.normalize(lowdim_raw)
        
        # 4. 拼装 Observation [L, 14]
        obs_keys = ['agl_1','agl_2','agl_3','agl_4','agl_5','agl_6','gripper_pos',
                    'agl2_1','agl2_2','agl2_3','agl2_4','agl2_5','agl2_6','gripper_pos2']
        obs_list = [lowdim_norm[k] for k in obs_keys]
        obs_tensor = torch.cat(obs_list, dim=1)

        # 5. 拼装 Action Label [L, 16, 14]
        act_keys = ['agl_1_act','agl_2_act','agl_3_act','agl_4_act','agl_5_act','agl_6_act','gripper_act',
                    'agl2_1_act','agl2_2_act','agl2_3_act','agl2_4_act','agl2_5_act','agl2_6_act','gripper_act2']
        act_list = [lowdim_norm[k] for k in act_keys]
        act_tensor = torch.cat(act_list, dim=2)

        return {
            'obs': obs_tensor,       # [L, 14]
            'actions': act_tensor,   # [L, 16, 14]
            'rgb': rgb_data,         # {cam: [L, 3, H, W]}
            'length': L
        }

def parallel_collate_fn(batch):
    """
    CPU 端 Collate：
    把图片保持为 List[Tensor] 形式，防止在这里做巨大的 Concat 导致 RAM 抖动。
    Observation 和 Action 做 Padding。
    """
    batch_size = len(batch)
    lengths = [b['length'] for b in batch]
    max_len = max(lengths)
    
    obs_dim = batch[0]['obs'].shape[1]
    act_shape = batch[0]['actions'].shape[1:] # (16, 14)
    
    padded_obs = torch.zeros(batch_size, max_len, obs_dim)
    padded_actions = torch.zeros(batch_size, max_len, *act_shape)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool) # True = Pad
    
    # 图片：{cam: [Tensor(L1), Tensor(L2), ...]}
    # 我们不在这里 Pad 图片，因为图片太大了。
    rgb_batch = {cam: [] for cam in batch[0]['rgb'].keys()}
    
    for i, item in enumerate(batch):
        l = item['length']
        padded_obs[i, :l] = item['obs']
        padded_actions[i, :l] = item['actions']
        mask[i, :l] = False
        
        for cam in item['rgb']:
            rgb_batch[cam].append(item['rgb'][cam]) # List of [L, 3, H, W]
            
    return {
        'obs': padded_obs,
        'actions': padded_actions,
        'mask': mask,
        'rgb_batch': rgb_batch, # List of Tensors
        'lengths': lengths
    }

def main():
    # 测试代码
    root_dir = "/home/sutai/data2/ZYL/ACT_data" # 修改这里
    if not os.path.exists(root_dir):
        print("Path not found, skipping fit.")
        return

    ds = MambaTrajectoryDataset(root_dir, mode='train', load_features=True)
    # 1. Fit
    ds.fit_scaler(num_workers=4)
    ds.save_scaler('scaler_params.pth')
    
    # 2. Load & Check
    ds.load_scaler('scaler_params.pth')
    item = ds[0]
    print("Obs:", item['obs'].shape)
    print("Act:", item['actions'].shape)
    if 'top' in item['rgb']:
        print("RGB top:", item['rgb']['top'].shape)

if __name__ == "__main__":
    main()