import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from mamba_policy_par import FrozenDinov2
from M_dataset_par import get_image_transform


# ================= 配置 =================
DATA_ROOT = "/home/sutai/data"
OUTPUT_ROOT = "/home/sutai/data"
CAMERA_NAMES = ['top']
BATCH_SIZE = 64 # NPY 写入很快，可以加大 Batch
DEVICE = "cuda"
RESIZE_HW = (640, 480) 
# =======================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_for_split(mode):
    input_dir = os.path.join(DATA_ROOT, mode)
    output_dir = os.path.join(OUTPUT_ROOT, mode)
    ensure_dir(output_dir)

    print(f"Loading DINOv2 for {mode}...")
    backbone = FrozenDinov2(layer_index=-4).to(DEVICE)
    backbone.eval()
    transform_func = get_image_transform(RESIZE_HW)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.h5') or f.endswith('.hdf5')])
    
    for filename in tqdm(files, desc=f"Processing {mode}"):
        in_path = os.path.join(input_dir, filename)
        # 名字改为 .npy，去掉 .h5 后缀
        name_no_ext = os.path.splitext(filename)[0]
        
        # 我们需要存多个相机的特征，为了方便，每个轨迹存为一个 .npz 或者 多个 .npy
        # 为了极致速度，推荐每个相机存一个 .npy，放在轨迹名文件夹下
        # 结构: OUTPUT/train/episode_0/top.npy
        
        traj_dir = os.path.join(output_dir, name_no_ext)
        ensure_dir(traj_dir)
        
        # 检查是否已完成 (查看 top.npy 是否存在)
        if os.path.exists(os.path.join(traj_dir, f"{CAMERA_NAMES[0]}.npy")):
            continue

        with h5py.File(in_path, 'r') as f:
            if 'observations/qpos' not in f: continue
            data_len = f['observations/qpos'].shape[0]
            
            # 1. 保存 Action/Qpos 为 npy (极快读取)
            if 'action' in f:
                np.save(os.path.join(traj_dir, 'action.npy'), f['action'][:])
            if 'observations/qpos' in f:
                np.save(os.path.join(traj_dir, 'qpos.npy'), f['observations/qpos'][:])
            
            # 2. 提取特征
            for cam_name in CAMERA_NAMES:
                img_path = f'observations/images/{cam_name}'
                if img_path not in f: continue
                
                imgs_dataset = f[img_path]
                
                # 预分配内存 (Numpy Array)
                # 先跑一次 dummy 确定维度
                if 'feat_dim' not in locals():
                    dummy = torch.zeros(1, 3, RESIZE_HW[1], RESIZE_HW[0]).to(DEVICE)
                    with torch.no_grad():
                        out = backbone(dummy)
                    _, C, Hf, Wf = out.shape
                    feat_dim = (data_len, C, Hf, Wf)
                
                # 创建一个巨大的 numpy 数组来存特征
                all_features = np.zeros(feat_dim, dtype=np.float32)
                
                for i in range(0, data_len, BATCH_SIZE):
                    end = min(i + BATCH_SIZE, data_len)
                    batch_imgs = imgs_dataset[i:end] # Read HDF5
                    
                    batch_tensor = torch.stack([transform_func(img) for img in batch_imgs]).to(DEVICE)
                    
                    with torch.no_grad():
                        features = backbone(batch_tensor)
                    
                    all_features[i:end] = features.cpu().numpy()
                
                # 保存为 npy
                np.save(os.path.join(traj_dir, f'{cam_name}.npy'), all_features)

if __name__ == "__main__":
    extract_for_split("train")
    extract_for_split("test")