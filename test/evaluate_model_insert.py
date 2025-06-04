import os
from inference_M import MyInferenceModel
import numpy as np
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from sim_env import make_sim_env, BOX_POSE
from visualize_episodes import save_videos
from train.mamba_policy import MambaPolicy, MambaConfig
from train.scaler_M import Scaler
from train.M_dataset import MambaSequenceDataset
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


def get_image(ts, camera_names):
    curr_images = {}  # 创建一个空字典用于存储图像数据
    for cam_name in camera_names:
        if cam_name in ts.observation['images']:
            # 获取图像并转换形状为 (c, h, w)
            # img_bgr = ts.observation['images'][cam_name]
            # img_bgr = np.transpose(img_bgr, (2, 0, 1))
            # img_bgr = np.expand_dims(img_bgr, axis=0)
            curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
            # 将图像数据加入字典，键为相机名称，值为图像数据
            curr_images[cam_name] = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        else:
            print(f"Warning: Camera '{cam_name}' not found in images.")
    return curr_images


scaler_path = 'scaler_params.pth'  # your own path
checkpoint = 'last.ckpt'  # your own path
results_dir = 'video'  # your own path
#  初始化推理模型
config = MambaConfig()
config.camera_names = ['top']
config.embed_dim = 2048
config.lowdim_dim = 14
config.d_model = 2048
config.action_dim = 14
config.sum_camera_feats = False
config.num_blocks = 4
lowdim_dict = {
    'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
    'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
    'gripper_pos': 1,
    'gripper_pos2': 1,
    'agl_1_act': (16, 1), 'agl_2_act': (16, 1), 'agl_3_act': (16, 1),
    'agl_4_act': (16, 1), 'agl_5_act': (16, 1), 'agl_6_act': (16, 1),
    'agl2_1_act': (16, 1), 'agl2_2_act': (16, 1), 'agl2_3_act': (16, 1),
    'agl2_4_act': (16, 1), 'agl2_5_act': (16, 1), 'agl2_6_act': (16, 1),
    'gripper_act': (16, 1), 'gripper_act2': (16, 1)
}
# 加载 Scaler 参数
# 初始化 Scaler 并加载参数
scaler = Scaler(lowdim_dict=lowdim_dict)
scaler.load(scaler_path)
print("Scaler loaded.")

# 加载虚拟环境
task_name = 'sim_insertion'
env = make_sim_env(task_name)
env_max_reward = env.task.max_reward
query_frequency = 1  # 每个时间步上都进行一次查询
max_timesteps = 400  # 最大任务时间长度
max_timesteps = int(max_timesteps * 1)
num_queries = 16  # 预测步长
num_rollouts = 50  # 评估的回合数
episode_returns = []  # 用于存储每个回合的总回报
highest_rewards = []  # 用于存储每个回合的最高奖励
state_dim = 14
onscreen_cam = 'angle'
DT = 0.02
camera_names = ['top']
infer_model = MyInferenceModel(
        checkpoint_path=checkpoint,
        lowdim_dict=lowdim_dict,
        config=config
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
print("Warmup inference model...")
# 预热推理模型
rgb_dummy = {'top': torch.zeros(1, 3, 480, 640).to(infer_model.device)}
lowdim = torch.zeros(1, 14).cuda()
_ = infer_model(lowdim, rgb_dummy)
infer_model.reset_hiddens()
print("reset hiddens finish")
print("warmup model finish")
for rollout_id in range(num_rollouts):
    rollout_id += 0
    # 增加FIFO加权动作队列
    all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()
    BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # 生成14维数组随机初始位置
    ts = env.reset()
    # onscreen render
    ax = plt.subplot()
    plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
    plt.ion()
    infer_model.reset_hiddens()
    image_list = []  # for visualization
    qpos_list = []
    target_qpos_list = []
    rewards = []
    temporal_agg = True  # 是否使用时间聚合
    # 增加渲染
    ax = plt.subplot()
    plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
    plt.ion()

    with torch.inference_mode():
        for t in range(max_timesteps):
            # update onscreen render and wait for DT
            image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            plt_img.set_data(image)
            plt.pause(DT)
            # process previous timestep to get qpos and image_list
            obs = ts.observation
            if 'images' in obs:
                image_list.append(obs['images'])
            else:
                image_list.append({'main': obs['image']})
            qpos_numpy = np.array(obs['qpos'])
            agl_1 = torch.from_numpy(qpos_numpy[0:1]).to(infer_model.device)
            agl_2 = torch.from_numpy(qpos_numpy[1:2]).to(infer_model.device)
            agl_3 = torch.from_numpy(qpos_numpy[2:3]).to(infer_model.device)
            agl_4 = torch.from_numpy(qpos_numpy[3:4]).to(infer_model.device)
            agl_5 = torch.from_numpy(qpos_numpy[4:5]).to(infer_model.device)
            agl_6 = torch.from_numpy(qpos_numpy[5:6]).to(infer_model.device)
            gripper_pos = torch.from_numpy(qpos_numpy[6:7]).to(infer_model.device)
            agl2_1 = torch.from_numpy(qpos_numpy[7:8]).to(infer_model.device)
            agl2_2 = torch.from_numpy(qpos_numpy[8:9]).to(infer_model.device)
            agl2_3 = torch.from_numpy(qpos_numpy[9:10]).to(infer_model.device)
            agl2_4 = torch.from_numpy(qpos_numpy[10:11]).to(infer_model.device)
            agl2_5 = torch.from_numpy(qpos_numpy[11:12]).to(infer_model.device)
            agl2_6 = torch.from_numpy(qpos_numpy[12:13]).to(infer_model.device)
            gripper_pos2 = torch.from_numpy(qpos_numpy[13:14]).to(infer_model.device)

            # 归一化低维数据
            lowdim_arm1_norm = infer_model.scaler.normalize(
                {'agl_1': agl_1, 'agl_2': agl_2, 'agl_3': agl_3, 'agl_4': agl_4,
                 'agl_5': agl_5, 'agl_6': agl_6, 'gripper_pos': gripper_pos})
            lowdim_arm1_norm = torch.cat([lowdim_arm1_norm['agl_1'].unsqueeze(1),
                                          lowdim_arm1_norm['agl_2'].unsqueeze(1),
                                          lowdim_arm1_norm['agl_3'].unsqueeze(1),
                                          lowdim_arm1_norm['agl_4'].unsqueeze(1),
                                          lowdim_arm1_norm['agl_5'].unsqueeze(1),
                                          lowdim_arm1_norm['agl_6'].unsqueeze(1),
                                          lowdim_arm1_norm['gripper_pos'].unsqueeze(1)], dim=1)  # [1,7]
            lowdim_arm2_norm = infer_model.scaler.normalize(
                {'agl2_1': agl2_1, 'agl2_2': agl2_2, 'agl2_3': agl2_3, 'agl2_4': agl2_4,
                 'agl2_5': agl2_5, 'agl2_6': agl2_6, 'gripper_pos2': gripper_pos2})
            lowdim_arm2_norm = torch.cat([lowdim_arm2_norm['agl2_1'].unsqueeze(1),
                                          lowdim_arm2_norm['agl2_2'].unsqueeze(1),
                                          lowdim_arm2_norm['agl2_3'].unsqueeze(1),
                                          lowdim_arm2_norm['agl2_4'].unsqueeze(1),
                                          lowdim_arm2_norm['agl2_5'].unsqueeze(1),
                                          lowdim_arm2_norm['agl2_6'].unsqueeze(1),
                                          lowdim_arm2_norm['gripper_pos2'].unsqueeze(1)], dim=1)  # [1,7]

            lowdim_norm = torch.cat([lowdim_arm1_norm, lowdim_arm2_norm], dim=1).float()
            curr_image = get_image(ts, camera_names)
            # 推理
            if t % query_frequency == 0:
                pred_action = infer_model(lowdim_norm, curr_image)
                pred_denorm = infer_model.denormalize(pred_action)
                a_hat = pred_denorm.view(1, 16, 14)
                all_actions = a_hat
            if temporal_agg:
                all_time_actions[[t], t:t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % query_frequency]

            target_qpos = raw_action.squeeze(0).cpu().numpy()
            ts = env.step(target_qpos)
            # for visualization
            qpos_list.append(qpos_numpy)
            target_qpos_list.append(target_qpos)
            rewards.append(ts.reward)
        plt.close()
        pass
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards is not None])
    episode_returns.append(episode_return)
    episode_highest_reward = np.max(rewards)
    highest_rewards.append(episode_highest_reward)
    print(
        f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward == env_max_reward}')
    save_videos(image_list, DT, video_path=os.path.join(results_dir, f'video{rollout_id}.mp4'))

success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
avg_return = np.mean(episode_returns)
summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
for r in range(env_max_reward + 1):
    more_or_equal_r = (np.array(highest_rewards) >= r).sum()
    more_or_equal_r_rate = more_or_equal_r / num_rollouts
    summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n'
print(summary_str)
print(f'成功率：{success_rate}, 平均回报：{avg_return}')
# # save success rate to txt
# result_file_name = 'result_' + '.txt'
# with open(os.path.join(results_dir, result_file_name), 'w') as f:
#     f.write(summary_str)
#     f.write(repr(episode_returns))
#     f.write('\n\n')
#     f.write(repr(highest_rewards))
# print(f'成功率：{success_rate}, 平均回报：{avg_return}')









