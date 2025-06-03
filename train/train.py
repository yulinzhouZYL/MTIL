import os
import torch
from torch import nn
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from metric_M import my_Metric
from scaler_M import Scaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from M_dataset import MambaSequenceDataset
from mamba_policy import MambaPolicy, MambaConfig  # mamba2 + policy

class LitMambaModel(pl.LightningModule):
    def __init__(self, config: MambaConfig, scaler: Scaler, future_steps: int = 16):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.metric = my_Metric()
        self.prev_traj_idx = -1  # 初始化上一个轨迹索引
        self.future_steps = future_steps
        # Register train and val sequence_loss as buffers
        self.train_sequence_loss = 0.0
        self.val_sequence_loss = 0.0
        # 1) 构建 MambaPolicy
        print("Starting training...")
        self.policy = MambaPolicy(
            camera_names = config.camera_names,
            backbone = config.backbone,
            pretrained_backbone = config.pretrained_backbone,
            freeze_backbone = config.freeze_backbone,
            embed_dim = config.embed_dim,
            lowdim_dim = 14,
            d_model = config.d_model,
            action_dim = 14,   # pose_act(12) + gripper_act(2) = 14
            sum_camera_feats = config.sum_camera_feats,
            num_blocks = config.num_blocks,
            future_steps=future_steps,
            img_size = config.img_size,
            mamba_cfg = {
                'd_state': config.d_state,
                'd_conv': config.d_conv,
                'expand': config.expand,
                'headdim': config.headdim,
                'activation': config.activation,
                'use_mem_eff_path': config.use_mem_eff_path,
            }
        )

        print("Model initialized.")
        # 2) scaler
        self.scaler = scaler
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy.to(device)
        self.scaler.to(device)
        # 3) 其他超参
        self.lr = 2e-4
        self.weight_decay = 5e-4

        self.train_total_loss = 0.0  # 记录整个 epoch 的训练损失总和
        self.train_total_steps = 0  # 记录整个 epoch 的训练步数
        self.val_total_loss = 0.0  # 记录整个 epoch 的验证损失总和
        self.val_total_steps = 0  # 记录整个 epoch 的验证步数

        # 4) 禁用自动优化
        self.automatic_optimization = False  # <--- 禁用自动优化
        self.std_agl_1 = 0.0036
        self.std_agl_2 = 0.5280
        self.std_agl_3 = 0.1980
        self.std_agl_4 = 0.0164
        self.std_agl_5 = 0.3592
        self.std_agl_6 = 0.5998
        self.std_agl2_1 = 0.1084
        self.std_agl2_2 = 0.5019
        self.std_agl2_3 = 0.4448
        self.std_agl2_4 = 0.1414
        self.std_agl2_5 = 0.3066
        self.std_agl2_6 = 0.2251
        self.std_grip1 = 0.2553
        self.std_grip2 = 0.2475


    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()  # 获取优化器

        rgb = batch['rgb']  # [B=1,C,H,W]
        lowdim = batch['lowdim']  # [B=1,D]
        traj_idx = batch['traj_idx'].item()  # 当前样本的轨迹索引

        # ==========  (optional, 对 lowdim 加随机扰动) ==========
        noise_agl = 0.02  # ~ 1 mm
        noise_gripper = 0.02
        noise_scale_agl_1 = noise_agl * self.std_agl_1
        noise_scale_agl_2 = noise_agl * self.std_agl_2
        noise_scale_agl_3 = noise_agl * self.std_agl_3
        noise_scale_agl_4 = noise_agl * self.std_agl_4
        noise_scale_agl_5 = noise_agl * self.std_agl_5
        noise_scale_agl_6 = noise_agl * self.std_agl_6
        noise_scale_agl2_1 = noise_agl * self.std_agl2_1
        noise_scale_agl2_2 = noise_agl * self.std_agl2_2
        noise_scale_agl2_3 = noise_agl * self.std_agl2_3
        noise_scale_agl2_4 = noise_agl * self.std_agl2_4
        noise_scale_agl2_5 = noise_agl * self.std_agl2_5
        noise_scale_agl2_6 = noise_agl * self.std_agl2_6
        noise_scale_gripper = noise_gripper * self.std_grip1
        noise_scale_gripper2 = noise_gripper * self.std_grip2
        # noise_scale_gripper1 = 0.3 * self.std_grip1
        # noise_scale_gripper2 = 0.3 * self.std_grip2
        # 仅在训练中加扰动, validation不加
        with torch.no_grad():
            # pose(9): x,y,z, rx,ry,rz, ...
            if 'agl_1' in lowdim:
                lowdim['agl_1'] += torch.randn_like(lowdim['agl_1']) * noise_scale_agl_1
            if 'agl_2' in lowdim:
                lowdim['agl_2'] += torch.randn_like(lowdim['agl_2']) * noise_scale_agl_2
            if 'agl_3' in lowdim:
                lowdim['agl_3'] += torch.randn_like(lowdim['agl_3']) * noise_scale_agl_3
            if 'agl_4' in lowdim:
                lowdim['agl_4'] += torch.randn_like(lowdim['agl_4']) * noise_scale_agl_4
            if 'agl_5' in lowdim:
                lowdim['agl_5'] += torch.randn_like(lowdim['agl_5']) * noise_scale_agl_5
            if 'agl_6' in lowdim:
                lowdim['agl_6'] += torch.randn_like(lowdim['agl_6']) * noise_scale_agl_6
            if 'agl2_1' in lowdim:
                lowdim['agl2_1'] += torch.randn_like(lowdim['agl2_1']) * noise_scale_agl2_1
            if 'agl2_2' in lowdim:
                lowdim['agl2_2'] += torch.randn_like(lowdim['agl2_2']) * noise_scale_agl2_2
            if 'agl2_3' in lowdim:
                lowdim['agl2_3'] += torch.randn_like(lowdim['agl2_3']) * noise_scale_agl2_3
            if 'agl2_4' in lowdim:
                lowdim['agl2_4'] += torch.randn_like(lowdim['agl2_4']) * noise_scale_agl2_4
            if 'agl2_5' in lowdim:
                lowdim['agl2_5'] += torch.randn_like(lowdim['agl2_5']) * noise_scale_agl2_5
            if 'agl2_6' in lowdim:
                lowdim['agl2_6'] += torch.randn_like(lowdim['agl2_6']) * noise_scale_agl2_6

        #     if 'gripper_pos' in lowdim:
        #         lowdim['gripper_pos'] += torch.randn_like(lowdim['gripper_pos']) * noise_scale_gripper
        #     if 'gripper_pos2' in lowdim:
        #         lowdim['gripper_pos2'] += torch.randn_like(lowdim['gripper_pos2']) * noise_scale_gripper2
        # # ========== (对 lowdim 加随机扰动) ==========

        # 检测是否是新轨迹
        if traj_idx != self.prev_traj_idx and self.prev_traj_idx != -1:
            # 执行优化步骤
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            optimizer.zero_grad()
            # 记录累积的训练损失
            self.log("train_loss", self.train_sequence_loss, prog_bar=True, sync_dist=False, batch_size=1)
            # 重置累积损失
            self.train_sequence_loss = 0.0
            # 初始化新的轨迹隐状态
            self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)
            self.prev_traj_idx = traj_idx  # 更新轨迹索引

        elif self.prev_traj_idx == -1:
            # 第一个轨迹，初始化隐藏状态
            self.hiddens= self.policy.init_hidden_states(batch_size=1, device=self.device)
            self.prev_traj_idx = traj_idx

        # 数据预处理
        for cam in rgb:
            x = rgb[cam]  # shape [B, C, H, W]
            rgb[cam] = x

        # lowdim归一化
        lowdim = self.scaler.normalize(lowdim)

        agl_1_arm = lowdim['agl_1']
        agl_2_arm = lowdim['agl_2']
        agl_3_arm = lowdim['agl_3']
        agl_4_arm = lowdim['agl_4']
        agl_5_arm = lowdim['agl_5']
        agl_6_arm = lowdim['agl_6']
        gripper_arm1 = lowdim['gripper_pos']
        agl2_1_arm = lowdim['agl2_1']
        agl2_2_arm = lowdim['agl2_2']
        agl2_3_arm = lowdim['agl2_3']
        agl2_4_arm = lowdim['agl2_4']
        agl2_5_arm = lowdim['agl2_5']
        agl2_6_arm = lowdim['agl2_6']
        gripper_arm2 = lowdim['gripper_pos2']
        concat_lowdim = torch.cat([agl_1_arm,agl_2_arm,agl_3_arm,agl_4_arm,agl_5_arm,agl_6_arm,gripper_arm1,
                                   agl2_1_arm,agl2_2_arm,agl2_3_arm,agl2_4_arm,agl2_5_arm,agl2_6_arm,gripper_arm2], dim=1)

        # 前向传播: 在 policy上 step
        pred_action, self.hiddens = self.policy.step(concat_lowdim, rgb, self.hiddens)

        # 隐状态断开计算图
        self.hiddens = [
            ((c.detach() if c is not None else None), (s.detach() if s is not None else None))
            for (c, s) in self.hiddens
        ]
        # 计算损失
        actions = torch.cat([
            lowdim['agl_1_act'],lowdim['agl_2_act'],lowdim['agl_3_act'],
            lowdim['agl_4_act'],lowdim['agl_5_act'],lowdim['agl_6_act'],
            lowdim['gripper_act'],
            lowdim['agl2_1_act'],lowdim['agl2_2_act'],lowdim['agl2_3_act'],
            lowdim['agl2_4_act'],lowdim['agl2_5_act'],lowdim['agl2_6_act'],
            lowdim['gripper_act2']
        ], dim=2)  # => [B,16,14]
        loss = F.mse_loss(pred_action, actions)

        # 反向传播
        self.manual_backward(loss)

        # 累积损失
        self.train_sequence_loss += loss.item()

        #  累积 epoch 级损失
        self.train_total_loss += loss.item()  # 整个 epoch 的损失总和
        self.train_total_steps += 1  # 整个 epoch 的总步数

        # 可选：清理缓存
        if batch_idx % 1000 == 0:
            torch.cuda.empty_cache()

        return loss  # 返回当前步骤的损失

    def validation_step(self, batch, batch_idx):

        rgb = batch['rgb']  # [B=1,C,H,W]
        lowdim = batch['lowdim']  # [B=1,D]
        traj_idx = batch['traj_idx'].item()  # 当前样本的轨迹索引

        # 检测是否是新轨迹
        if traj_idx != self.prev_traj_idx:
            if self.prev_traj_idx != -1 and self.val_sequence_loss > 0.0:
                # 记录验证损失
                self.log("val_loss", self.val_sequence_loss, prog_bar=True, sync_dist=False, batch_size=1)
                # 重置累积损失
                self.val_sequence_loss = 0.0
            # 初始化新的隐藏状态
            self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)
            self.prev_traj_idx = traj_idx  # 更新轨迹索引

        # 数据预处理
        for cam in rgb:
            x = rgb[cam]  # shape [B, C, H, W]
            rgb[cam] = x

        # lowdim归一化
        lowdim = self.scaler.normalize(lowdim)

        concat_lowdim = torch.cat([lowdim['agl_1'],lowdim['agl_2'],lowdim['agl_3'],lowdim['agl_4'],
                                 lowdim['agl_5'],lowdim['agl_6'],lowdim['gripper_pos'],
                                   lowdim['agl2_1'], lowdim['agl2_2'], lowdim['agl2_3'], lowdim['agl2_4'],
                                   lowdim['agl2_5'], lowdim['agl2_6'], lowdim['gripper_pos2']], dim=1)

        pred_action, self.hiddens= self.policy.step(concat_lowdim, rgb, self.hiddens)

        self.hiddens = [
            ((c.detach() if c is not None else None), (s.detach() if s is not None else None))
            for (c, s) in self.hiddens
        ]

        actions = torch.cat([
            lowdim['agl_1_act'],lowdim['agl_2_act'],lowdim['agl_3_act'],
            lowdim['agl_4_act'],lowdim['agl_5_act'],lowdim['agl_6_act'],
            lowdim['gripper_act'],
            lowdim['agl2_1_act'],lowdim['agl2_2_act'],lowdim['agl2_3_act'],
            lowdim['agl2_4_act'],lowdim['agl2_5_act'],lowdim['agl2_6_act'],
            lowdim['gripper_act2']
        ], dim=2)
        loss = F.mse_loss(pred_action, actions)

        # 累积验证损失
        self.val_sequence_loss += loss.item()
        self.val_total_loss += loss.item()
        self.val_total_steps += 1

        # 反归一化动作，用于计算真实差距
        pred_action = self.denormalize(pred_action)
        actions = self.denormalize(actions)

        self.metric.update(pred_action, actions)

        return loss  # 返回当前步骤的损失

    def on_train_epoch_end(self):
        optimizer = self.optimizers()  # 获取优化器

        # 处理最后一条轨迹的累积损失
        if self.train_sequence_loss > 0.0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)  # 梯度裁剪（可选）
            optimizer.step()
            optimizer.zero_grad()
            self.log("train_loss", self.train_sequence_loss, prog_bar=True, sync_dist=False, batch_size=1)
            self.train_sequence_loss = 0.0  # 重置累积损失
            self.lr_scheduler_obj.step()  # 学习率调度器步进

        self.log("train_epoch_loss",
                 self.train_total_loss / self.train_total_steps if self.train_total_steps > 0 else 0.0,
                 prog_bar=True, sync_dist=True)
        self.train_total_loss = 0.0  # 重置整个 epoch 的损失总和
        self.train_total_steps = 0  # 重置整个 epoch 的总步数

    def denormalize(self, actions):
        # [B,20]: arm1 => [0:10], arm2 => [10:20]
        # arm1 => pose_act(9)+gripper_act(1)
        # arm2 => pose_act2(9)+gripper_act2(1)
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

     # cosine 优化器，平滑降低学习率
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.policy.parameters()),
            lr=self.lr,  # 使用修改后的学习率
            weight_decay=self.weight_decay
        )
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=200,  # 余弦周期共多少个epoch
            eta_min=0.5e-6  # 最小学习率
        )
        scheduler = {
            'scheduler': scheduler_obj,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        self.lr_scheduler_obj = scheduler_obj  # 存储实际的调度器对象
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        # 这里 optimizer 和 scheduler 已经创建完毕，可以安全访问
        if hasattr(self.trainer, 'optimizers') and len(self.trainer.optimizers) > 0:
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = self.lr  # 将学习率设置为 1e-4
                current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
                print(f"Current Learning Rate: {current_lr}")

    def on_validation_epoch_end(self):
        # 记录并重置验证指标
        self.log_dict(self.metric.compute(), prog_bar=True, sync_dist=True)
        self.metric.reset()

        # 获取当前 epoch 的验证损失
        val_loss = self.trainer.callback_metrics.get("val_loss")

        if val_loss is not None:
            # 使用存储的调度器对象
            # self.lr_scheduler_obj.step(val_loss)
            # print(f"Scheduler stepped with val_loss: {val_loss.item()}")

            # 打印当前学习率以确认调度器是否生效
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr}")

        self.log("val_epoch_loss", self.val_total_loss / self.val_total_steps if self.val_total_steps > 0 else 0.0,
                 prog_bar=True, sync_dist=True)
        self.val_total_loss = 0.0  # 重置整个 epoch 的验证损失总和
        self.val_total_steps = 0  # 重置整个 epoch 的验证步数

#  main
def main():
    seed_everything(42)

    # 1)  config
    config = MambaConfig()
    config.camera_names = ['top']
    config.freeze_backbone = True
    config.embed_dim = 2048
    config.lowdim_dim = 14
    config.d_model = 2048
    config.action_dim = 14
    config.sum_camera_feats = False
    config.num_blocks = 4
    config.img_size = (640,480)

    # 2) 构造 Dataset
    train_dataset = MambaSequenceDataset(
        root_dir="data100",  # put your own data path here
        mode="train",
        resize_hw=(640, 480),
        use_pose10d=True,
        selected_cameras=config.camera_names
    )
    val_dataset = MambaSequenceDataset(
        root_dir="data100",
        mode="test",
        resize_hw=(640, 480),
        use_pose10d=True,
        selected_cameras=config.camera_names
    )

    # 3) Initialize and fit the scaler
    # Define lowdim_dict based on the dataset's lowdim_keys
    lowdim_dict = {
        'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
        'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
        'gripper_pos': 1,
        'gripper_pos2': 1,
        'agl_1_act': (16,1), 'agl_2_act': (16,1), 'agl_3_act': (16,1),
        'agl_4_act': (16,1), 'agl_5_act': (16,1), 'agl_6_act': (16,1),
        'agl2_1_act': (16,1), 'agl2_2_act': (16,1), 'agl2_3_act': (16,1),
        'agl2_4_act': (16,1), 'agl2_5_act': (16,1), 'agl2_6_act': (16,1),
        'gripper_act':(16,1), 'gripper_act2':(16,1)
    }

    # 加载归一化参数
    scaler = Scaler(lowdim_dict=lowdim_dict)
    scaler.load('scaler_params.pth')  # put your own scaler data path here

    # Assign loaded scaler to train and val datasets
    train_dataset.scaler = scaler
    val_dataset.scaler = scaler

    # 4) 构造 DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # 设置为1，保证整条轨迹可以完整加载。
        shuffle=False,  # 不打乱顺序，适合时序训练模式
        num_workers=20,  # 根据 CPU 核心数量设置为
        pin_memory=True,  # 若使用 GPU 加速，开启 pin_memory 提升数据加载性能
        drop_last=False,  # 不丢弃最后一个 batch，即使它不满 batch_size
        collate_fn=None,  # 默认拼接，若需要 padding 时再定义自定义 collate_fn
        prefetch_factor=4,  # 每个 worker 预取2个 batch 提升效率
        persistent_workers=True,  # 保持 worker 持续运行（加速多次 epoch 数据加载）
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 设置为1，保证整条轨迹可以完整加载
        shuffle=False,  # 不打乱顺序，适合测试模式
        num_workers=20,  # 根据 CPU 核心数量，通常设置为 4~8
        pin_memory=True,  # 若使用 GPU 加速，开启 pin_memory 提升数据加载性能
        drop_last=False,  # 不丢弃最后一个 batch，即使它不满 batch_size
        collate_fn=None,  # 默认拼接，若需要 padding 时再定义自定义 collate_fn
        prefetch_factor=4,  # 每个 worker 预取2个 batch 提升效率
        persistent_workers=True,  # 保持 worker 持续运行（加速多次 epoch 数据加载）
    )

    # 5) 构造LightningModule
    lit_model = LitMambaModel(config, scaler=scaler)
    checkpoint_path = ('last.ckpt')  # put your own ckpt path here
    checkpoint = torch.load(checkpoint_path)
    lit_model.load_state_dict(checkpoint['state_dict'], strict=True)

    # 6) trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb = ModelCheckpoint(monitor='val_epoch_loss', mode='min',
                              save_last=True,
                              save_top_k=5,  # 保存最好的x个检查点
                              filename="{epoch}-{val_loss:.4f}",)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],  # single GPU
        max_epochs=200,
        callbacks=[lr_monitor, ckpt_cb],
        precision=32
    )

    # 7) fit
    trainer.fit(lit_model, train_loader, val_loader)

    # done

if __name__=="__main__":
    main()

