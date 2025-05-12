import torch
from torchmetrics import Metric
import torch.nn.functional as F

class my_Metric(Metric):
    def __init__(self):
        super(my_Metric, self).__init__()
        self.metric_names = [
            'agl_1_act_mse_angle_error', 'agl_1_act_mae_angle_error',
            'agl_2_act_mse_angle_error', 'agl_2_act_mae_angle_error',
            'agl_3_act_mse_angle_error', 'agl_3_act_mae_angle_error',
            'agl_4_act_mse_angle_error', 'agl_4_act_mae_angle_error',
            'agl_5_act_mse_angle_error', 'agl_5_act_mae_angle_error',
            'agl_6_act_mse_angle_error', 'agl_6_act_mae_angle_error',
            'agl2_1_act_mse_angle_error', 'agl2_1_act_mae_angle_error',
            'agl2_2_act_mse_angle_error', 'agl2_2_act_mae_angle_error',
            'agl2_3_act_mse_angle_error', 'agl2_3_act_mae_angle_error',
            'agl2_4_act_mse_angle_error', 'agl2_4_act_mae_angle_error',
            'agl2_5_act_mse_angle_error', 'agl2_5_act_mae_angle_error',
            'agl2_6_act_mse_angle_error', 'agl2_6_act_mae_angle_error',
            'gripper_act_mse_width_error',
            'gripper_act_mae_width_error',
            'gripper_act2_mse_width_error',
            'gripper_act2_mae_width_error'
        ]

        for metric_name in self.metric_names:
            self.add_state(metric_name, default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_action, gt_action):
        with torch.no_grad():
            # arm1
            agl_1_act_mse_angle_error = F.mse_loss(pred_action[..., 0:1], gt_action[..., 0:1])
            agl_1_act_mae_angle_error = F.l1_loss(pred_action[..., 0:1], gt_action[..., 0:1])
            agl_2_act_mse_angle_error = F.mse_loss(pred_action[..., 1:2], gt_action[..., 1:2])
            agl_2_act_mae_angle_error = F.l1_loss(pred_action[..., 1:2], gt_action[..., 1:2])
            agl_3_act_mse_angle_error = F.mse_loss(pred_action[..., 2:3], gt_action[..., 2:3])
            agl_3_act_mae_angle_error = F.l1_loss(pred_action[..., 2:3], gt_action[..., 2:3])
            agl_4_act_mse_angle_error = F.mse_loss(pred_action[..., 3:4], gt_action[..., 3:4])
            agl_4_act_mae_angle_error = F.l1_loss(pred_action[..., 3:4], gt_action[..., 3:4])
            agl_5_act_mse_angle_error = F.mse_loss(pred_action[..., 4:5], gt_action[..., 4:5])
            agl_5_act_mae_angle_error = F.l1_loss(pred_action[..., 4:5], gt_action[..., 4:5])
            agl_6_act_mse_angle_error = F.mse_loss(pred_action[..., 5:6], gt_action[..., 5:6])
            agl_6_act_mae_angle_error = F.l1_loss(pred_action[..., 5:6], gt_action[..., 5:6])

            self.agl_1_act_mse_angle_error += agl_1_act_mse_angle_error
            self.agl_1_act_mae_angle_error += agl_1_act_mae_angle_error
            self.agl_2_act_mse_angle_error += agl_2_act_mse_angle_error
            self.agl_2_act_mae_angle_error += agl_2_act_mae_angle_error
            self.agl_3_act_mse_angle_error += agl_3_act_mse_angle_error
            self.agl_3_act_mae_angle_error += agl_3_act_mae_angle_error
            self.agl_4_act_mse_angle_error += agl_4_act_mse_angle_error
            self.agl_4_act_mae_angle_error += agl_4_act_mae_angle_error
            self.agl_5_act_mse_angle_error += agl_5_act_mse_angle_error
            self.agl_5_act_mae_angle_error += agl_5_act_mae_angle_error
            self.agl_6_act_mse_angle_error += agl_6_act_mse_angle_error
            self.agl_6_act_mae_angle_error += agl_6_act_mae_angle_error

            # arm2
            agl2_1_act_mse_angle_error = F.mse_loss(pred_action[..., 7:8], gt_action[..., 7:8])
            agl2_1_act_mae_angle_error = F.l1_loss(pred_action[..., 7:8], gt_action[..., 7:8])
            agl2_2_act_mse_angle_error = F.mse_loss(pred_action[..., 8:9], gt_action[..., 8:9])
            agl2_2_act_mae_angle_error = F.l1_loss(pred_action[..., 8:9], gt_action[..., 8:9])
            agl2_3_act_mse_angle_error = F.mse_loss(pred_action[..., 9:10], gt_action[..., 9:10])
            agl2_3_act_mae_angle_error = F.l1_loss(pred_action[..., 9:10], gt_action[..., 9:10])
            agl2_4_act_mse_angle_error = F.mse_loss(pred_action[..., 10:11], gt_action[..., 10:11])
            agl2_4_act_mae_angle_error = F.l1_loss(pred_action[..., 10:11], gt_action[..., 10:11])
            agl2_5_act_mse_angle_error = F.mse_loss(pred_action[..., 11:12], gt_action[..., 11:12])
            agl2_5_act_mae_angle_error = F.l1_loss(pred_action[..., 11:12], gt_action[..., 11:12])
            agl2_6_act_mse_angle_error = F.mse_loss(pred_action[..., 12:13], gt_action[..., 12:13])
            agl2_6_act_mae_angle_error = F.l1_loss(pred_action[..., 12:13], gt_action[..., 12:13])

            self.agl2_1_act_mse_angle_error += agl2_1_act_mse_angle_error
            self.agl2_1_act_mae_angle_error += agl2_1_act_mae_angle_error
            self.agl2_2_act_mse_angle_error += agl2_2_act_mse_angle_error
            self.agl2_2_act_mae_angle_error += agl2_2_act_mae_angle_error
            self.agl2_3_act_mse_angle_error += agl2_3_act_mse_angle_error
            self.agl2_3_act_mae_angle_error += agl2_3_act_mae_angle_error
            self.agl2_4_act_mse_angle_error += agl2_4_act_mse_angle_error
            self.agl2_4_act_mae_angle_error += agl2_4_act_mae_angle_error
            self.agl2_5_act_mse_angle_error += agl2_5_act_mse_angle_error
            self.agl2_5_act_mae_angle_error += agl2_5_act_mae_angle_error
            self.agl2_6_act_mse_angle_error += agl2_6_act_mse_angle_error
            self.agl2_6_act_mae_angle_error += agl2_6_act_mae_angle_error

            gripper_act_mse_width_error = F.mse_loss(pred_action[..., 6:7], gt_action[..., 6:7])
            gripper_act_mae_width_error = F.l1_loss(pred_action[..., 6:7], gt_action[..., 6:7])
            self.gripper_act_mse_width_error += gripper_act_mse_width_error
            self.gripper_act_mae_width_error += gripper_act_mae_width_error

            gripper_act2_mse_width_error = F.mse_loss(pred_action[..., 13:14], gt_action[..., 13:14])
            gripper_act2_mae_width_error = F.l1_loss(pred_action[..., 13:14], gt_action[..., 13:14])
            self.gripper_act2_mse_width_error += gripper_act2_mse_width_error
            self.gripper_act2_mae_width_error += gripper_act2_mae_width_error

            # 更新 total
            self.total += 1

    def compute(self):
        return {
            'agl_1_act_mse_angle_error': self.agl_1_act_mse_angle_error.float() / self.total,
            'agl_1_act_mae_angle_error': self.agl_1_act_mae_angle_error.float() / self.total,
            'agl_2_act_mse_angle_error': self.agl_2_act_mse_angle_error.float() / self.total,
            'agl_2_act_mae_angle_error': self.agl_2_act_mae_angle_error.float() / self.total,
            'agl_3_act_mse_angle_error': self.agl_3_act_mse_angle_error.float() / self.total,
            'agl_3_act_mae_angle_error': self.agl_3_act_mae_angle_error.float() / self.total,
            'agl_4_act_mse_angle_error': self.agl_4_act_mse_angle_error.float() / self.total,
            'agl_4_act_mae_angle_error': self.agl_4_act_mae_angle_error.float() / self.total,
            'agl_5_act_mse_angle_error': self.agl_5_act_mse_angle_error.float() / self.total,
            'agl_5_act_mae_angle_error': self.agl_5_act_mae_angle_error.float() / self.total,
            'agl_6_act_mse_angle_error': self.agl_6_act_mse_angle_error.float() / self.total,
            'agl_6_act_mae_angle_error': self.agl_6_act_mae_angle_error.float() / self.total,
            'agl2_1_act_mse_angle_error': self.agl2_1_act_mse_angle_error.float() / self.total,
            'agl2_1_act_mae_angle_error': self.agl2_1_act_mae_angle_error.float() / self.total,
            'agl2_2_act_mse_angle_error': self.agl2_2_act_mse_angle_error.float() / self.total,
            'agl2_2_act_mae_angle_error': self.agl2_2_act_mae_angle_error.float() / self.total,
            'agl2_3_act_mse_angle_error': self.agl2_3_act_mse_angle_error.float() / self.total,
            'agl2_3_act_mae_angle_error': self.agl2_3_act_mae_angle_error.float() / self.total,
            'agl2_4_act_mse_angle_error': self.agl2_4_act_mse_angle_error.float() / self.total,
            'agl2_4_act_mae_angle_error': self.agl2_4_act_mae_angle_error.float() / self.total,
            'agl2_5_act_mse_angle_error': self.agl2_5_act_mse_angle_error.float() / self.total,
            'agl2_5_act_mae_angle_error': self.agl2_5_act_mae_angle_error.float() / self.total,
            'agl2_6_act_mse_angle_error': self.agl2_6_act_mse_angle_error.float() / self.total,
            'agl2_6_act_mae_angle_error': self.agl2_6_act_mae_angle_error.float() / self.total,
            'gripper_act_mse_width_error': self.gripper_act_mse_width_error.float() / self.total,
            'gripper_act_mae_width_error': self.gripper_act_mae_width_error.float() / self.total,
            'gripper_act2_mse_width_error': self.gripper_act2_mse_width_error.float() / self.total,
            'gripper_act2_mae_width_error': self.gripper_act2_mae_width_error.float() / self.total
        }