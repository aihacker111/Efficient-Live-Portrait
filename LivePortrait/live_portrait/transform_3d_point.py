# coding: utf-8
# Author: Vo Nguyen An Tin
# Email: tinprocoder0908@gmail.com

import torch.nn.functional as F
import torch
import numpy as np
from LivePortrait.live_portrait.retarget_portrait import RetargetStitchPortrait


class Transform3DFunction(RetargetStitchPortrait):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_rotation_matrix(pitch_, yaw_, roll_):
        """ the input is in degree
        """
        # calculate the rotation matrix: vps @ rot

        # transform to radian
        PI = np.pi
        pitch = pitch_ / 180 * PI
        yaw = yaw_ / 180 * PI
        roll = roll_ / 180 * PI

        device = pitch.device

        if pitch.ndim == 1:
            pitch = pitch.unsqueeze(1)
        if yaw.ndim == 1:
            yaw = yaw.unsqueeze(1)
        if roll.ndim == 1:
            roll = roll.unsqueeze(1)

        # calculate the euler matrix
        bs = pitch.shape[0]
        ones = torch.ones([bs, 1]).to(device)
        zeros = torch.zeros([bs, 1]).to(device)
        x, y, z = pitch, yaw, roll

        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x),
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([bs, 3, 3])

        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([bs, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([bs, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)  # transpose

    @staticmethod
    def headpose_predict_to_degree(predicted):
        """
        pred: (bs, 66) or (bs, 1) or others
        """
        if predicted.ndim > 1 and predicted.shape[1] == 66:
            # NOTE: note that the average is modified to 97.5
            device = predicted.device
            idx_tensor = [idx for idx in range(0, 66)]
            idx_tensor = torch.FloatTensor(idx_tensor).to(device)
            predicted = F.softmax(predicted, dim=1)
            degree = torch.sum(predicted * idx_tensor, axis=1) * 3 - 97.5
            return degree
        return predicted

    def transform_keypoint(self, kp_info: dict):
        """
        transform the implicit keypoints with the pose, shift, and expression deformation
        kp: BxNx3
        """
        kp = kp_info['kp']  # (bs, k, 3)
        pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

        t, exp = kp_info['t'], kp_info['exp']
        scale = kp_info['scale']

        pitch = self.headpose_predict_to_degree(pitch)
        yaw = self.headpose_predict_to_degree(yaw)
        roll = self.headpose_predict_to_degree(roll)

        bs = kp.shape[0]
        if kp.ndim == 2:
            num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
        else:
            num_kp = kp.shape[1]  # Bxnum_kpx3

        rot_mat = self.get_rotation_matrix(pitch, yaw, roll)  # (bs, 3, 3)

        # Eqn.2: s * (R * x_c,s + exp) + t
        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
        kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

        return kp_transformed

    def calc_retargeting_ratio(self, driving_lmk_lst):
        input_eye_ratio_lst = []
        input_lip_ratio_lst = []
        for lmk in driving_lmk_lst:
            # for eyes retargeting
            input_eye_ratio_lst.append(self.calc_eye_close_ratio(lmk[None]))
            # for lip retargeting
            input_lip_ratio_lst.append(self.calc_lip_close_ratio(lmk[None]))
        return input_eye_ratio_lst, input_lip_ratio_lst

    def calc_combined_eye_ratio(self, input_eye_ratio, source_lmk):
        eye_close_ratio = self.calc_eye_close_ratio(source_lmk[None])
        eye_close_ratio_tensor = torch.from_numpy(eye_close_ratio).float()
        input_eye_ratio_tensor = torch.Tensor([input_eye_ratio[0][0]]).reshape(1, 1)
        # [c_s,eyes, c_d,eyes,i]
        combined_eye_ratio_tensor = torch.cat([eye_close_ratio_tensor, input_eye_ratio_tensor], dim=1)
        return combined_eye_ratio_tensor

    def calc_combined_lip_ratio(self, input_lip_ratio, source_lmk):
        lip_close_ratio = self.calc_lip_close_ratio(source_lmk[None])
        lip_close_ratio_tensor = torch.from_numpy(lip_close_ratio).float()
        # [c_s,lip, c_d,lip,i]
        input_lip_ratio_tensor = torch.Tensor([input_lip_ratio[0]])
        if input_lip_ratio_tensor.shape != [1, 1]:
            input_lip_ratio_tensor = input_lip_ratio_tensor.reshape(1, 1)
        combined_lip_ratio_tensor = torch.cat([lip_close_ratio_tensor, input_lip_ratio_tensor], dim=1)
        return combined_lip_ratio_tensor
