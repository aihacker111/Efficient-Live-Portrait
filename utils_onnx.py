import os.path as osp
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import os
from Face2Vid.utils.io import load_driving_info
from Face2Vid.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from rich.progress import track

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # NOTE: enforce single thread

DTYPE = np.float32
CV2_INTERP = cv2.INTER_LINEAR


class TransformFunction:
    def __init__(self):
        pass

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

    def concat_feat(self, stitch_session, kp_source, kp_driving, lip_ratio, eye_ratio,
                    lip=False, eye=False) -> torch.Tensor:
        """
        kp_source: (bs, k, 3)
        kp_driving: (bs, k, 3)
        Return: (bs, 2k*3)
        """
        alert = 'batch size must be equal'
        if lip == False and eye == False:
            bs_src = kp_source.shape[0]
            bs_dri = kp_driving.shape[0]
            assert bs_src == bs_dri, alert

            feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
            delta = stitch_session.run(None, {'input': np.array(feat)})
            return delta[0]
        elif lip == True and eye == False:
            bs_src = kp_source.shape[0]
            bs_dri = lip_ratio.shape[0]
            assert bs_src == bs_dri, alert

            feat = torch.cat([kp_source.view(bs_src, -1), lip_ratio.view(bs_dri, -1)], dim=1)
            delta_lip = stitch_session.run(None, {'input': np.array(feat)})
            return delta_lip[0]
        elif lip == False and eye == True:
            bs_src = kp_source.shape[0]
            bs_dri = eye_ratio.shape[0]
            assert bs_src == bs_dri, alert

            feat = torch.cat([kp_source.view(bs_src, -1), eye_ratio.view(bs_dri, -1)], dim=1)
            delta_eye = stitch_session.run(None, {'input': np.array(feat)})
            return delta_eye[0]

    @staticmethod
    def prepare_driving_videos(imgs, single_image):
        """ construct the input as standard
        imgs: NxBxHxWx3, uint8
        """
        if isinstance(imgs, list):
            _imgs = np.array(imgs)[..., np.newaxis]  # TxHxWx3x1
        elif isinstance(imgs, np.ndarray):
            _imgs = imgs
        else:
            raise ValueError(f'imgs type error: {type(imgs)}')
        y = _imgs.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)  # clip to 0~1
        if single_image:
            y = torch.from_numpy(y).permute(0, 4, 1, 3, 2)  # TxHxWx3x1 -> Tx1x3xHxW
        else:
            y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)
        y = np.array(y).astype('float32')
        return y

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

    def retarget_lip(self, session, kp_source, lip_close_ratio) -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        """
        delta_lip = self.concat_feat(stitch_session=session, kp_source=kp_source, kp_driving=None,
                                     lip_ratio=lip_close_ratio, eye_ratio=None, lip=True)

        return delta_lip

    def retarget_eye(self, session, kp_source, eye_close_ratio) -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        """
        delta_lip = self.concat_feat(stitch_session=session, kp_source=kp_source, kp_driving=None, lip_ratio=None,
                                     eye_ratio=eye_close_ratio, eye=True)

        return delta_lip

    def calculate_distance_ratio(self, lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int,
                                 eps: float = 1e-6) -> np.ndarray:
        """
        Calculate the ratio of the distance between two pairs of landmarks.

        Parameters:
        lmk (np.ndarray): Landmarks array of shape (B, N, 2).
        idx1, idx2, idx3, idx4 (int): Indices of the landmarks.
        eps (float): Small value to avoid division by zero.

        Returns:
        np.ndarray: Calculated distance ratio.
        """
        return (np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) /
                (np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps))

    def calc_lip_close_ratio(self, lmk: np.ndarray) -> np.ndarray:
        """
        Calculate the lip-close ratio.

        Parameters:
        lmk (np.ndarray): Landmarks array of shape (B, N, 2).

        Returns:
        np.ndarray: Calculated lip-close ratio.
        """
        return self.calculate_distance_ratio(lmk, 90, 102, 48, 66)

    def calc_combined_eye_ratio(self, input_eye_ratio, source_lmk):
        eye_close_ratio = calc_eye_close_ratio(source_lmk[None])
        eye_close_ratio_tensor = torch.from_numpy(eye_close_ratio).float()
        input_eye_ratio_tensor = torch.Tensor([input_eye_ratio[0][0]]).reshape(1, 1)
        # [c_s,eyes, c_d,eyes,i]
        combined_eye_ratio_tensor = torch.cat([eye_close_ratio_tensor, input_eye_ratio_tensor], dim=1)
        return combined_eye_ratio_tensor

    def calc_combined_lip_ratio(self, input_lip_ratio, source_lmk):
        lip_close_ratio = calc_lip_close_ratio(source_lmk[None])
        lip_close_ratio_tensor = torch.from_numpy(lip_close_ratio).float()
        # [c_s,lip, c_d,lip,i]
        input_lip_ratio_tensor = torch.Tensor([input_lip_ratio[0]])
        if input_lip_ratio_tensor.shape != [1, 1]:
            input_lip_ratio_tensor = input_lip_ratio_tensor.reshape(1, 1)
        combined_lip_ratio_tensor = torch.cat([lip_close_ratio_tensor, input_lip_ratio_tensor], dim=1)
        return combined_lip_ratio_tensor


class ParsingPaste(TransformFunction):
    def __init__(self):
        super().__init__()

    def parse_output(self, out) -> np.ndarray:
        """ construct the output as standard
        return: 1xHxWx3, uint8
        """
        out = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)  # clip to 0~1
        out = np.clip(out * 255, 0, 255).astype(np.uint8)  # 0~1 -> 0~255

        return out

    def prepare_paste_back(self, mask_crop, crop_m_c2o, dsize):
        """prepare mask for later image paste back
        """
        if mask_crop is None:
            mask_crop = cv2.imread(self.make_abs_path('./assets/mask_template.png'), cv2.IMREAD_COLOR)
        mask_ori = self._transform_img(mask_crop, crop_m_c2o, dsize)
        mask_ori = mask_ori.astype(np.float32) / 255.
        return mask_ori

    def paste_back(self, image_to_processed, crop_m_c2o, rgb_ori, mask_ori):
        """paste back the image
        """
        dsize = (rgb_ori.shape[1], rgb_ori.shape[0])
        result = self._transform_img(image_to_processed[0], crop_m_c2o, dsize=dsize)
        result = np.clip(mask_ori * result + (1 - mask_ori) * rgb_ori, 0, 255).astype(np.uint8)
        return result

    def make_abs_path(self, fn):
        return osp.join(osp.dirname(osp.realpath(__file__)), fn)

    def _transform_img(self, img, M, dsize, flags=CV2_INTERP, borderMode=None):
        """ conduct similarity or affine transformation to the image, do not do border operation!
        img:
        M: 2x3 matrix or 3x3 matrix
        dsize: target shape (width, height)
        """
        if isinstance(dsize, tuple) or isinstance(dsize, list):
            _dsize = tuple(dsize)
        else:
            _dsize = (dsize, dsize)

        if borderMode is not None:
            return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags, borderMode=borderMode,
                                  borderValue=(0, 0, 0))
        else:
            return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)

    def process_source_motion(self, img_rgb, source_lmk, source_motion, crop_info, cfg, cropper):
        template_lst = None
        input_eye_ratio_lst = None
        input_lip_ratio_lst = None
        driving_rgb_lst = load_driving_info(source_motion)
        driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
        i_d_lst = self.prepare_driving_videos(driving_rgb_lst_256, single_image=False)
        n_frames = i_d_lst.shape[0]
        if cfg.flag_eye_retargeting or cfg.flag_lip_retargeting:
            driving_lmk_lst = cropper.get_retargeting_lmk_info(driving_rgb_lst)
            input_eye_ratio_lst, input_lip_ratio_lst = self.calc_retargeting_ratio(source_lmk,
                                                                                   driving_lmk_lst)
        mask_ori = self.prepare_paste_back(cfg.mask_crop, crop_info['M_c2o'],
                                           dsize=(img_rgb.shape[1], img_rgb.shape[0]))
        i_p_paste_lst = []
        return mask_ori, driving_rgb_lst, i_d_lst, i_p_paste_lst, template_lst, n_frames, input_eye_ratio_lst, input_lip_ratio_lst

    def calc_retargeting_ratio(self, source_lmk, driving_lmk_lst):
        input_eye_ratio_lst = []
        input_lip_ratio_lst = []
        for lmk in driving_lmk_lst:
            # for eyes retargeting
            input_eye_ratio_lst.append(calc_eye_close_ratio(lmk[None]))
            # for lip retargeting
            input_lip_ratio_lst.append(calc_lip_close_ratio(lmk[None]))
        return input_eye_ratio_lst, input_lip_ratio_lst

    def stitch(self, session, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        delta = self.concat_feat(session, kp_source, kp_driving, lip_ratio=None, eye_ratio=None)

        return delta

    def stitching(self, session, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        bs, num_kp = kp_source.shape[:2]

        kp_driving_new = kp_driving.clone()
        delta = self.stitch(session, kp_source, kp_driving_new)

        delta_exp = delta[..., :3 * num_kp].reshape(bs, num_kp, 3)  # 1x20x3
        delta_tx_ty = delta[..., 3 * num_kp:3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2

        kp_driving_new += delta_exp
        kp_driving_new[..., :2] += delta_tx_ty

        return kp_driving_new

    def concat_frames(self, I_p_lst, driving_rgb_lst, img_rgb):
        # TODO: add more concat style, e.g., left-down corner driving
        out_lst = []
        for idx, _ in track(enumerate(I_p_lst), total=len(I_p_lst), description='Concatenating result...'):
            source_image_drived = np.squeeze(I_p_lst[idx])
            image_drive = driving_rgb_lst[idx]
            # resize images to match source_image_drived shape
            h, w, _ = source_image_drived.shape
            image_drive_resized = cv2.resize(image_drive, (w, h))
            img_rgb_resized = cv2.resize(img_rgb, (w, h))

            # concatenate images horizontally
            frame = np.concatenate((image_drive_resized, img_rgb_resized, source_image_drived), axis=1)
            out_lst.append(frame)
        return out_lst

    def mkdir(self, d, log=False):
        # return self-assined `d`, for one line code
        if not osp.exists(d):
            os.makedirs(d, exist_ok=True)
            if log:
                print(f"Make dir: {d}")
        return d
