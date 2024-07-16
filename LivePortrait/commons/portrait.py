from LivePortrait.utils.io import load_driving_info
from .portrait_output import ParsingPaste
import cv2
import torch
import numpy as np


class PortraitController(ParsingPaste):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_source_image(self, img: np.ndarray) -> torch.Tensor:
        """ construct the input as standard
        img: HxWx3, uint8, 256x256
        """
        h, w = img.shape[:2]
        if h != self.cfg.input_shape[0] or w != self.cfg.input_shape[1]:
            x = cv2.resize(img, (self.cfg.input_shape[0], self.cfg.input_shape[1]))
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        return x

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

    def process_source_motion(self, img_rgb, source_motion, crop_info, cfg, cropper):
        template_lst = None
        input_eye_ratio_lst = None
        input_lip_ratio_lst = None
        driving_rgb_lst = load_driving_info(source_motion)
        driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
        i_d_lst = self.prepare_driving_videos(driving_rgb_lst_256, single_image=False)
        n_frames = i_d_lst.shape[0]
        if cfg.flag_eye_retargeting or cfg.flag_lip_retargeting:
            driving_lmk_lst = cropper.get_retargeting_lmk_info(driving_rgb_lst)
            input_eye_ratio_lst, input_lip_ratio_lst = self.calc_retargeting_ratio(driving_lmk_lst)
        mask_ori = self.prepare_paste_back(cfg.mask_crop, crop_info['M_c2o'],
                                           dsize=(img_rgb.shape[1], img_rgb.shape[0]))
        i_p_paste_lst = []
        return mask_ori, driving_rgb_lst, i_d_lst, i_p_paste_lst, template_lst, n_frames, input_eye_ratio_lst, input_lip_ratio_lst

    def algorithm(self, x_s, x_d_i_info, r_s, x_s_info, lip_delta_before_animation, cfg):
        r_d_i = self.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

        r_new = r_d_i @ r_s
        delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_s_info['exp'])
        scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_s_info['scale'])
        t_new = x_s_info['t'] + (x_d_i_info['t'] - x_s_info['t'])
        t_new[..., 2].fill_(0)  # zero tz

        x_d_i_new = scale_new * (x_s @ r_new + delta_new) + t_new
        if cfg.flag_lip_zero and lip_delta_before_animation is not None:
            x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
        return x_s, x_d_i_new

    def get_kp_info(self, session, x, x_s, r_s, x_s_info, lip_delta_before_animation, single_image=False,
                    run_local=False):
        """
        Get the implicit keypoint information.
        Args:
            session: ONNX session for model inference
            x: Input image or video frame
            x_s, r_s, x_s_info, lip_delta_before_animation: Additional inputs for algorithm processing
            single_image: Boolean flag indicating if x is a single image
            run_local: Boolean flag for local run mode
        Returns:
            Keypoint information as a dictionary or a tuple
        """
        # Prepare input based on flags
        if single_image == False and run_local == False:
            x = cv2.resize(x, (256, 256)) if run_local == False else np.array(x)
            x = self.prepare_driving_videos([x], single_image)[0]
        # Perform inference with ONNX model
        outputs = session['m_session'].run(None, {'input': np.array(x)})

        # Convert outputs to tensors
        kps_info = {
            'pitch': torch.tensor(outputs[0]),
            'yaw': torch.tensor(outputs[1]),
            'roll': torch.tensor(outputs[2]),
            't': torch.tensor(outputs[3]),
            'exp': torch.tensor(outputs[4]),
            'scale': torch.tensor(outputs[5]),
            'kp': torch.tensor(outputs[6]),
        }

        # Process keypoint information
        bs = kps_info['kp'].shape[0]
        kps_info['pitch'] = self.headpose_predict_to_degree(kps_info['pitch'])[:, None]  # Bx1
        kps_info['yaw'] = self.headpose_predict_to_degree(kps_info['yaw'])[:, None]  # Bx1
        kps_info['roll'] = self.headpose_predict_to_degree(kps_info['roll'])[:, None]  # Bx1
        kps_info['kp'] = kps_info['kp'].reshape(bs, -1, 3)  # BxNx3
        kps_info['exp'] = kps_info['exp'].reshape(bs, -1, 3)  # BxNx3

        if single_image or run_local:
            return kps_info

        x_s, x_d_i_new = self.algorithm(x_s, kps_info, r_s, x_s_info, lip_delta_before_animation, self.cfg)
        return x_s, x_d_i_new

    @staticmethod
    def get_3d_feature(session, source):
        outputs = session['f_session'].run(None, {'input': source})
        return outputs[0]

    def warp_decode(self, session, feature_3d, kp_source, kp_driving):
        """
        'occlusion_map': outputs[0],  # Example name, adjust as necessary
        'deformation': outputs[1],  # Example name, adjust as necessary
        'out': outputs[2]  # Example name, adjust as necessary
        """

        ort_inputs = {
            session['w_session'].get_inputs()[0].name: np.array(feature_3d),
            session['w_session'].get_inputs()[1].name: np.array(kp_driving),
            session['w_session'].get_inputs()[2].name: np.array(kp_source)
        }
        outputs = session['w_session'].run(None, ort_inputs)

        generator = session['g_session'].run(None, {'input': outputs[2]})
        return self.parse_output(generator[0])[0]
