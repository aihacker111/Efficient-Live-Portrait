from LivePortrait.commons.utils.utils import load_driving_info
from .portrait_output import ParsingPaste
from ..face_analyze import FaceCropper
from LivePortrait.commons import EfficientLivePortraitPredictor
import cv2
import torch
import numpy as np


class PortraitController(ParsingPaste, FaceCropper):
    def __init__(self, use_tensorrt, half, **kwargs):
        super().__init__()
        self.predictor = EfficientLivePortraitPredictor(use_tensorrt, half, **kwargs)
        self.cropper = FaceCropper(**kwargs)
        self.cfg = kwargs

    def prepare_source_image(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        if h != self.cfg['input_shape'][0] or w != self.cfg['input_shape'][1]:
            x = cv2.resize(img, (self.cfg['input_shape'][0], self.cfg['input_shape'][1]))
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
        return y

    def process_source_motion(self, img_rgb, source_motion, crop_info, cropper):
        template_lst = None
        input_eye_ratio_lst = None
        input_lip_ratio_lst = None
        driving_rgb_lst = load_driving_info(source_motion)
        driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
        i_d_lst = self.prepare_driving_videos(driving_rgb_lst_256, single_image=False)
        n_frames = i_d_lst.shape[0]
        if self.cfg['flag_eye_retargeting'] or self.cfg['flag_lip_retargeting']:
            driving_lmk_lst = cropper.get_retargeting_lmk_info(driving_rgb_lst)
            input_eye_ratio_lst, input_lip_ratio_lst = self.calc_retargeting_ratio(driving_lmk_lst)
        mask_ori = self.prepare_paste_back(crop_info['M_c2o'],
                                           dsize=(img_rgb.shape[1], img_rgb.shape[0]))
        i_p_paste_lst = []
        return mask_ori, driving_rgb_lst, i_d_lst, i_p_paste_lst, template_lst, n_frames, input_eye_ratio_lst, input_lip_ratio_lst

    def process_multiple_source_motion(self, source_motion, crop_info, max_faces, cropper, cropping_video):
        template_lst = None
        input_eye_ratio_lst = None
        input_lip_ratio_lst = None
        if cropping_video:
            cropped_face_motion_info = self.cropper.crop_source_video(source_motion, max_faces)
        else:
            cropped_face_motion_info = self.cropper.calc_lmks_from_cropped_video(source_motion)
        result_list = []

        # Iterate over the faces in crop_info and match with motion data
        for face_index, crop_f in enumerate(crop_info):
            # Iterate over all face keys in the current crop_f
            for face_key in crop_f.keys():
                face_data = crop_f[face_key]

                # Check if there's corresponding motion data
                if face_index < len(cropped_face_motion_info):
                    crop_face_motion = cropped_face_motion_info[face_index]

                    # Iterate over all motion keys to find the corresponding data
                    for motion_key in crop_face_motion.keys():
                        motion_data = crop_face_motion[motion_key]

                        # Update face_data with motion data
                        # face_data['frame_crop_lst'] = motion_data['frame_crop_lst']
                        # face_data['M_c2o_lst'] = motion_data['M_c2o_lst']
                        n_frames = len(motion_data['frame_crop_lst'])
                        original_fps = motion_data['fps']
                        # Resize frames for the current face
                        # driving_rgb_lst_256 = [cv2.resize(frame, (256, 256)) for frame in face_data['frame_crop_lst']]
                        driving_rgb_lst_256 = motion_data['frame_crop_lst']
                        # Prepare driving videos for the current face
                        i_d_lst = self.prepare_driving_videos(driving_rgb_lst_256, single_image=False)

                        # Calculate retargeting ratios if needed
                        if self.cfg['flag_eye_retargeting'] or self.cfg['flag_lip_retargeting']:
                            driving_lmk_lst = cropper.get_retargeting_lmk_info(motion_data['frame_crop_lst'])
                            input_eye_ratio_lst, input_lip_ratio_lst = self.calc_retargeting_ratio(driving_lmk_lst)

                        # Prepare mask for the current face
                        mask_ori = self.prepare_paste_back(face_data['M_c2o'],
                                                           dsize=(face_data['img_rgb'].shape[1],
                                                                  face_data['img_rgb'].shape[0]))
                        face_data['mask_ori'] = mask_ori

                        # Update result list with combined face_motion_{face_index} and face_{face_index}
                        result_list.append({
                            f"face_motion_{face_index}": {
                                'fps': original_fps,
                                'i_d_lst': i_d_lst,
                                'template_lst': template_lst,
                                'n_frames': n_frames,
                                'input_eye_ratio_lst': input_eye_ratio_lst,
                                'input_lip_ratio_lst': input_lip_ratio_lst,
                            },
                            f"face_image_{face_index}": face_data  # Store the face data
                        })

        return result_list

    def algorithm(self, x_s, x_d_i_info, r_s, x_s_info, lip_delta_before_animation):
        r_d_i = self.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

        r_new = r_d_i @ r_s
        delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_s_info['exp'])
        scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_s_info['scale'])
        t_new = x_s_info['t'] + (x_d_i_info['t'] - x_s_info['t'])
        t_new[..., 2].fill_(0)  # zero tz

        x_d_i_new = scale_new * (x_s @ r_new + delta_new) + t_new
        if self.cfg['flag_lip_zero'] and lip_delta_before_animation is not None:
            x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
        return x_s, x_d_i_new

    def get_kp_info(self, x, x_s, r_s, x_s_info, lip_delta_before_animation, single_image=False, run_local=False):
        if single_image == False and run_local == False:
            x = cv2.resize(x, (256, 256)) if run_local == False else x
            x = self.prepare_driving_videos([x], single_image)[0]
        inputs = {'img': np.array(x)}
        outputs = self.predictor.run_time(engine_name='motion_extractor', task='m_session', inputs_onnx=inputs,
                                          inputs_tensorrt=[x])
        kps_info = {
            'pitch': torch.tensor(outputs[0]),
            'yaw': torch.tensor(outputs[1]),
            'roll': torch.tensor(outputs[2]),
            't': torch.tensor(outputs[3]),
            'exp': torch.tensor(outputs[4]),
            'scale': torch.tensor(outputs[5]),
            'kp': torch.tensor(outputs[6]),
        }

        bs = kps_info['kp'].shape[0]
        kps_info['pitch'] = self.headpose_predict_to_degree(kps_info['pitch'])[:, None]  # Bx1
        kps_info['yaw'] = self.headpose_predict_to_degree(kps_info['yaw'])[:, None]  # Bx1
        kps_info['roll'] = self.headpose_predict_to_degree(kps_info['roll'])[:, None]  # Bx1
        kps_info['kp'] = kps_info['kp'].reshape(bs, -1, 3)  # BxNx3
        kps_info['exp'] = kps_info['exp'].reshape(bs, -1, 3)  # BxNx3

        if single_image or run_local:
            return kps_info

        x_s, x_d_i_new = self.algorithm(x_s, kps_info, r_s, x_s_info, lip_delta_before_animation)
        return x_s, x_d_i_new

    def get_3d_feature(self, source):
        inputs = {'img': np.array(source)}
        outputs = self.predictor.run_time(engine_name='feature_extractor', task='f_session', inputs_onnx=inputs,
                                          inputs_tensorrt=[source])
        return outputs[0]

    def warp_decode(self, feature_3d, kp_source, kp_driving):
        inputs = {
            'feature_3d': np.array(feature_3d),
            'kp_driving': np.array(kp_driving),
            'kp_source': np.array(kp_source)
        }
        generator = self.predictor.run_time(engine_name='generator', task='gw_session',
                                            inputs_onnx=inputs, inputs_tensorrt=[feature_3d, kp_driving, kp_source])
        return self.parse_output(generator[0])[0]
