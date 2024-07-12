import onnxruntime as ort
import torch
from torchvision import transforms
from Face2Vid.utils.io import load_image_rgb, resize_to_limit
from Face2Vid.utils.cropper import Cropper
from Face2Vid.commons.config.inference_config import InferenceConfig
from Face2Vid.utils.video import images2video
from Face2Vid.utils.helper import basename
from tqdm import tqdm
from utils_onnx import ParsingPaste
import numpy as np
import cv2
import os.path as osp


class LivePortraitONNX(ParsingPaste):
    def __init__(self, cfg, motion_extractor_path, appearance_feature_extractor_path, warping_path, spade_path,
                 stitch_path, lip_path, eye_path):
        super().__init__()
        self.cfg = cfg
        self.cropper = Cropper(crop_cfg=self.cfg)
        self.motion_extractor = motion_extractor_path
        self.features_extractor = appearance_feature_extractor_path
        self.warping = warping_path
        self.spade = spade_path
        self.stitch_path = stitch_path
        self.lip_path = lip_path
        self.eye_path = eye_path
        self.initialized_models()

    def initialized_models(self):
        self.m_session = ort.InferenceSession(self.motion_extractor)
        self.f_session = ort.InferenceSession(self.features_extractor)
        self.w_session = ort.InferenceSession(self.warping)
        self.g_session = ort.InferenceSession(self.spade)

        self.s_session = ort.InferenceSession(self.stitch_path)
        self.s_l_session = ort.InferenceSession(self.lip_path)
        self.s_e_session = ort.InferenceSession(self.eye_path)

        self.m_input_name = self.m_session.get_inputs()[0].name
        self.m_output_name = self.m_session.get_outputs()[0].name

        self.g_input_name = self.g_session.get_inputs()[0].name
        self.g_output_name = self.g_session.get_outputs()[0].name

        self.f_input_name = self.f_session.get_inputs()[0].name
        self.f_output_name = self.f_session.get_outputs()[0].name

        self.w_input_names = [input.name for input in self.w_session.get_inputs()]
        self.w_output_names = [output.name for output in self.w_session.get_outputs()]

    def prepare_source(self, img: np.ndarray) -> torch.Tensor:
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
    def transform(image):
        preprocess_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess_transform(image)

    def prepare_portrait(self, source_image_path):
        # Load and preprocess source image
        img_rgb = load_image_rgb(source_image_path)
        img_rgb = resize_to_limit(img_rgb, self.cfg.ref_max_shape, self.cfg.ref_shape_n)
        # log(f"Load source image from {source_image_path}")
        crop_info = self.cropper.crop_single_image(img_rgb)
        source_lmk = crop_info['lmk_crop']
        _, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']

        if self.cfg.flag_do_crop:
            i_s = self.prepare_source(img_crop_256x256)
        else:
            i_s = self.prepare_source(img_rgb)

        x_s_info = self.get_kp_info(i_s, x_s=None, r_s=None, x_s_info=None, lip_delta_before_animation=None,
                                    single_image=True)
        x_c_s = x_s_info['kp']
        r_s = self.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.get_3d_feature(np.array(i_s))
        x_s = self.transform_keypoint(x_s_info)

        lip_delta_before_animation = None
        if self.cfg.flag_lip_zero:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(
                c_d_lip_before_animation, crop_info['lmk_crop'])
            if combined_lip_ratio_tensor_before_animation[0][0] < self.cfg.lip_zero_threshold:
                self.cfg.flag_lip_zero = False
            else:
                lip_delta_before_animation = self.retarget_lip(self.s_l_session, x_s,
                                                               combined_lip_ratio_tensor_before_animation)
        return source_lmk, x_c_s, x_s, f_s, r_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, img_crop_256x256

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

    def get_kp_info(self, x, x_s, r_s, x_s_info, lip_delta_before_animation, single_image=False,
                    run_local=False) -> tuple or dict:
        """ get the implicit keypoint information
        x: Bx3xHxW, normalized to 0~1
        flag_refine_info: whether to transform the pose to degrees and the dimension of the reshape
        return: A dict contains keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'
        """
        if single_image:
            x = np.array(x)
        elif run_local and single_image == False:
            x = np.array(x)
        else:
            x = cv2.resize(x, (256, 256))
            x = self.prepare_driving_videos([x], single_image)[0]
        # Perform inference with ONNX model
        outputs = self.m_session.run(None, {self.m_input_name: x})
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
        if single_image:
            return kps_info
        elif run_local:
            return kps_info
        elif single_image == False and run_local == False:
            x_s, x_d_i_new = live_portrait.algorithm(x_s, kps_info, r_s, x_s_info, lip_delta_before_animation,
                                                     self.cfg)
            return x_s, x_d_i_new

    def get_3d_feature(self, source):
        outputs = self.f_session.run([self.f_output_name], {self.f_input_name: source})
        feature_3d = torch.tensor(outputs[0]).float()
        return feature_3d

    def warp_decode(self, feature_3d, kp_source, kp_driving, img_rgb, crop_info, mask_ori):
        ort_inputs = {
            self.w_input_names[0]: np.array(feature_3d),
            self.w_input_names[1]: np.array(kp_source),
            self.w_input_names[2]: np.array(kp_driving)
        }

        outputs = self.w_session.run(self.w_output_names, ort_inputs)

        warp = {
            'occlusion_map': outputs[0],  # Example name, adjust as necessary
            'deformation': outputs[1],  # Example name, adjust as necessary
            'out': outputs[2]  # Example name, adjust as necessary
        }

        generator = self.g_session.run(None, {self.g_input_name: warp['out']})
        i_p_i = self.parse_output(torch.from_numpy(generator[0]))
        if self.cfg.flag_pasteback:
            i_p_i_to_ori_blend = self.paste_back(i_p_i, crop_info['M_c2o'], img_rgb, mask_ori)
            return i_p_i_to_ori_blend, i_p_i
        else:
            return i_p_i

    def generate(self, n_frames, source_lmk, crop_info, img_rgb, mask_ori, i_d_lst, i_p_paste_lst, x_s,
                 r_s, f_s, x_s_info, x_c_s, eye_ratio_lst, lip_ratio_lst, lip_delta_before_animation):
        i_p_lst = []
        r_d_0, x_d_0_info = None, None
        for i in tqdm(range(n_frames), desc='Animating...', total=n_frames):
            i_d_i = i_d_lst[i]
            x_d_i_info = self.get_kp_info(i_d_i, x_s, r_s, x_s_info, lip_delta_before_animation, run_local=True)
            r_d_i = self.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

            if i == 0:
                r_d_0 = r_d_i
                x_d_0_info = x_d_i_info

            if self.cfg.flag_relative:
                r_new = (r_d_i @ r_d_0.permute(0, 2, 1)) @ r_s
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            else:
                r_new = r_d_i
                delta_new = x_d_i_info['exp']
                scale_new = x_s_info['scale']
                t_new = x_d_i_info['t']

            t_new[..., 2].fill_(0)  # zero tz
            x_d_i_new = scale_new * (x_c_s @ r_new + delta_new) + t_new

            # Algorithm 1:
            if not self.cfg.flag_stitching and not self.cfg.flag_eye_retargeting and not self.cfg.flag_lip_retargeting:
                # without stitching or retargeting
                if self.cfg.flag_lip_zero:
                    x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                else:
                    pass
            elif self.cfg.flag_stitching and not self.cfg.flag_eye_retargeting and not self.cfg.flag_lip_retargeting:
                # with stitching and without retargeting
                if self.cfg.flag_lip_zero:
                    x_d_i_new = self.stitching(self.s_session, x_s, x_d_i_new) + lip_delta_before_animation.reshape(-1,
                                                                                                                    x_s.shape[
                                                                                                                        1],
                                                                                                                    3)

                else:
                    x_d_i_new = self.stitching(self.s_session, x_s, x_d_i_new)

            else:
                eyes_delta, lip_delta = None, None

                if self.cfg.flag_eye_retargeting:
                    c_d_eyes_i = eye_ratio_lst[i]
                    combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i,
                                                                             source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.retarget_eye(self.s_e_session, x_s, combined_eye_ratio_tensor)
                if self.cfg.flag_lip_retargeting:
                    c_d_lip_i = lip_ratio_lst[i]
                    combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i,
                                                                             source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.retarget_lip(self.s_l_session, x_s, combined_lip_ratio_tensor)

                if self.cfg.flag_relative:  # use x_s
                    x_d_i_new = x_s + \
                                (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                                (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                                (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                                (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)

                if self.cfg.flag_stitching:
                    x_d_i_new = self.stitching(self.s_session, x_s, x_d_i_new)

            i_p_i_to_ori_blend, i_p_i = self.warp_decode(f_s, x_s, x_d_i_new, img_rgb, crop_info, mask_ori)
            i_p_paste_lst.append(i_p_i_to_ori_blend)
            i_p_lst.append(i_p_i)
        return i_p_lst


if __name__ == '__main__':
    video_path = 'assets/examples/driving/d2.mp4'
    source_img = 'assets/examples/source/s3.jpg'
    live_portrait = LivePortraitONNX(cfg=InferenceConfig(),
                                     motion_extractor_path='live_portraiet_onnx/onnx/motion_extractor.onnx',
                                     appearance_feature_extractor_path='live_portraiet_onnx/onnx/appearance_feature_extractor.onnx',
                                     warping_path='live_portraiet_onnx/onnx/warping.onnx',
                                     spade_path='live_portraiet_onnx/onnx/spade_generator.onnx',
                                     stitch_path='live_portraiet_onnx/onnx/stitching_retargeting.onnx',
                                     lip_path='live_portraiet_onnx/onnx/stitching_retargeting_lip.onnx',
                                     eye_path='live_portraiet_onnx/onnx/stitching_retargeting_eye.onnx')

    # cap = cv2.VideoCapture(0)
    source_landmark, x_c_ss, x_ss, f_ss, r_ss, \
        x_ss_info, lip_delta_before_animations, crops_info, \
        imgs_rgb, imgs_crop_256x256 = live_portrait.prepare_portrait(source_image_path=source_img)

    mask_ori, driving_rgb_lsts, i_d_lsts, i_p_paste_lsts, template_lsts, n_framess, input_eye_ratio_lsts, input_lip_ratio_lsts = live_portrait.process_source_motion(
        imgs_rgb, source_landmark, video_path, crops_info, live_portrait.cfg, live_portrait.cropper)

    result = live_portrait.generate(n_framess, source_landmark, crops_info, imgs_rgb, mask_ori, i_d_lsts,
                                    i_p_paste_lsts, x_ss, r_ss, f_ss, x_ss_info, x_c_ss, input_eye_ratio_lsts,
                                    input_lip_ratio_lsts,
                                    lip_delta_before_animations)
    live_portrait.mkdir('animations')
    frames_concatenated = live_portrait.concat_frames(result, driving_rgb_lsts, imgs_crop_256x256)
    wfp_concat = osp.join('animations',
                          f'{basename(source_img)}--{basename(source_img)}_concat.mp4')
    images2video(frames_concatenated, wfp=wfp_concat)

    wfp = osp.join('animations', f'{basename(source_img)}--{basename(source_img)}.mp4')
    images2video(i_p_paste_lsts, wfp=wfp)

    # output_path = 'output_video.mp4'

    # Initialize ImageIO writer
    # writer = imageio.get_writer(output_path, fps=30)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     mask_ori, driving_rgb_lst, i_d_lst, i_p_paste_lst, template_lst, n_frames, input_eye_ratio_lst, input_lip_ratio_lst = live_portrait.process_source_motion(
    #         img_rgb, source_lmk, video_path, crop_info, live_portrait.cfg)
    #     x_s, x_d_i_new = live_portrait.get_kp_info(frame, x_s, r_s, x_s_info, lip_delta_before_animation)
    #     i_p_i = live_portrait.warp_decode(np.array(f_s), np.array(x_s), np.array(x_d_i_new), img_rgb, crop_info, mask_ori)
    #     # writer.append_data(i_p_i)
    #     cv2.imshow('a', i_p_i[:, :, ::-1])
    #     if cv2.waitKey(1) & 0xff == ord('q'):
    #         break
    # cap.release()
