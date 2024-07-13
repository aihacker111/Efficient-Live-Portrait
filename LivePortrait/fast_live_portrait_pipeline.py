import cv2
import onnxruntime as ort
import numpy as np
import os.path as osp
from tqdm import tqdm
from LivePortrait.utils import load_image_rgb, resize_to_limit, Cropper, images2video, basename
from LivePortrait.commons import PortraitController, Config


class LivePortraitONNX(PortraitController):
    def __init__(self, cfg=Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.cropper = Cropper(crop_cfg=self.cfg)
        self._model_sessions = None
        self.model_sessions()

    def model_sessions(self):
        if self._model_sessions is None:
            self._model_sessions = self._initialize_sessions()
        return self._model_sessions

    def _initialize_sessions(self):
        m_session = ort.InferenceSession(self.cfg.checkpoint_M)
        f_session = ort.InferenceSession(self.cfg.checkpoint_F)
        w_session = ort.InferenceSession(self.cfg.checkpoint_W)
        g_session = ort.InferenceSession(self.cfg.checkpoint_G)

        s_session = ort.InferenceSession(self.cfg.checkpoint_S)
        s_l_session = ort.InferenceSession(self.cfg.checkpoint_SL)
        s_e_session = ort.InferenceSession(self.cfg.checkpoint_SE)

        m_input_name = m_session.get_inputs()[0].name
        m_output_name = m_session.get_outputs()[0].name

        g_input_name = g_session.get_inputs()[0].name
        g_output_name = g_session.get_outputs()[0].name

        f_input_name = f_session.get_inputs()[0].name
        f_output_name = f_session.get_outputs()[0].name

        w_input_names = [input.name for input in w_session.get_inputs()]
        w_output_names = [output.name for output in w_session.get_outputs()]

        return {
            'm_session': m_session, 'm_input_name': m_input_name, 'm_output_name': m_output_name,
            'g_session': g_session, 'g_input_name': g_input_name, 'g_output_name': g_output_name,
            'f_session': f_session, 'f_input_name': f_input_name, 'f_output_name': f_output_name,
            'w_session': w_session, 'w_input_names': w_input_names, 'w_output_names': w_output_names,
            's_session': s_session, 's_l_session': s_l_session, 's_e_session': s_e_session
        }

    def prepare_portrait(self, source_image_path):
        # Load and preprocess source image
        img_rgb = load_image_rgb(source_image_path)
        img_rgb = resize_to_limit(img_rgb, self.cfg.ref_max_shape, self.cfg.ref_shape_n)
        # log(f"Load source image from {source_image_path}")
        crop_info = self.cropper.crop_single_image(img_rgb)
        source_lmk = crop_info['lmk_crop']
        _, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']

        if self.cfg.flag_do_crop:
            i_s = self.prepare_source_image(img_crop_256x256)
        else:
            i_s = self.prepare_source_image(img_rgb)

        x_s_info = self.get_kp_info(self._model_sessions, i_s, x_s=None, r_s=None, x_s_info=None,
                                    lip_delta_before_animation=None, single_image=True)
        x_c_s = x_s_info['kp']
        r_s = self.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.get_3d_feature(self._model_sessions, np.array(i_s))
        x_s = self.transform_keypoint(x_s_info)

        lip_delta_before_animation = None
        if self.cfg.flag_lip_zero:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(
                c_d_lip_before_animation, crop_info['lmk_crop'])
            if combined_lip_ratio_tensor_before_animation[0][0] < self.cfg.lip_zero_threshold:
                self.cfg.flag_lip_zero = False
            else:
                lip_delta_before_animation = self.retarget_lip(self._model_sessions['s_l_session'], x_s,
                                                               combined_lip_ratio_tensor_before_animation)
        return source_lmk, x_c_s, x_s, f_s, r_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, img_crop_256x256

    def generate(self, n_frames, source_lmk, crop_info, img_rgb, mask_ori, i_d_lst, i_p_paste_lst, x_s,
                 r_s, f_s, x_s_info, x_c_s, eye_ratio_lst, lip_ratio_lst, lip_delta_before_animation):

        i_p_lst = []
        r_d_0, x_d_0_info = None, None
        for i in tqdm(range(n_frames), desc='Animating...', total=n_frames):
            i_d_i = i_d_lst[i]
            x_d_i_info = self.get_kp_info(self._model_sessions, i_d_i, x_s, r_s, x_s_info, lip_delta_before_animation,
                                          run_local=True)
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
                    x_d_i_new = self.stitching(self._model_sessions['s_session'], x_s,
                                               x_d_i_new) + lip_delta_before_animation.reshape(-1,
                                                                                               x_s.shape[
                                                                                                   1],
                                                                                               3)

                else:
                    x_d_i_new = self.stitching(self._model_sessions['s_session'], x_s, x_d_i_new)

            else:
                eyes_delta, lip_delta = None, None

                if self.cfg.flag_eye_retargeting:
                    c_d_eyes_i = eye_ratio_lst[i]
                    combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i,
                                                                             source_lmk)
                    # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                    eyes_delta = self.retarget_eye(self._model_sessions['s_e_session'], x_s, combined_eye_ratio_tensor)
                if self.cfg.flag_lip_retargeting:
                    c_d_lip_i = lip_ratio_lst[i]
                    combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i,
                                                                             source_lmk)
                    # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                    lip_delta = self.retarget_lip(self._model_sessions['s_e_session'], x_s, combined_lip_ratio_tensor)

                if self.cfg.flag_relative:  # use x_s
                    x_d_i_new = x_s + \
                                (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                                (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new + \
                                (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                                (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)

                if self.cfg.flag_stitching:
                    x_d_i_new = self.stitching(self._model_sessions['s_session'], x_s, x_d_i_new)

            i_p_i = self.warp_decode(self._model_sessions, f_s, x_s, x_d_i_new)
            i_p_lst.append(i_p_i)
            i_p_i_to_ori_blend = self.paste_back(i_p_i, crop_info['M_c2o'], img_rgb, mask_ori)
            i_p_paste_lst.append(i_p_i_to_ori_blend)
        return i_p_lst

    def render(self, live_portrait, video_path_or_id=None, image_path=None, real_time=False):
        """
        Video_path_or_id is use for 2 process, please make sure video_id only use for real-time demo
        """
        source_landmark, x_c_s, x_s, f_s, r_s, \
            x_s_info, lip_delta_before_animation, crop_info, \
            img_rgb, imgs_crop_256x256 = live_portrait.prepare_portrait(source_image_path=image_path)
        if real_time:
            cap = cv2.VideoCapture(int(video_path_or_id) if real_time else video_path_or_id)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                x_s, x_d_i_new = live_portrait.get_kp_info(self._model_sessions, frame, x_s, r_s, x_s_info,
                                                           lip_delta_before_animation)
                i_p_i = live_portrait.warp_decode(self._model_sessions, np.array(f_s), np.array(x_s),
                                                  np.array(x_d_i_new))
                if live_portrait.cfg.flag_pasteback:
                    mask_ori = live_portrait.prepare_paste_back(live_portrait.cfg.mask_crop, crop_info['M_c2o'],
                                                                dsize=(img_rgb.shape[1], img_rgb.shape[0]))
                    i_p_i_to_ori_blend = live_portrait.paste_back(i_p_i, crop_info['M_c2o'], img_rgb, mask_ori)
                    cv2.imshow('a', i_p_i_to_ori_blend[:, :, ::-1])
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:

            mask_ori, driving_rgb_lst, i_d_lsts, i_p_paste_lst, _, n_frames, input_eye_ratio_lsts, input_lip_ratio_lsts = live_portrait.process_source_motion(
                img_rgb, video_path_or_id, crop_info, live_portrait.cfg, live_portrait.cropper)

            result = live_portrait.generate(n_frames, source_landmark, crop_info, img_rgb, mask_ori, i_d_lsts,
                                            i_p_paste_lst, x_s, r_s, f_s, x_s_info, x_c_s, input_eye_ratio_lsts,
                                            input_lip_ratio_lsts,
                                            lip_delta_before_animation)
            live_portrait.mkdir('animations')
            frames_concatenated = live_portrait.concat_frames(result, driving_rgb_lst, imgs_crop_256x256)
            wfp_concat = osp.join('animations',
                                  f'{basename(image_path)}--{basename(image_path)}_concat.mp4')
            images2video(frames_concatenated, wfp=wfp_concat)

            wfp = osp.join('animations', f'{basename(image_path)}--{basename(image_path)}.mp4')
            images2video(i_p_paste_lst, wfp=wfp)
