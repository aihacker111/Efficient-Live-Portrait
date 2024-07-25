import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm
from LivePortrait.commons import load_image_rgb, resize_to_limit, basename, images2video
from LivePortrait.live_portrait import PortraitController


class EfficientLivePortrait(PortraitController):
    def __init__(self, use_tensorrt, half, cropping_video, **kwargs):
        super().__init__(use_tensorrt, half, **kwargs)
        self.config = kwargs
        self.cropping_video = cropping_video

    def prepare_multiple_portrait(self, source_image_path, ref_img):
        # Load and preprocess source image
        img_rgb = load_image_rgb(source_image_path)
        img_rgb = resize_to_limit(img_rgb, self.config['ref_max_shape'], self.config['ref_shape_n'])
        if ref_img is not None:
            ref_img = load_image_rgb(ref_img)
            ref_img = resize_to_limit(ref_img, self.config['ref_max_shape'], self.config['ref_shape_n'])
        # Crop multiple faces
        crop_info = self.cropper.crop_multiple_faces(img_rgb, ref_img)
        # Initialize updated crop_info list
        updated_crop_info = []

        for crop_f in crop_info:
            # Extract face key
            face_key = list(crop_f.keys())[0]
            face_data = crop_f[face_key]

            source_lmk = face_data.get('lmk_crop', None)
            img_crop_256x256 = face_data.get('img_crop_256x256', None)

            if self.config['flag_do_crop']:
                i_s = self.prepare_source_image(img_crop_256x256)
            else:
                i_s = self.prepare_source_image(img_rgb)

            x_s_info = self.get_kp_info(i_s, x_s=None, r_s=None, x_s_info=None,
                                        lip_delta_before_animation=None, single_image=True)
            x_c_s = x_s_info['kp']
            r_s = self.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            f_s = self.get_3d_feature(i_s)
            x_s = self.transform_keypoint(x_s_info)

            lip_delta_before_animation = None
            if self.config['flag_lip_zero']:
                c_d_lip_before_animation = [0.]
                combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(
                    c_d_lip_before_animation, source_lmk)
                if combined_lip_ratio_tensor_before_animation[0][0] < self.config['lip_zero_threshold']:
                    self.config['flag_lip_zero'] = False
                else:
                    lip_delta_before_animation = self.retarget_lip(self.predictor, x_s,
                                                                   combined_lip_ratio_tensor_before_animation)

            # Update the face_data with new variables
            face_data.update({
                'x_c_s': x_c_s,
                'r_s': r_s,
                'f_s': f_s,
                'x_s': x_s,
                'x_s_info': x_s_info,
                'lip_delta_before_animation': lip_delta_before_animation,
                'img_rgb': img_rgb,
                'img_crop_256x256': img_crop_256x256
            })

            # Add the updated face_data to the updated_crop_info list
            updated_crop_info.append({face_key: face_data})
        # Return the updated list of dictionaries
        return updated_crop_info

    def prepare_webcam_portrait(self, source_image_path):
        # Load and preprocess source image
        img_rgb = load_image_rgb(source_image_path)
        img_rgb = resize_to_limit(img_rgb, self.config['ref_max_shape'], self.config['ref_shape_n'])
        # log(f"Load source image from {source_image_path}")
        crop_info = self.cropper.crop_single_image(img_rgb)
        source_lmk = crop_info['lmk_crop']

        _, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']

        if self.config['flag_do_crop']:
            i_s = self.prepare_source_image(img_crop_256x256)
        else:
            i_s = self.prepare_source_image(img_rgb)

        x_s_info = self.get_kp_info(i_s, x_s=None, r_s=None, x_s_info=None,
                                    lip_delta_before_animation=None, single_image=True)
        x_c_s = x_s_info['kp']
        r_s = self.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.get_3d_feature(i_s)
        x_s = self.transform_keypoint(x_s_info)

        lip_delta_before_animation = None
        if self.config['flag_lip_zero']:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(
                c_d_lip_before_animation, crop_info['lmk_crop'])
            if combined_lip_ratio_tensor_before_animation[0][0] < self.config['lip_zero_threshold']:
                self.config['flag_lip_zero'] = False
            else:
                lip_delta_before_animation = self.retarget_lip(self.predictor, x_s,
                                                               combined_lip_ratio_tensor_before_animation)
        return source_lmk, x_c_s, x_s, f_s, r_s, x_s_info, lip_delta_before_animation, crop_info, img_rgb, img_crop_256x256

    def generate(self, input_list):
        i_p_lst = []
        i_p_paste_lst = []
        r_d_0 = None
        x_d_0_info = None
        # Initialize dictionaries to hold motion and image data
        face_motion_data = {}
        face_image_data = {}

        # Gather all motion and image data
        for input_dict in input_list:
            for key, value in input_dict.items():
                if 'face_motion' in key:
                    face_motion_data[key] = value
                elif 'face_image' in key:
                    face_image_data[key] = value

        if not face_image_data:
            return [], []  # Return empty lists if no face image data is available

        # Get the number of frames and initialize the base image
        n_frames = next(iter(face_motion_data.values()))['n_frames']
        original_fps = next(iter(face_motion_data.values()))['fps']
        # Process each frame
        for i in tqdm(range(n_frames), desc='Animating...', total=n_frames):
            img_rgb = next(iter(face_image_data.values()))['img_rgb']
            for face_key in face_image_data.keys():
                face_index = int(face_key.split('_')[-1])  # Extract face_index from key
                face_data = face_image_data[face_key]
                motion_key = f"face_motion_{face_index}"
                motion_data = face_motion_data[motion_key]
                # Extract motion data for the current frame
                i_d_i = motion_data['i_d_lst'][i]
                x_d_i_info = self.get_kp_info(i_d_i, face_data['x_s'], face_data['r_s'], face_data['x_s_info'],
                                              face_data['lip_delta_before_animation'], run_local=True)
                r_d_i = self.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

                if i == 0:
                    r_d_0 = r_d_i
                    x_d_0_info = x_d_i_info

                # Compute new rotation, scale, and translation
                if self.config['flag_relative']:
                    r_new = (r_d_i @ r_d_0.permute(0, 2, 1)) @ face_data['r_s']
                    delta_new = face_data['x_s_info']['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                    scale_new = face_data['x_s_info']['scale'] if self.cropping_video else face_data['x_s_info']['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                    t_new = face_data['x_s_info']['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    r_new = r_d_i
                    delta_new = x_d_i_info['exp']
                    scale_new = face_data['x_s_info']['scale']
                    t_new = x_d_i_info['t']

                t_new[..., 2].fill_(0)  # Zero tz
                x_d_i_new = scale_new * (face_data['x_c_s'] @ r_new + delta_new) + t_new

                # Apply stitching and retargeting
                if not self.config['flag_stitching'] and not self.config['flag_eye_retargeting'] and not self.config[
                    'flag_lip_retargeting']:
                    if self.config['flag_lip_zero']:
                        x_d_i_new += face_data['lip_delta_before_animation'].reshape(-1, face_data['x_s'].shape[1], 3)
                elif self.config['flag_stitching'] and not self.config['flag_eye_retargeting'] and not self.config[
                    'flag_lip_retargeting']:
                    if self.config['flag_lip_zero']:
                        x_d_i_new = self.stitching(self.predictor, face_data['x_s'], x_d_i_new) + face_data[
                            'lip_delta_before_animation'].reshape(-1, face_data['x_s'].shape[1], 3)
                    else:
                        x_d_i_new = self.stitching(self.predictor, face_data['x_s'], x_d_i_new)
                else:
                    eyes_delta, lip_delta = None, None
                    if self.config['flag_eye_retargeting']:
                        c_d_eyes_i = motion_data['input_eye_ratio_lst'][i]
                        combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i, face_data['source_lmk'])
                        eyes_delta = self.retarget_eye(self.predictor, face_data['x_s'], combined_eye_ratio_tensor)
                    if self.config['flag_lip_retargeting']:
                        c_d_lip_i = motion_data['input_lip_ratio_lst'][i]
                        combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i, face_data['source_lmk'])
                        lip_delta = self.retarget_lip(self.predictor, face_data['x_s'], combined_lip_ratio_tensor)

                    if self.config['flag_relative']:  # Use x_s
                        x_d_i_new = face_data['x_s'] + \
                                    (eyes_delta.reshape(-1, face_data['x_s'].shape[1],
                                                        3) if eyes_delta is not None else 0) + \
                                    (lip_delta.reshape(-1, face_data['x_s'].shape[1],
                                                       3) if lip_delta is not None else 0)
                    else:  # Use x_d,i
                        x_d_i_new = x_d_i_new + \
                                    (eyes_delta.reshape(-1, face_data['x_s'].shape[1],
                                                        3) if eyes_delta is not None else 0) + \
                                    (lip_delta.reshape(-1, face_data['x_s'].shape[1],
                                                       3) if lip_delta is not None else 0)

                    if self.config['flag_stitching']:
                        x_d_i_new = self.stitching(self.predictor, face_data['x_s'], x_d_i_new)

                # Decode and accumulate frame results
                i_p_i = self.warp_decode(face_data['f_s'], face_data['x_s'], x_d_i_new)

                img_rgb = self.paste_back(i_p_i, face_data['M_c2o'], img_rgb, face_data['mask_ori'])

                i_p_paste_lst.append(img_rgb)

        return i_p_lst, i_p_paste_lst, original_fps

    def render(self, video_path_or_id, image_path, ref_img, max_faces, real_time):
        """
        Video_path_or_id is use for 2 process, please make sure video_id only use for real-time demo
        """

        if real_time:
            _, _, x_s, f_s, r_s, \
                x_s_info, lip_delta_before_animation, crop_info, \
                img_rgb, _ = self.prepare_webcam_portrait(source_image_path=image_path)
            cap = cv2.VideoCapture(int(video_path_or_id) if real_time else video_path_or_id)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                x_s, x_d_i_new = self.get_kp_info(frame, x_s, r_s, x_s_info,
                                                  lip_delta_before_animation)
                i_p_i = self.warp_decode(f_s, np.array(x_s),
                                         np.array(x_d_i_new))
                if self.config['flag_pasteback']:
                    mask_ori = self.prepare_paste_back(self.config['mask_crop'], crop_info['M_c2o'])
                    i_p_i_to_ori_blend = self.paste_back(i_p_i, crop_info['M_c2o'], img_rgb, mask_ori)
                    cv2.imshow('a', i_p_i_to_ori_blend[:, :, ::-1])
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            crop_infos = self.prepare_multiple_portrait(source_image_path=image_path, ref_img=ref_img)
            source_motion_dict = self.process_multiple_source_motion(
                video_path_or_id, crop_infos, max_faces, self.cropper, cropping_video=self.cropping_video)
            _, i_p_paste_lst, original_fps = self.generate(source_motion_dict)
            self.mkdir('animations')

            wfp = osp.join('animations', f'{basename(image_path)}--{basename(video_path_or_id)}.mp4')
            wfp_audio = osp.join('animations', f'{basename(image_path)}--{basename(video_path_or_id)}_audio.mp4')
            images2video(i_p_paste_lst, wfp=wfp, video_path_original=video_path_or_id, wfp_audio=wfp_audio, fps=original_fps, add_video_func=self.add_audio_to_video)
