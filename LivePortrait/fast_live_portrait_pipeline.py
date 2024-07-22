import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm
from LivePortrait.face_analyze import FaceCropper
from LivePortrait.commons import load_image_rgb, resize_to_limit, basename, images2video
from LivePortrait.live_portrait import PortraitController


class EfficientLivePortrait(PortraitController):
    def __init__(self, use_tensorrt, half, **kwargs):
        super().__init__(use_tensorrt, half, **kwargs)
        self.cropper = FaceCropper(**kwargs)
        self.config = kwargs

    def prepare_multiple_portrait(self, source_image_path, ref_img):
        # Load and preprocess source image
        img_rgb = load_image_rgb(source_image_path)
        ref_img = load_image_rgb(ref_img)
        img_rgb = resize_to_limit(img_rgb, self.config['ref_max_shape'], self.config['ref_shape_n'])
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

    def prepare_portrait(self, source_image_path):
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

    def generate(self,
                 crop_info,
                 n_frames,
                 i_d_lst,
                 eye_ratio_lst,
                 lip_ratio_lst):

        i_p_lst = []
        i_p_paste_lst = []
        r_d_0, x_d_0_info = None, None
        first_face = crop_info[0]
        first_face_key = list(first_face.keys())[0]  # Get the first key (e.g., 'face_0')
        img_rgb = first_face[first_face_key].get('img_rgb', None)
        for i in tqdm(range(n_frames), desc='Animating...', total=n_frames):
            frame_accumulator = np.zeros_like(img_rgb, dtype=np.float32)
            for crop_f in crop_info:
                # Extract face key
                face_key = list(crop_f.keys())[0]
                face_data = crop_f[face_key]

                i_d_i = i_d_lst[i]
                x_d_i_info = self.get_kp_info(i_d_i, face_data['x_s'], face_data['r_s'], face_data['x_s_info'],
                                              face_data['lip_delta_before_animation'],
                                              run_local=True)
                r_d_i = self.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

                if i == 0:
                    r_d_0 = r_d_i
                    x_d_0_info = x_d_i_info

                if self.config['flag_relative']:
                    r_new = (r_d_i @ r_d_0.permute(0, 2, 1)) @ face_data['r_s']
                    delta_new = face_data['x_s_info']['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                    scale_new = face_data['x_s_info']['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                    t_new = face_data['x_s_info']['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    r_new = r_d_i
                    delta_new = x_d_i_info['exp']
                    scale_new = face_data['x_s_info']['scale']
                    t_new = x_d_i_info['t']

                t_new[..., 2].fill_(0)  # zero tz
                x_d_i_new = scale_new * (face_data['x_c_s'] @ r_new + delta_new) + t_new

                # Algorithm 1:
                if not self.config['flag_stitching'] and not self.config['flag_eye_retargeting'] and not self.config['flag_lip_retargeting']:
                    # without stitching or retargeting
                    if self.config['flag_lip_zero']:
                        x_d_i_new += face_data['lip_delta_before_animation'].reshape(-1, face_data['x_s'].shape[1], 3)
                    else:
                        pass
                elif self.config['flag_stitching'] and not self.config['flag_eye_retargeting'] and not self.config['flag_lip_retargeting']:
                    # with stitching and without retargeting
                    if self.config['flag_lip_zero']:
                        x_d_i_new = self.stitching(self.predictor, face_data['x_s'],
                                                   x_d_i_new) + face_data['lip_delta_before_animation'].reshape(-1,
                                                                                                                face_data[
                                                                                                                    'x_s'].shape[
                                                                                                                    1],
                                                                                                                3)

                    else:
                        x_d_i_new = self.stitching(self.predictor, face_data['x_s'], x_d_i_new)

                else:
                    eyes_delta, lip_delta = None, None

                    if self.config['flag_eye_retargeting']:
                        c_d_eyes_i = eye_ratio_lst[i]
                        combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i,
                                                                                 face_data['source_lmk'])
                        # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                        eyes_delta = self.retarget_eye(self.predictor, face_data['x_s'], combined_eye_ratio_tensor)
                    if self.config['flag_lip_retargeting']:
                        c_d_lip_i = lip_ratio_lst[i]
                        combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i,
                                                                                 face_data['source_lmk'])
                        # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                        lip_delta = self.retarget_lip(self.predictor, face_data['x_s'], combined_lip_ratio_tensor)

                    if self.config['flag_relative']:  # use x_s
                        x_d_i_new = face_data['x_s'] + \
                                    (eyes_delta.reshape(-1, face_data['x_s'].shape[1],
                                                        3) if eyes_delta is not None else 0) + \
                                    (lip_delta.reshape(-1, face_data['x_s'].shape[1],
                                                       3) if lip_delta is not None else 0)
                    else:  # use x_d,i
                        x_d_i_new = x_d_i_new + \
                                    (eyes_delta.reshape(-1, face_data['x_s'].shape[1],
                                                        3) if eyes_delta is not None else 0) + \
                                    (lip_delta.reshape(-1, face_data['x_s'].shape[1],
                                                       3) if lip_delta is not None else 0)

                    if self.config['flag_stitching']:
                        x_d_i_new = self.stitching(self.predictor, face_data['x_s'], x_d_i_new)

                i_p_i = self.warp_decode(face_data['f_s'], face_data['x_s'], x_d_i_new)
                i_p_lst.append(i_p_i)
                i_p_i_to_ori_blend = self.paste_back(i_p_i, face_data['M_c2o'], img_rgb,
                                                     face_data['mask_ori'])
                frame_accumulator += i_p_i_to_ori_blend.astype(np.float32) / len(crop_info)
            frame_accumulator = np.clip(frame_accumulator, 0, 255).astype(np.uint8)
            i_p_paste_lst.append(frame_accumulator)
        return i_p_lst, i_p_paste_lst

    def render(self, video_path_or_id=None, image_path=None, ref_img=None, real_time=False):
        """
        Video_path_or_id is use for 2 process, please make sure video_id only use for real-time demo
        """

        if real_time:
            _, _, x_s, f_s, r_s, \
                x_s_info, lip_delta_before_animation, crop_info, \
                img_rgb, _ = self.prepare_portrait(source_image_path=image_path)
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
            i_d_lsts, _, n_frames, input_eye_ratio_lsts, input_lip_ratio_lsts = self.process_multiple_source_motion(
                video_path_or_id, crop_infos, self.cropper)
            _, i_p_paste_lst = self.generate(
                crop_infos,
                n_frames,
                i_d_lsts,
                input_eye_ratio_lsts,
                input_lip_ratio_lsts)
            self.mkdir('animations')

            wfp = osp.join('animations', f'{basename(image_path)}--{basename(video_path_or_id)}.mp4')
            images2video(i_p_paste_lst, wfp=wfp)
