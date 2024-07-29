# coding: utf-8
# Author: Vo Nguyen An Tin
# Email: tinprocoder0908@gmail.com

from LivePortrait.face_analyze.modules import FaceAnalysis, LandmarkRunner
from LivePortrait.commons.utils.utils import load_image_rgb, resize_to_limit
from LivePortrait.face_analyze.utils.crop import crop_image, contiguous
import numpy as np
import os.path as osp
from typing import List, Union, Tuple
from dataclasses import dataclass, field
import cv2
import imageio
from tqdm import tqdm

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


@dataclass
class Trajectory:
    start: int = -1  # start frame
    end: int = -1  # end frame
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    bbox_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # bbox list
    M_c2o_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # M_c2o list

    frame_rgb_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame list
    lmk_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame crop list


class FaceCropper:
    def __init__(self, **kwargs) -> None:
        device_id = kwargs.get('device_id', 0)
        self.cfg = kwargs
        self.landmark_runner = LandmarkRunner(
            ckpt_path=self.cfg['ckpt_landmark'],
            onnx_provider='cuda',
            device_id=device_id
        )
        self.landmark_runner.warmup()

        self.face_analysis_wrapper = FaceAnalysis(
            det_path=self.cfg['ckpt_det'],
            rec_path=self.cfg['ckpt_arc_face'],
            landmark_106_path=self.cfg['ckpt_landmark_106']
        )
        self.face_analysis_wrapper.prepare(ctx_id=device_id, det_size=(512, 512))
        self.face_analysis_wrapper.warmup()

    def crop_single_image(self, obj, **kwargs):
        direction = kwargs.get('direction', 'large-small')
        img_rgb = None
        # crop and align a single image
        if isinstance(obj, str):
            img_rgb = load_image_rgb(obj)
        elif isinstance(obj, np.ndarray):
            img_rgb = obj

        src_face = self.face_analysis_wrapper.get_detector(
            img_rgb,
            flag_do_landmark_2d_106=True,
            direction=direction
        )

        src_face = src_face[0]
        pts = src_face.landmark_2d_106
        # crop the face
        ret_dct = crop_image(
            img_rgb,  # ndarray
            pts,  # 106x2 or Nx2
            dsize=kwargs.get('dsize', 512),
            scale=kwargs.get('scale', 2.3),
            vy_ratio=kwargs.get('vy_ratio', -0.15),
        )
        # update a 256x256 version for network input or else
        ret_dct['img_crop_256x256'] = cv2.resize(ret_dct['img_crop'], (256, 256), interpolation=cv2.INTER_AREA)
        ret_dct['pt_crop_256x256'] = ret_dct['pt_crop'] * 256 / kwargs.get('dsize', 512)

        recon_ret = self.landmark_runner.run(img_rgb, pts)
        lmk = recon_ret['pts']
        ret_dct['lmk_crop'] = lmk

        return ret_dct

    def crop_multiple_faces(self, obj, ref_img, **kwargs):
        direction = kwargs.get('direction', 'large-small')
        img_rgb = None
        ret_ref = None
        # Load the image
        if isinstance(obj, str):
            img_rgb = load_image_rgb(obj)
        elif isinstance(obj, np.ndarray):
            img_rgb = obj
        if ref_img is not None:
            if isinstance(obj, str):
                ref_img = load_image_rgb(ref_img)
            ref_faces = self.face_analysis_wrapper.get_detector(ref_img,
                                                                flag_do_landmark_2d_106=True,
                                                                direction=direction)
            ref_face = ref_faces[0]
            pts_ref = ref_face.landmark_2d_106
            # crop the face
            ret_ref = crop_image(
                ref_img,  # ndarray
                pts_ref,  # 106x2 or Nx2
                dsize=kwargs.get('dsize', 512),
                scale=kwargs.get('scale', 2.3),
                vy_ratio=kwargs.get('vy_ratio', -0.15),
            )
        # Detect multiple faces
        faces = self.face_analysis_wrapper.get_detector(
            img_rgb,
            flag_do_landmark_2d_106=True,
            direction=direction
        )
        crops_info = []
        for i in range(len(faces)):
            pts = faces[i].landmark_2d_106
            # Crop the face
            ret_dct = crop_image(
                img_rgb,  # ndarray
                pts,
                dsize=kwargs.get('dsize', 512),
                scale=kwargs.get('scale', 2.3),
                vy_ratio=kwargs.get('vy_ratio', -0.15),
            )
            # Update a 256x256 version for network input or else
            ret_dct['img_crop_256x256'] = cv2.resize(ret_dct['img_crop'], (256, 256), interpolation=cv2.INTER_AREA)
            ret_dct['pt_crop_256x256'] = ret_dct['pt_crop'] * 256 / kwargs.get('dsize', 512)

            # Run the landmark runner
            recon_ret = self.landmark_runner.run(img_rgb, pts)
            lmk = recon_ret['pts']
            ret_dct['lmk_crop'] = lmk
            crops_info.append({f'face_{i}': ret_dct})
        info_dict = self.face_analysis_wrapper.get_face_id(ret_ref, crops_info)
        return info_dict

    def get_retargeting_lmk_info(self, driving_rgb_lst):
        driving_lmk_lst = []
        for driving_image in driving_rgb_lst:
            ret_dct = self.crop_single_image(driving_image)
            driving_lmk_lst.append(ret_dct['lmk_crop'])
        return driving_lmk_lst

    @staticmethod
    def load_video(video_info, n_frames=-1):
        reader = imageio.get_reader(video_info, "ffmpeg")
        # Extract FPS from video metadata
        fps = reader.get_meta_data()['fps']
        ret = []
        for idx, frame_rgb in enumerate(reader):
            if 0 < n_frames <= idx:
                break
            ret.append(frame_rgb)

        reader.close()
        return ret, fps

    def crop_source_video(self,
                          driving_video,
                          max_faces,
                          use_for_vid2vid=False,
                          **kwargs):
        # os.makedirs('/Users/macbook/Downloads/Efficient-Face2Vid-Portrait/colab/img_crop', exist_ok=True)
        source_rgb_lst, fps = self.load_video(driving_video)
        source_rgb_lst = [resize_to_limit(img, 1280, 2) for img in
                          source_rgb_lst]
        """Tracking based landmarks/alignment and cropping"""
        trajectory_dict = {}
        direction = kwargs.get("direction", "large-small")
        trajectory = Trajectory()
        for idx, frame_rgb in tqdm(enumerate(source_rgb_lst), total=len(source_rgb_lst), desc="Processing Frames"):
            if idx == 0 or all(traj.start == -1 for traj in trajectory_dict.values()):
                src_faces = self.face_analysis_wrapper.get_detector(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=direction,
                    max_face_num=0,
                )
                for face_id, src_face in enumerate(src_faces[:max_faces]):
                    lmk = src_face.landmark_2d_106
                    lmk = self.landmark_runner.run(frame_rgb, lmk)
                    trajectory.start, trajectory.end = idx, idx
                    trajectory.lmk_lst.append(lmk)
                    trajectory_dict[face_id] = trajectory
            else:
                for face_id, trajectory in trajectory_dict.items():
                    if face_id >= max_faces:
                        break
                    if trajectory.end == idx - 1:
                        lmk = self.landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1]['pts'])
                        trajectory.end = idx
                        trajectory.lmk_lst.append(lmk)

            for face_id, trajectory in list(trajectory_dict.items())[:max_faces]:
                if idx >= trajectory.start:
                    lmk = trajectory.lmk_lst[-1]

                    # Crop the face
                    ret_dct = crop_image(
                        frame_rgb,  # ndarray
                        lmk['pts'],  # 106x2 or Nx2
                        dsize=self.cfg['dsize'],
                        scale=self.cfg['scale'],
                        vx_ratio=self.cfg['vx_ratio'],
                        vy_ratio=self.cfg['vy_ratio'],
                        flag_do_rot=self.cfg['flag_do_rot'],
                    )
                    lmk_2 = self.landmark_runner.run(frame_rgb, lmk['pts'])
                    ret_dct["lmk_crop"] = lmk_2['pts']

                    # Ensure landmarks are in the correct format
                    if isinstance(ret_dct["lmk_crop"], dict):
                        ret_dct["lmk_crop"] = np.array(list(ret_dct["lmk_crop"].values()))
                    elif not isinstance(ret_dct["lmk_crop"], np.ndarray):
                        ret_dct["lmk_crop"] = np.array(ret_dct["lmk_crop"])

                    # Update a 256x256 version for network input
                    ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256),
                                                             interpolation=cv2.INTER_AREA)
                    ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.cfg['dsize']
                    trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop_256x256"])
                    trajectory.lmk_crop_lst.append(ret_dct["lmk_crop_256x256"])
                    trajectory.M_c2o_lst.append(ret_dct['M_c2o'])

        if use_for_vid2vid:
            return source_rgb_lst, fps
        return [
            {
                f"face_control_{face_id}": {
                    "fps": fps,
                    "frame_crop_lst": trajectory.frame_rgb_crop_lst,
                    "lmk_crop_lst": trajectory.lmk_crop_lst,
                    "M_c2o_lst": trajectory.M_c2o_lst,
                }
            } for face_id, trajectory in trajectory_dict.items()
        ]

    def calc_lmks_from_cropped_video(self, driving_video, use_for_vid2vid=False, **kwargs):
        """Tracking based landmarks/alignment and cropping"""
        source_rgb_lst, fps = self.load_video(driving_video)
        source_rgb_lst = [resize_to_limit(img, 1280, 2) for img in source_rgb_lst]
        trajectory = Trajectory()
        trajectory_dict = {}
        direction = kwargs.get("direction", "large-small")

        for idx, frame_rgb in tqdm(enumerate(source_rgb_lst), total=len(source_rgb_lst), desc="Processing Frames"):
            if idx == 0 or all(traj.start == -1 for traj in trajectory_dict.values()):
                src_face = self.face_analysis_wrapper.get_detector(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=direction,
                    max_face_num=0,
                )
                if len(src_face) == 0:
                    print(f"No face detected in the frame #{idx}")
                    raise Exception(f"No face detected in the frame #{idx}")
                elif len(src_face) > 1:
                    print(
                        f"More than one face detected in the driving frame_{idx}, only pick one face by rule {direction}.")
                src_face = src_face[0]
                lmk = src_face.landmark_2d_106
                lmk = self.landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
                trajectory.lmk_lst.append(lmk)
                trajectory_dict[0] = trajectory
            else:
                lmk = self.landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1]['pts'])
                trajectory.end = idx
                trajectory.lmk_lst.append(lmk)
            trajectory.frame_rgb_crop_lst.append(cv2.resize(frame_rgb, (256, 256)))
        if use_for_vid2vid:
            return source_rgb_lst, fps
        return [
            {
                f"face_control_{face_id}": {
                    "fps": fps,
                    "frame_crop_lst": trajectory.frame_rgb_crop_lst,  # Empty list in this case
                    "lmk_crop_lst": trajectory.lmk_lst,  # Using lmk_lst as lmk_crop_lst
                    "M_c2o_lst": trajectory.M_c2o_lst,  # Empty list in this case
                }
            } for face_id, trajectory in trajectory_dict.items()
        ]
