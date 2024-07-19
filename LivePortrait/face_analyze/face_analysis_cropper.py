# coding: utf-8
from LivePortrait.face_analyze.modules import FaceAnalysis, LandmarkRunner
from LivePortrait.commons.utils.utils import load_image_rgb
from LivePortrait.face_analyze.utils.crop import crop_image

import numpy as np
import os.path as osp
from typing import List, Union, Tuple
from dataclasses import dataclass, field
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


@dataclass
class Trajectory:
    start: int = -1  # 起始帧 闭区间
    end: int = -1  # 结束帧 闭区间
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # lmk list
    bbox_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # bbox list
    frame_rgb_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame list
    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)  # frame crop list


class FaceCropper(object):
    def __init__(self, **kwargs) -> None:
        device_id = kwargs.get('device_id', 0)
        cfg = kwargs
        self.landmark_runner = LandmarkRunner(
            ckpt_path=cfg['ckpt_landmark'],
            onnx_provider='cuda',
            device_id=device_id
        )
        self.landmark_runner.warmup()

        self.face_analysis_wrapper = FaceAnalysis(
            det_path=cfg['ckpt_det'],
            landmark_106_path=cfg['ckpt_landmark_106']
        )
        self.face_analysis_wrapper.prepare(ctx_id=device_id, det_size=(512, 512))
        self.face_analysis_wrapper.warmup()

        self.crop_cfg = kwargs.get('crop_cfg', None)

    def update_config(self, user_args):
        for k, v in user_args.items():
            if hasattr(self.crop_cfg, k):
                setattr(self.crop_cfg, k, v)

    def crop_single_image(self, obj, **kwargs):
        direction = kwargs.get('direction', 'large-small')
        img_rgb = None
        # crop and align a single image
        if isinstance(obj, str):
            img_rgb = load_image_rgb(obj)
        elif isinstance(obj, np.ndarray):
            img_rgb = obj

        src_face = self.face_analysis_wrapper.get(
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

    def get_retargeting_lmk_info(self, driving_rgb_lst):
        # TODO: implement a tracking-based version
        driving_lmk_lst = []
        for driving_image in driving_rgb_lst:
            ret_dct = self.crop_single_image(driving_image)
            driving_lmk_lst.append(ret_dct['lmk_crop'])
        return driving_lmk_lst
