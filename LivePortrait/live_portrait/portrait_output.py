# coding: utf-8
# Author: Vo Nguyen An Tin
# Email: tinprocoder0908@gmail.com

import numpy as np
import cv2
import os.path as osp
import os
import subprocess
from rich.progress import track
from .transform_3d_point import Transform3DFunction

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # NOTE: enforce single thread

DTYPE = np.float32
CV2_INTERP = cv2.INTER_LINEAR


class ParsingPaste(Transform3DFunction):
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse_output(out) -> np.ndarray:
        """ construct the output as standard
        return: 1xHxWx3, uint8
        """
        out = np.transpose(out, [0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)  # clip to 0~1
        out = np.clip(out * 255, 0, 255).astype(np.uint8)  # 0~1 -> 0~255

        return out

    def prepare_paste_back(self, crop_m_c2o, dsize):
        """prepare mask for later image paste back
        """
        mask_crop = cv2.imread(self.make_abs_path('./resources/mask_template.png'), cv2.IMREAD_COLOR)
        mask_ori = self._transform_img(mask_crop, crop_m_c2o, dsize)
        mask_ori = mask_ori.astype(np.float32) / 255.
        return mask_ori

    def paste_back(self, image_to_processed, crop_m_c2o, rgb_ori, mask_ori):
        """paste back the image
        """
        dsize = (rgb_ori.shape[1], rgb_ori.shape[0])
        result = self._transform_img(image_to_processed, crop_m_c2o, dsize=dsize)
        result = mask_ori * result + (1 - mask_ori) * rgb_ori
        result = np.clip(result, 0, 255).astype('uint8')
        return result

    @staticmethod
    def make_abs_path(fn):
        return osp.join(osp.dirname(osp.realpath(__file__)), fn)

    # @staticmethod
    # def _transform_img(img, M, dsize, flags=CV2_INTERP, borderMode=None):
    #     """ conduct similarity or affine transformation to the image, do not do border operation!
    #     img:
    #     M: 2x3 matrix or 3x3 matrix
    #     dsize: target shape (width, height)
    #     """
    #     if isinstance(dsize, tuple) or isinstance(dsize, list):
    #         _dsize = tuple(dsize)
    #     else:
    #         _dsize = (dsize, dsize)
    #
    #     if borderMode is not None:
    #         return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags, borderMode=borderMode,
    #                               borderValue=(0, 0, 0))
    #     else:
    #         return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)
    @staticmethod
    def _transform_img(img, M, dsize, scale=1.0, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0)):
        """
        Conduct similarity or affine transformation to the image, with scaling consideration.

        Parameters:
        img: Input image.
        M: 2x3 or 3x3 transformation matrix.
        dsize: Target shape (width, height).
        scale: Scale factor for the output image.
        flags: Interpolation flags (default is cv2.INTER_LINEAR).
        borderMode: Pixel extrapolation method (default is cv2.BORDER_CONSTANT).
        borderValue: Value used in case of a constant border (default is (0, 0, 0)).

        Returns:
        Transformed image.
        """
        # Adjust dsize according to the scale factor
        if isinstance(dsize, (tuple, list)):
            _dsize = (int(dsize[0] * scale), int(dsize[1] * scale))
        else:
            _dsize = (int(dsize * scale), int(dsize * scale))

        # Perform the transformation
        if M.shape[0] == 3:  # If 3x3 matrix, use warpPerspective
            return cv2.warpPerspective(img, M, dsize=_dsize, flags=flags, borderMode=borderMode,
                                       borderValue=borderValue)
        else:  # If 2x3 matrix, use warpAffine
            return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags, borderMode=borderMode,
                                  borderValue=borderValue)

    def exec_cmd(self, cmd):
        return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def add_audio_to_video(self, silent_video_path: str, audio_video_path: str, output_video_path: str):
        cmd = [
            'ffmpeg',
            '-y',
            '-i', f'"{silent_video_path}"',
            '-i', f'"{audio_video_path}"',
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'copy',
            '-shortest',
            f'"{output_video_path}"'
        ]

        try:
            self.exec_cmd(' '.join(cmd))
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")

    @staticmethod
    def concat_frames(i_p_lst, driving_rgb_lst):
        out_lst = []

        # Iterate over each list of driving frames and corresponding i_p frames
        for frame_index in range(len(i_p_lst)):
            source_image_drive = i_p_lst[frame_index]

            # Concatenate corresponding driving frames from each list
            for driving_frames in driving_rgb_lst:
                image_drive = driving_frames[frame_index]
                # Resize images to match source_image_drive shape
                h, w, _ = source_image_drive.shape
                image_drive_resized = cv2.resize(image_drive, (w, h))

                # Concatenate images horizontally
                frame = np.concatenate((image_drive_resized, source_image_drive), axis=1)
                out_lst.append(frame)
        return out_lst

    @staticmethod
    def mkdir(d, log=False):
        # return self-assined `d`, for one line code
        if not osp.exists(d):
            os.makedirs(d, exist_ok=True)
            if log:
                print(f"Make dir: {d}")
        return d
