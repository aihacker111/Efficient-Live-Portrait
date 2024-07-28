# coding: utf-8
# Author: Vo Nguyen An Tin
# Email: tinprocoder0908@gmail.com

from LivePortrait import EfficientLivePortrait
from LivePortrait.commons import save_config_to_yaml
import argparse
import warnings
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")


def main(video_path, source_img, source_video_path, ref_img, max_faces, use_tensorrt, mode, half_precision, use_face_id, cropping_video):
    cfg_yaml = save_config_to_yaml()
    kwargs = OmegaConf.load(cfg_yaml)
    live_portrait = EfficientLivePortrait(use_tensorrt, half_precision, cropping_video, **kwargs)

    if use_face_id:
        live_portrait.render(video_path_or_id=video_path, image_path=source_img, source_video_path=source_video_path, ref_img=ref_img, max_faces=max_faces, mode=mode)
    else:
        live_portrait.render(video_path_or_id=video_path, image_path=source_img, source_video_path=source_video_path, ref_img=None, max_faces=max_faces, mode=mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live Portrait Rendering Script')
    parser.add_argument('--driving_video', type=str, required=True,
                        help='Path to the driving video or your webcam id')
    parser.add_argument('--source_image', type=str, required=False, help='Path to the source image')
    parser.add_argument('--source_video', type=str, required=False, help='Path to the source video')
    parser.add_argument('--condition_image', type=str, help='Path to the ref image')
    parser.add_argument('--max_faces', type=int, default=1, required=False, help='Enter number of faces you want to extract in Video Motion')
    parser.add_argument('--run_time', action='store_true', help='Turn on TensorRT mode')
    parser.add_argument('--half_precision', action='store_true', help='Half Precision on TensorRT mode')
    parser.add_argument('--mode', type=str, default='image', help='Enable real-time webcam demo')
    parser.add_argument('--use_face_id', action='store_true', help='Use reference image for face ID')
    parser.add_argument('--cropping_video', action='store_true', required=False, help='Auto Cropping 1:1 Video')
    args = parser.parse_args()

    main(args.driving_video, args.source_image, args.source_video, args.condition_image, args.max_faces, args.run_time, args.mode, args.half_precision, args.use_face_id, args.cropping_video)
