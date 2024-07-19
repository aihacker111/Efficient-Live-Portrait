import argparse
import warnings
from omegaconf import OmegaConf
from LivePortrait import EfficientLivePortrait
from LivePortrait.commons import save_config_to_yaml

warnings.filterwarnings("ignore")


def main(video_path, source_img, use_tensorrt, real_time):
    cfg_yaml = save_config_to_yaml()
    kwargs = OmegaConf.load(cfg_yaml)
    live_portrait = EfficientLivePortrait(use_tensorrt, **kwargs)
    live_portrait.render(live_portrait, video_path_or_id=video_path, image_path=source_img, real_time=real_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live Portrait Rendering Script')
    parser.add_argument('-v', '--video_path_or_webcam_id', type=str, required=True, help='Path to the driving video or your webcam id')
    parser.add_argument('-i', '--source_img', type=str, required=True, help='Path to the source image')
    parser.add_argument('-e', '--use_tensorrt', action='store_true', help='Turn on TensorRT mode')
    parser.add_argument('-r', '--real_time', action='store_true', help='Enable real-time webcam demo')
    args = parser.parse_args()

    main(args.video_path_or_webcam_id, args.source_img, args.use_tensorrt, args.real_time)

