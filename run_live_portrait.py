# coding: utf-8
# Author: Vo Nguyen An Tin
# Email: tinprocoder0908@gmail.com

from LivePortrait import EfficientLivePortrait
from LivePortrait.commons import save_config_to_yaml
import argparse
import warnings
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")


def main(video_path, source_img, source_video_path, ref_img, max_faces, use_tensorrt, task, half_precision, use_face_id,
         cropping_video, get_source_audio,
         use_diffusion, lcm_steps, prompt, negative_prompt, width, height, seed):
    cfg_yaml = save_config_to_yaml()
    kwargs = OmegaConf.load(cfg_yaml)
    live_portrait = EfficientLivePortrait(use_tensorrt, half_precision, cropping_video, **kwargs)

    if use_face_id:
        live_portrait.render(video_path_or_id=video_path, image_path=source_img, source_video_path=source_video_path,
                             ref_img=ref_img, max_faces=max_faces, task=task, audio_from_source=get_source_audio,
                             use_diffusion=use_diffusion, lcm_steps=lcm_steps, prompt=prompt,
                             negative_prompt=negative_prompt, width=width, height=height, seed=seed)
    else:
        live_portrait.render(video_path_or_id=video_path, image_path=source_img, source_video_path=source_video_path,
                             ref_img=None, max_faces=max_faces, task=task, audio_from_source=get_source_audio,
                             use_diffusion=use_diffusion, lcm_steps=lcm_steps, prompt=prompt,
                             negative_prompt=negative_prompt, width=width, height=height, seed=seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live Portrait Rendering Script')
    parser.add_argument('--driving_video', type=str, required=True,
                        help='Path to the driving video or your webcam id')
    parser.add_argument('--source_image', type=str, required=False, help='Path to the source image')
    parser.add_argument('--source_video', type=str, required=False, help='Path to the source video')
    parser.add_argument('--condition_image', type=str, help='Path to the ref image (use only for Face-ID mode')
    parser.add_argument('--max_faces', type=int, default=1, required=False,
                        help='Enter number of faces you want to extract in Video Motion')
    parser.add_argument('--run_time', action='store_true', help='Turn on TensorRT mode')
    parser.add_argument('--half_precision', action='store_true', help='Half Precision on TensorRT mode')
    parser.add_argument('--task', type=str, default='image',
                        help='Options for real-time webcam or source_image or source_video demo')
    parser.add_argument('--use_face_id', action='store_true', help='Use reference image for face ID')
    parser.add_argument('--cropping_video', action='store_true', required=False, help='Auto Cropping 1:1 Video')
    parser.add_argument('--get_source_audio', action='store_true', required=False,
                        help='If you want to get the output with audio from source video')
    parser.add_argument('--use_diffusion', action='store_true', required=False,
                        help='Enable SDXL-Lightning Controlnet OpenPose')
    parser.add_argument('--lcm_steps', type=int, default=4, required=False,
                        help='Step for Img2Img translation: 1 or 2 or 4 or 8')
    parser.add_argument('--prompt', type=str, required=False, help='Positive Prompt')
    parser.add_argument('--negative_prompt', type=str, required=False,
                        default='deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation',
                        help='Negative Prompt')
    parser.add_argument('--width', type=int, default=1024, required=False, help='width of image output')
    parser.add_argument('--height', type=int, default=1024, required=False, help='height of image output')
    parser.add_argument('--seed', type=int, default=6681501646976930000, required=False, help='Positive Prompt')

    args = parser.parse_args()

    main(args.driving_video, args.source_image, args.source_video, args.condition_image, args.max_faces, args.run_time,
         args.task, args.half_precision, args.use_face_id, args.cropping_video, args.get_source_audio,
         args.use_diffusion, args.lcm_steps, args.prompt, args.negative_prompt, args.width, args.height, args.seed)
