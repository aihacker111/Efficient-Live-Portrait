from diffusers import StableDiffusionXLControlNetPipeline, \
    ControlNetModel, EulerDiscreteScheduler
from .open_pose import OpenPose
from huggingface_hub import hf_hub_download
import torch


class SDXLLightningOpenPose(OpenPose):
    def __init__(self):
        super().__init__()

    @staticmethod
    def load_pipeline(lcm_steps):
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = f"sdxl_lightning_{lcm_steps}step_lora.safetensors"
        controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base, controlnet=controlnet, torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe.load_lora_weights(hf_hub_download(repo, ckpt))
        pipe.fuse_lora()
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.to('cuda')
        return pipe

    def generate(self, image_path, prompt, negative_prompt, lcm_steps, width, height, seed):
        openpose_image = self.get_pose(image_path)
        pipe = self.load_pipeline(lcm_steps)
        images = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=lcm_steps,
            num_images_per_prompt=1,
            guidance_scale=0,
            image=openpose_image.resize((width, height)),
            generator=torch.manual_seed(seed),
        ).images
        return images[0]
