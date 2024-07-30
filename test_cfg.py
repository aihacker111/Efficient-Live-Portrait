from LivePortrait.controlnet_sdxl_lightning.sdxl_lightning import SDXLLightningOpenPose

sdxl_lightning_openpose = SDXLLightningOpenPose()
image_path = ''
prompt = ''
negative_prompt = 'deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation'
lcm_steps = 4
width = 1024
height = 1024
seed = 20242024
result = sdxl_lightning_openpose.generate(image_path=image_path, prompt=prompt, negative_prompt=negative_prompt, lcm_steps=lcm_steps, width=width, height=height, seed=seed)
result.save('result.png')