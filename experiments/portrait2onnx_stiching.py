import torch
from Face2Vid.utils.helper import load_model
from Face2Vid.commons.config.inference_config import InferenceConfig
from Face2Vid.utils.rprint import rlog as log
import yaml


def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat


# # Load the inference configuration
cfg = InferenceConfig()

# Load the model configuration
with open(cfg.models_config, 'r') as config_file:
    model_config = yaml.safe_load(config_file)

# Load the motion extractor model
stitching_retargeting_module = load_model(cfg.checkpoint_S, model_config, cfg.device, 'stitching_retargeting_module')
log('Load warping_module done.')

# Set the model to evaluation mode
stitching_retargeting_module['eye'].eval()

# Create dummy input data
kp_source = torch.randn(1, 21, 3)  # Example shape, adjust as necessary
# lip_ratio = torch.randn(1, 2)
eye_ratio = torch.randn(1, 3)
# kp_driving = torch.randn(1, 21, 3)  # Example shape, adjust as necessary
feat_stiching = concat_feat(kp_source, eye_ratio)
# Export the model
torch.onnx.export(
    stitching_retargeting_module['eye'],
    feat_stiching,
    "stitching_retargeting_eye.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# import onnxruntime as ort
# import torch
# import numpy as np
# ort_session = ort.InferenceSession('../live_portraiet_onnx/onnx/stitching_retargeting.onnx')
#
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
#
# kp_source = torch.randn(1, 21, 3)  # Example shape, adjust as necessary
# kp_driving = torch.randn(1, 21, 3)  # Example shape, adjust as necessary
# feat_stiching = concat_feat(kp_source, kp_driving)
# # Prepare inputs for ONNX runtime
#
# # Run inference
# ort_outs = ort_session.run(None, {'input': np.array(feat_stiching)})
#
# # Process the output as necessary
# output = ort_outs[0]
# print("Inference output:", output)
