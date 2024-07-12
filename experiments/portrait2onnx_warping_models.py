import os
os.system("pip install torch==1.13.0")
from Face2Vid.utils.helper import load_model
from Face2Vid.commons.config.inference_config import InferenceConfig
from Face2Vid.utils.rprint import rlog as log
import torch
import yaml


# Load the inference configuration
cfg = InferenceConfig()

# Load the model configuration
with open(cfg.models_config, 'r') as config_file:
    model_config = yaml.safe_load(config_file)

# Load the motion extractor model
warping_module = load_model(cfg.checkpoint_W, model_config, cfg.device, 'warping_module')
log('Load warping_module done.')

# Set the model to evaluation mode
warping_module.eval()

# Create dummy input data
feature_3d = torch.randn(1, 32, 16, 64, 64)  # Example shape, adjust as necessary
kp_source = torch.randn(1, 21, 3)  # Example shape, adjust as necessary
kp_driving = torch.randn(1, 21, 3)  # Example shape, adjust as necessary

# Export the model
torch.onnx.export(
    warping_module,
    (feature_3d, kp_source, kp_driving),
    "warping.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['feature_3d', 'kp_source', 'kp_driving'],
    output_names=['output'],
    dynamic_axes={'feature_3d': {0: 'batch_size'}, 'kp_source': {0: 'batch_size'}, 'kp_driving': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
os.system("pip install -U torch")

# import onnxruntime as ort
# import torch
#
# ort_session = ort.InferenceSession('live_portraiet_onnx/warping_module.onnx')
#
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
#
# feature_3d = torch.randn(1, 32, 16, 64, 64)  # Example shape, adjust as necessary
# kp_source = torch.randn(1, 21, 3)  # Example shape, adjust as necessary
# kp_driving = torch.randn(1, 21, 3)  # Example shape, adjust as necessary
# # Prepare inputs for ONNX runtime
# ort_inputs = {
#     'feature_3d': to_numpy(feature_3d),
#     'kp_source': to_numpy(kp_source),
#     'kp_driving': to_numpy(kp_driving)
# }
#
# # Run inference
# ort_outs = ort_session.run(None, ort_inputs)
#
# # Process the output as necessary
# output = ort_outs[0]
# print("Inference output:", output)
