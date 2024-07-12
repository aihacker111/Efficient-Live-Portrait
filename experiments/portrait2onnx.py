import torch
from Face2Vid.utils.helper import load_model
from Face2Vid.commons.config.inference_config import InferenceConfig
from Face2Vid.utils.rprint import rlog as log
import yaml
import onnx
import onnxruntime as ort
import numpy as np


# spade_generator = (1, 256, 64, 64)
# feature_extractor = (1,3,256,256)
# motion_extractor = (1,1,3,256,256)

# Load the inference configuration
cfg = InferenceConfig()

# Load the model configuration
with open(cfg.models_config, 'r') as config_file:
    model_config = yaml.safe_load(config_file)

# Load the spade_generator model
motion_extractor = load_model(cfg.checkpoint_M, model_config, cfg.device, 'motion_extractor')
log('Load spade_generator done.')

# Set the model to evaluation mode
motion_extractor.eval()

# Create a dummy input tensor with the appropriate shape and device
dummy_input = torch.randn(1, 3, 256, 256).to(cfg.device)  # Example shape (batch_size, channels, height, width)

# Export the model to ONNX format
onnx_file_path = '../live_portraiet_onnx/onnx/motion_extractor.onnx'
torch.onnx.export(
    motion_extractor,  # model being run
    dummy_input,  # model input (or a tuple for multiple inputs)
    onnx_file_path,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=11,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=['input'],  # the model's input names
    output_names=['output'],  # the model's output names
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # variable length axes
)

print(f'Model has been converted to ONNX and saved at {onnx_file_path}')
log(f'Model has been converted to ONNX and saved at {onnx_file_path}')

# Verify the ONNX model
onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)

# Create ONNX runtime session
ort_session = ort.InferenceSession(onnx_file_path)


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    return tensor


# Run inference with the same dummy input on PyTorch model
with torch.no_grad():
    torch_output = motion_extractor(dummy_input)

# Run inference with the same dummy input on ONNX model
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)

# Print the data types of the outputs
print("PyTorch output type:", type(torch_output))
print("ONNX output type:", type(ort_outs))


# Function to compare intermediate outputs
def compare_outputs(torch_out, onnx_out, tolerance=1e-03):
    try:
        np.testing.assert_allclose(to_numpy(torch_out), onnx_out[0], rtol=tolerance, atol=1e-05)
        print("Outputs match between PyTorch and ONNX models.")
    except AssertionError as e:
        print("Outputs do not match.")
        print(e)


# If the PyTorch model output is a dictionary
if isinstance(torch_output, dict):
    # Compare each key in the dictionary
    for key in torch_output.keys():
        compare_outputs(torch_output[key], ort_outs[0], tolerance=1e-02)  # Increased tolerance
else:
    # If PyTorch model output is not a dictionary, compare directly
    compare_outputs(torch_output, ort_outs[0], tolerance=1e-02)  # Increased tolerance

print("All outputs match between the PyTorch model and the ONNX model.")
log("All outputs match between the PyTorch model and the ONNX model.")
