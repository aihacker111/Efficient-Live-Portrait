import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader
import onnxsim
import numpy as np
import os

# Paths to the ONNX model files
model_fp32 = '/Users/macbook/Downloads/Efficient-Face2Vid-Portrait/experiment_examples/portrait2onnx/warping.onnx'
model_int8 = 'warping.quant.onnx'

# Load the ONNX model
onnx_model = onnx.load(model_fp32)

# Simplify the ONNX model to remove unnecessary nodes
# model_simp, check = onnxsim.simplify(onnx_model)
# assert check, "Simplified ONNX model could not be validated"

# Save the simplified model
# onnx.save(model_simp, model_fp32)


# Define a calibration data reader
class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self):
        self.data = iter([{"input": np.random.rand(1, 256, 64, 64).astype(np.float32)}])  # Example input shape and type

    def get_next(self):
        return next(self.data, None)

    def rewind(self):
        self.data = iter([{"input": np.random.rand(1, 256, 64, 64).astype(np.float32)}])

# class MyCalibrationDataReader(CalibrationDataReader):
#     def __init__(self):
#         # Example input shapes and types for a model with 3 inputs
#         self.data = iter([{
#             "feature_3d": np.random.rand(1, 32, 16, 64, 64).astype(np.float32),
#             "kp_source": np.random.rand(1, 21, 3).astype(np.float32),
#             "kp_driving": np.random.rand(1, 21, 3).astype(np.float32)
#         }])
#
#     def get_next(self):
#         return next(self.data, None)
#
#     def rewind(self):
#         self.data = iter([{
#             "feature_3d": np.random.rand(1, 32, 16, 64, 64).astype(np.float32),
#             "kp_source": np.random.rand(1, 21, 3).astype(np.float32),
#             "kp_driving": np.random.rand(1, 21, 3).astype(np.float32)
#         }])


# calibration_data_reader = MyCalibrationDataReader()

# Quantize the simplified ONNX model
# quantize_static(model_fp32, model_int8, calibration_data_reader)
quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QInt8)
print(f"Quantized model saved to {model_int8}")
