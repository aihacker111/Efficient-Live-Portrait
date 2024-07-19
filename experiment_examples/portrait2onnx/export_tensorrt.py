#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import pdb
import sys
import logging
import argparse
import ctypes
import numpy as np
import torch
from pathlib import Path
import tensorrt as trt
import logging
import platform

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


def load_plugins(logger: trt.Logger):
    # 加载插件库
    ctypes.CDLL("/content/libgrid_sample_3d_plugin.so", mode=ctypes.RTLD_GLOBAL)
    # 初始化TensorRT的插件库
    trt.init_libnvinfer_plugins(logger, "")


LOGGER = logging.getLogger(__name__)


def colorstr(*input):
    return "".join(input)


def check_requirements(pkg, cmds):
    import os
    os.system(f"pip install {pkg} {cmds}")


def check_version(version, minimum, hard=True):
    assert version >= minimum, f'TensorRT version {version} is less than minimum required {minimum}'


def export_engine(ims, onnx_file, file, half, dynamic, workspace=4, verbose=False, int8=False,
                  prefix=colorstr("TensorRT:")):
    """
    Exports a YOLOv5 model to TensorRT engine format, requiring GPU and TensorRT>=8.0.0.

    Args:
        model (torch.nn.Module): YOLOv5 model to be exported (not used in this function).
        ims (list of torch.Tensor): List of input tensors of shape (B, C, H, W).
        onnx_file (Path): Path to the pre-existing ONNX model.
        file (Path): Path to save the exported model.
        half (bool): Set to True to export with FP16 precision.
        dynamic (bool): Set to True to enable dynamic input shapes.
        workspace (int): Workspace size in GB (default is 4).
        verbose (bool): Set to True for verbose logging output.
        int8 (bool): Set to True to enable INT8 precision.
        prefix (str): Log message prefix.

    Returns:
        (Path, None): Tuple containing the path to the exported model and None.

    Raises:
        AssertionError: If executed on CPU instead of GPU.
        RuntimeError: If there is a failure in parsing the ONNX file.
    """
    assert ims[0].device.type != "cpu", "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"

    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == "Linux":
            check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
        import tensorrt as trt

    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
    assert onnx_file.exists(), f"failed to export ONNX file: {onnx_file}"
    f = file.with_suffix(".engine")  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
#     load_plugins(logger)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
     # Correct conversion: workspace size in GB to bytes
    workspace_size_bytes = workspace * (2 ** 30)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_bytes)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        for error in range(parser.num_errors):
            LOGGER.error(parser.get_error(error))
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        for im in ims:
            if im.shape[0] <= 1:
                LOGGER.warning(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
        profile = builder.create_optimization_profile()
        for inp, im in zip(inputs, ims):
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    LOGGER.info(
        f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} {'with INT8' if int8 else ''} engine as {f}")
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = trt.IInt8MinMaxCalibrator()

    build = builder.build_serialized_network
    serialized_engine = build(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the TensorRT engine!")

    with open(f, "wb") as t:
        t.write(serialized_engine)

    return f, None


# Usage Example

import torch
from pathlib import Path

# Example input tensors (should match the input shapes expected by your ONNX model)
# input_tensor1 = torch.randn(1, 32, 16, 64, 64).cuda()  # Adjust the shape as necessary for your model
# input_tensor2 = torch.randn(1, 21, 3).cuda()  # Adjust the shape as necessary for your model
# input_tensor3 = torch.randn(1, 21, 3).cuda()  # Adjust the shape as necessary for your model
input_tensor = torch.randn(1, 3, 256, 256).cuda()
# Path to your existing ONNX model
onnx_model_path = Path("/kaggle/working/motion_extractor.onnx")

# Path where you want to save the TensorRT engine
engine_output_path = Path("motion_extractor_fp16.engine")

# Set half precision to True if you want FP16 precision, otherwise set it to False
use_half_precision = True

# Set dynamic to True if you want dynamic input shapes, otherwise set it to False
use_dynamic_shapes = True

# Set int8 to True if you want INT8 precision, otherwise set it to False
use_int8_precision = False

# Call the function to export the engine
export_engine(
    ims=[input_tensor],
    onnx_file=onnx_model_path,
    file=engine_output_path,
    half=use_half_precision,
    dynamic=use_dynamic_shapes,
    workspace=12,  # Workspace size in GB
    verbose=True,  # Set to False to reduce logging output
    int8=use_int8_precision
)

print(f"TensorRT engine has been successfully exported to {engine_output_path}")

