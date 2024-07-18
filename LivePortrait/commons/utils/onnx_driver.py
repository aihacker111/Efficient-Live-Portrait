import onnxruntime as ort
import torch
import numpy as np
from typing import Dict


class ONNXEngine:
    def __init__(self):
        pass

    @staticmethod
    def get_providers() -> list:
        """Returns the list of providers based on the current device."""
        if ort.get_device() == 'GPU':
            return ['CUDAExecutionProvider']
        elif ort.get_device() == 'CPU':
            return ['CPUExecutionProvider', 'CoreMLExecutionProvider']
        else:
            return []

    def initialize_sessions(self, cfg) -> Dict[str, ort.InferenceSession]:
        """
        Initialize ONNX InferenceSession instances for each model checkpoint.

        Args:
        - cfg (Config): Configuration object containing checkpoint paths.

        Returns:
        - Dict[str, ort.InferenceSession]: Dictionary mapping session names to InferenceSession objects.
        """
        providers = self.get_providers()

        model_sessions = {}
        model_names = ['wg', 'm', 'f', 's', 's_l', 's_e']  # Example model names

        for name in model_names:
            model_path = getattr(cfg, f"checkpoint_{name.upper()}")
            model_sessions[f"{name}_session"] = ort.InferenceSession(model_path, providers=providers)

        return model_sessions

    @staticmethod
    def inference_single_input(session, input_tensor):
        """Perform inference with a single input tensor."""
        io_binding = session.io_binding()

        input_tensor = torch.tensor(input_tensor, device='cuda').contiguous()
        input_name = session.get_inputs()[0].name
        io_binding.bind_input(
            name=input_name,
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(input_tensor.shape),
            buffer_ptr=input_tensor.data_ptr(),
        )

        for output_name in [output.name for output in session.get_outputs()]:
            io_binding.bind_output(output_name)

        session.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()

        return outputs

    @staticmethod
    def inference_multiple_inputs(session, inputs):
        """Perform inference with multiple input tensors."""
        io_binding = session.io_binding()

        for idx, input_tensor in enumerate(inputs):
            input_name = session.get_inputs()[idx].name
            input_tensor = torch.tensor(input_tensor, device='cuda').contiguous()
            io_binding.bind_input(
                name=input_name,
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(input_tensor.shape),
                buffer_ptr=input_tensor.data_ptr(),
            )

        for output_name in [output.name for output in session.get_outputs()]:
            io_binding.bind_output(output_name)

        session.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()

        return outputs
