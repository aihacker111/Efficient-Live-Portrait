import onnxruntime as ort
import numpy as np
import torch


class ONNXPredictor:
    def __init__(self, cfg):
        self._session = self.initialize_sessions(cfg)

    @staticmethod
    def get_providers():
        # Check for CUDA execution provider
        if ort.get_device() == 'GPU':
            return ['CUDAExecutionProvider']

        # If the device is CPU, add CoreML and CPU execution providers
        if ort.get_device() == 'CPU':
            available_providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            return available_providers

    def initialize_sessions(self, cfg):
        providers = self.get_providers()

        model_dict = {
            'wg_session': ort.InferenceSession(cfg.checkpoint_GW, providers=providers),
            'm_session': ort.InferenceSession(cfg.checkpoint_M, providers=providers),
            'f_session': ort.InferenceSession(cfg.checkpoint_F, providers=providers),
            's_session': ort.InferenceSession(cfg.checkpoint_S, providers=providers),
            's_l_session': ort.InferenceSession(cfg.checkpoint_SL, providers=providers),
            's_e_session': ort.InferenceSession(cfg.checkpoint_SE, providers=providers)
        }

        return model_dict

    @staticmethod
    def inference_single_input(session, input_tensor):
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

        # Bind output tensors
        for output_name in [output.name for output in session.get_outputs()]:
            io_binding.bind_output(output_name)

        session.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()

        return outputs

    @staticmethod
    def inference_multiple_inputs(session, inputs):
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

        # Bind output tensors
        for output_name in [output.name for output in session.get_outputs()]:
            io_binding.bind_output(output_name)

        session.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()

        return outputs

    def inference(self, task, inputs, single_input=True):
        session = self._session[task]
        if ort.get_device() == 'CPU':
            if single_input:
                name = session.get_inputs()[0].name
                inputs = {name: np.array(inputs)}
            else:
                inputs = {input_name: np.array(input_tensor) for input_name, input_tensor in
                          zip([input.name for input in session.get_inputs()], inputs)}
            outputs = session.run(None, inputs)
        else:
            if single_input:
                outputs = self.inference_single_input(session, inputs)
            else:
                outputs = self.inference_multiple_inputs(session, inputs)

        return outputs
