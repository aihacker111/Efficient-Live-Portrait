import onnxruntime as ort
import numpy as np


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

    def inference(self, task, inputs, single_input=True):
        session = self._session[task]
        if ort.get_device() == 'CPU':
            if single_input:
                name = session.get_inputs()[0].name
                inputs = {name: np.array(inputs)}
            else:
                inputs = {input_name: np.array(input_tensor) for input_name, input_tensor in zip([input.name for input in session.get_inputs()], inputs)}
            outputs = session.run(None, inputs)
        else:
            io_binding = session.io_binding()

            if single_input:
                inputs = [inputs]  # Ensure inputs is a list for consistent processing

            input_names = [input.name for input in session.get_inputs()]

            # Bind input tensors
            for input_name, input_tensor in zip(input_names, inputs):
                input_tensor = input_tensor.contiguous()
                io_binding.bind_input(
                    name=input_name,  # Input name as specified in the ONNX model
                    device_type='cuda',
                    device_id=0,
                    element_type=np.float32,
                    shape=tuple(input_tensor.shape),
                    buffer_ptr=input_tensor.data_ptr(),  # Buffer pointer to the tensor data
                )

            output_names = [output.name for output in session.get_outputs()]

            # Bind output tensors
            for output_name in output_names:
                io_binding.bind_output(output_name)

            session.run_with_iobinding(io_binding)
            outputs = io_binding.copy_outputs_to_cpu()

        return outputs

