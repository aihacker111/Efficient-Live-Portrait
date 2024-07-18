# efficient_live_portrait_predictor.py

from .utils import ONNXEngine, TensorRTEngine
import onnxruntime as ort
import numpy as np


class EfficientLivePortraitPredictor(ONNXEngine, TensorRTEngine):
    def __init__(self, cfg, use_tensorrt=False):
        TensorRTEngine.__init__(self, cfg)
        ONNXEngine.__init__(self)
        # TensorRTEngine.__init__(self, cfg)
        self.cfg = cfg
        self.use_tensorrt = use_tensorrt
        self._session = self.initialize_sessions(cfg) if not use_tensorrt else None

    def run_time(self, engine_name, task,  inputs, single_input=True):
        if self.use_tensorrt:
            return self.inference_tensorrt(inputs, engine_name)
        else:
            return self.inference_onnx(task, inputs, single_input)

    def inference_onnx(self, task, inputs, single_input=True):
        session = self._session[task]
        if ort.get_device() == 'CPU':
            if single_input:
                name = session.get_inputs()[0].name
                inputs = {name: np.array(inputs)}
            else:
                inputs = {input.name: np.array(input_tensor) for input, input_tensor in
                          zip(session.get_inputs(), inputs)}
            outputs = session.run(None, inputs)
        else:
            if single_input:
                outputs = self.inference_single_input(session, inputs)
            else:
                outputs = self.inference_multiple_inputs(session, inputs)
        return outputs
