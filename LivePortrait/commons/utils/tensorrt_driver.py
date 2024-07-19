import tensorrt as trt
import pycuda.driver as cuda
import pycuda.gpuarray
import pycuda.autoinit
import numpy as np
import ctypes

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class Binding:
    def __init__(self, engine, idx_or_name):
        self.name = idx_or_name if isinstance(idx_or_name, str) else engine.get_tensor_name(idx_or_name)
        if not self.name:
            raise IndexError(f"Binding index out of range: {idx_or_name}")
        self.is_input = engine.get_tensor_mode(self.name) == trt.TensorIOMode.INPUT
        dtype = engine.get_tensor_dtype(self.name)
        dtype_map = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8,
            trt.DataType.BOOL: np.bool_,
        }
        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = np.int32
        if hasattr(trt.DataType, 'INT64'):
            dtype_map[trt.DataType.INT64] = np.int64
        self.dtype = dtype_map[dtype]
        self.shape = tuple(engine.get_tensor_shape(self.name))
        self._host_buf = None
        self._device_buf = None

    @property
    def host_buffer(self):
        if self._host_buf is None:
            self._host_buf = cuda.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf

    @property
    def device_buffer(self):
        if self._device_buf is None:
            self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
        return self._device_buf

    def get_async(self, stream):
        self.device_buffer.get_async(stream, self.host_buffer)
        return self.host_buffer


class TensorRTEngine:
    def __init__(self, **kwargs):
        self.cfg = kwargs
        self.model_paths = {
            'feature_extractor': self.cfg['rt_F'],
            'motion_extractor': self.cfg['rt_M'],
            'generator': self.cfg['rt_GW'],
            'stitching_retargeting': self.cfg['rt_S'],
            'stitching_retargeting_eye': self.cfg['rt_SE'],
            'stitching_retargeting_lip': self.cfg['rt_SL']
        }
        self.plugin_path = "/content/drive/MyDrive/libgrid_sample_3d_plugin.so"
        self.load_plugins(TRT_LOGGER)
        self.engines = {}
        self.contexts = {}
        self.bindings = {}
        self.binding_addresses = {}
        self.inputs = {}
        self.outputs = {}
        self.stream = cuda.Stream()
        self.initialize_engines()

    def load_plugins(self, logger: trt.Logger):
        ctypes.CDLL(self.cfg['grid_sample_3d'], mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(logger, "")

    def initialize_engines(self):
        """
        'feature_extractor': [(1, 3, 256, 256)],
        'motion_extractor': [(1, 3, 256, 256)],
        'generator': [(1, 32, 16, 64, 64), (1, 21, 3), (1, 21, 3)],
        'stitching_retargeting': [(1, 126)],
        'stitching_retargeting_eye': [(1, 66)],
        'stitching_retargeting_lip': [(1, 65)]
        """
        for model_name, model_path in self.model_paths.items():
            engine = self.load_engine(model_path)
            context = engine.create_execution_context()
            bindings = [Binding(engine, i) for i in range(engine.num_io_tensors)]
            self.engines[model_name] = engine
            self.contexts[model_name] = context
            self.bindings[model_name] = bindings
            self.binding_addresses[model_name] = [b.device_buffer.ptr for b in bindings]
            self.inputs[model_name] = [b for b in bindings if b.is_input]
            self.outputs[model_name] = [b for b in bindings if not b.is_input]
            self.prepare_buffers(model_name)

    @staticmethod
    def load_engine(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def prepare_buffers(self, model_name):
        for binding in self.inputs[model_name] + self.outputs[model_name]:
            _ = binding.device_buffer  # Force buffer allocation

    @staticmethod
    def check_input_validity(input_idx, input_array, input_binding):
        if not input_array.flags['C_CONTIGUOUS']:
            input_array = np.ascontiguousarray(input_array)
        if input_array.shape != input_binding.shape:
            if not (input_binding.shape == (1,) and input_array.shape == ()):
                raise ValueError(
                    f"Wrong shape for input {input_idx}. Expected {input_binding.shape}, got {input_array.shape}.")
        if input_array.dtype != input_binding.dtype:
            if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
                input_array = input_array.astype(np.int32)
                if not np.array_equal(input_array, input_array.astype(np.int64)):
                    raise TypeError(
                        f"Wrong dtype for input {input_idx}. Expected {input_binding.dtype}, got {input_array.dtype}. Cannot safely cast.")
            else:
                raise TypeError(
                    f"Wrong dtype for input {input_idx}. Expected {input_binding.dtype}, got {input_array.dtype}.")
        return input_array

    def run_sequential_tasks(self, model_name, inputs):
        if model_name not in self.engines:
            raise ValueError(f"Model name {model_name} not found in engines.")
        engine = self.engines[model_name]
        context = self.contexts[model_name]
        binding_addresses = self.binding_addresses[model_name]
        inputs_bindings = self.inputs[model_name]
        outputs_bindings = self.outputs[model_name]
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in inputs_bindings]
        if len(inputs) != len(inputs_bindings):
            raise ValueError(f"Number of input arrays does not match number of input bindings for model {model_name}.")
        for i, (input_array, input_binding) in enumerate(zip(inputs, inputs_bindings)):
            input_array = self.check_input_validity(i, input_array, input_binding)
            input_binding.device_buffer.set_async(input_array, self.stream)
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if i < len(inputs) and engine.is_shape_inference_io(tensor_name):
                context.set_tensor_address(tensor_name, inputs[i].ctypes.data)
            else:
                context.set_tensor_address(tensor_name, binding_addresses[i])
        try:
            context.execute_async_v3(self.stream.handle)
            self.stream.synchronize()
        except Exception as e:
            print(f"Error during inference for model {model_name}: {e}")
            return None, 0
        try:
            outputs = [output.get_async(self.stream) for output in outputs_bindings]
        except Exception as e:
            print(f"Error retrieving output for model {model_name}: {e}")
            return None, 0
        return outputs

    def inference_tensorrt(self, task, inputs):
        # Define input shapes
        input_map_keys = {
            task: inputs
        }

        # Execute tasks sequentially
        # shape = input_map_keys[task]
        # Ensure inputs is a list of arrays
        if isinstance(inputs, list):
            shape = [input.shape for input in inputs]
        else:
            shape = [input_map_keys[task][i].shape for i in range(len(input_map_keys[task]))]
        inputs = [np.random.randn(*s).astype(self.inputs[task][i].dtype) for i, s in enumerate(shape)]
        result = self.run_sequential_tasks(task, inputs)
        return result


if __name__ == "__main__":
    pass
    # Initialize TensorRT Engine
    # trt_engine = TensorRTEngine()

    # Run the sequential tasks
    # results = trt_engine.run_sequential_tasks(task='motion_extractor', inputs=[(1, 3, 256, 256)])

    # print(results)

