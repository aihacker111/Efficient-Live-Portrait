# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.gpuarray
# import pycuda.autoinit
# import numpy as np
#
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#
#
# class Binding:
#     def __init__(self, engine, idx_or_name):
#         self.name = idx_or_name if isinstance(idx_or_name, str) else engine.get_tensor_name(idx_or_name)
#         if not self.name:
#             raise IndexError(f"Binding index out of range: {idx_or_name}")
#         self.is_input = engine.get_tensor_mode(self.name) == trt.TensorIOMode.INPUT
#         dtype = engine.get_tensor_dtype(self.name)
#         dtype_map = {
#             trt.DataType.FLOAT: np.float32,
#             trt.DataType.HALF: np.float16,
#             trt.DataType.INT8: np.int8,
#             trt.DataType.BOOL: np.bool_,
#         }
#         if hasattr(trt.DataType, 'INT32'):
#             dtype_map[trt.DataType.INT32] = np.int32
#         if hasattr(trt.DataType, 'INT64'):
#             dtype_map[trt.DataType.INT64] = np.int64
#         self.dtype = dtype_map[dtype]
#         self.shape = tuple(engine.get_tensor_shape(self.name))
#         self._host_buf = None
#         self._device_buf = None
#
#     @property
#     def host_buffer(self):
#         if self._host_buf is None:
#             self._host_buf = cuda.pagelocked_empty(self.shape, self.dtype)
#         return self._host_buf
#
#     @property
#     def device_buffer(self):
#         if self._device_buf is None:
#             self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
#         return self._device_buf
#
#     def get_async(self, stream):
#         self.device_buffer.get_async(stream, self.host_buffer)
#         return self.host_buffer
#
#
# class TensorRTEngine:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.engined = self.load_engine(self.cfg)
#         self.context = self.engined.create_execution_context()
#         self.stream = cuda.Stream()
#         self.bindings = [Binding(self.engined, i) for i in range(self.engined.num_io_tensors)]
#         self.binding_address = [b.device_buffer.ptr for b in self.bindings]
#         self.inputs = [b for b in self.bindings if b.is_input]
#         self.outputs = [b for b in self.bindings if not b.is_input]
#         for binding in self.inputs + self.outputs:
#             _ = binding.device_buffer  # Force buffer allocation
#
#     @staticmethod
#     def load_engine(engine_file_path):
#         with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#             return runtime.deserialize_cuda_engine(f.read())
#
#     @staticmethod
#     def check_input_validity(input_idx, input_array, input_binding):
#         # Check shape
#         if input_array.shape != input_binding.shape:
#             if not (input_binding.shape == (1,) and input_array.shape == ()):
#                 raise ValueError(
#                     f"Wrong shape for input {input_idx}. Expected {input_binding.shape}, got {input_array.shape}.")
#
#         # Check dtype and handle int64 to int32 conversion
#         if input_array.dtype != input_binding.dtype:
#             if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
#                 input_array = input_array.astype(np.int32)
#                 if not np.array_equal(input_array, input_array.astype(np.int64)):
#                     raise TypeError(
#                         f"Wrong dtype for input {input_idx}. Expected {input_binding.dtype}, got {input_array.dtype}. Cannot safely cast.")
#             else:
#                 raise TypeError(
#                     f"Wrong dtype for input {input_idx}. Expected {input_binding.dtype}, got {input_array.dtype}.")
#
#         return input_array
#
#     def inference_tensorrt(self, inputs):
#         if isinstance(inputs, dict):
#             inputs = [inputs[b.name] for b in self.inputs]
#         for i, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
#             input_binding.device_buffer.set_async(self.check_input_validity(i, input_array, input_binding), self.stream)
#         for i in range(self.engined.num_io_tensors):
#             tensor_name = self.engined.get_tensor_name(i)
#             self.context.set_tensor_address(tensor_name, inputs[i].ctypes.data if i < len(
#                 inputs) and self.engined.is_shape_inference_io(tensor_name) else self.binding_address[i])
#         self.context.execute_async_v3(self.stream.handle)
#         self.stream.synchronize()
#         return [output.get_async(self.stream) for output in self.outputs]



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
    def __init__(self, cfg):
        self.cfg = cfg
        self.load_plugins(TRT_LOGGER)
        self.engines = {}
        self.contexts = {}
        self.bindings = {}
        self.binding_addresses = {}
        self.inputs = {}
        self.outputs = {}
        self.stream = cuda.Stream()
        self.initialize_engine()

    def load_plugins(self, logger: trt.Logger):
        ctypes.CDLL(self.cfg.grid_sample_3d, mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(logger, "")

    def initialize_engine(self):
        # Load engines and set up contexts, bindings, and addresses
        model_paths = {
            'F': self.cfg.rt_F,
            'M': self.cfg.rt_M,
            'GW': self.cfg.rt_GW,
            'S': self.cfg.rt_S,
            'SE': self.cfg.rt_SE,
            'SL': self.cfg.rt_SL
        }
        for name, path in model_paths.items():
            self.engines[name] = self.load_engine(path)
            self.contexts[name] = self.engines[name].create_execution_context()
            bindings = [Binding(self.engines[name], i) for i in range(self.engines[name].num_io_tensors)]
            self.bindings[name] = bindings
            self.binding_addresses[name] = [b.device_buffer.ptr for b in bindings]
            self.inputs[name] = [b for b in bindings if b.is_input]
            self.outputs[name] = [b for b in bindings if not b.is_input]
            self.prepare_buffers(name)

    @staticmethod
    def load_engine(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def prepare_buffers(self, model_name):
        # Allocate memory only once
        for binding in self.inputs[model_name] + self.outputs[model_name]:
            _ = binding.device_buffer  # Force buffer allocation

    @staticmethod
    def check_input_validity(input_idx, input_array, input_binding):
        # Check shape
        if input_array.shape != input_binding.shape:
            if not (input_binding.shape == (1,) and input_array.shape == ()):
                raise ValueError(
                    f"Wrong shape for input {input_idx}. Expected {input_binding.shape}, got {input_array.shape}.")

        # Check dtype and handle int64 to int32 conversion
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

    def inference_tensorrt(self, model_name, inputs):
        # Ensure model_name is valid
        if model_name not in self.engines:
            raise ValueError(f"Model name {model_name} not found in engines.")

        # Select the correct engine, context, and bindings based on the model_name
        engine = self.engines[model_name]
        context = self.contexts[model_name]
        binding_addresses = self.binding_addresses[model_name]
        inputs_bindings = self.inputs[model_name]
        outputs_bindings = self.outputs[model_name]

        # Convert inputs if necessary (e.g., dictionary to list based on bindings)
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in inputs_bindings]

        # Ensure the number of inputs matches the number of bindings
        if len(inputs) != len(inputs_bindings):
            raise ValueError(f"Number of input arrays does not match number of input bindings for model {model_name}.")

        # Transfer input data to device
        for i, (input_array, input_binding) in enumerate(zip(inputs, inputs_bindings)):
            input_array = self.check_input_validity(i, input_array, input_binding)
            input_binding.device_buffer.set_async(input_array, self.stream)

        # Set tensor addresses in the context
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            if i < len(inputs) and engine.is_shape_inference_io(tensor_name):
                context.set_tensor_address(tensor_name, inputs[i].ctypes.data)
            else:
                context.set_tensor_address(tensor_name, binding_addresses[i])

        # Perform inference
        context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()

        # Retrieve and return output data
        return [output.get_async(self.stream) for output in outputs_bindings]
