from contextlib import ExitStack
from pathlib import Path
from typing import Union, Optional

import numpy as np
import tensorrt as trt
from pycuda import autoinit
from pycuda import driver as cuda


class MemoryBuffer:
    def __init__(
        self,
        mem_dtype: np.dtype,
        host: Optional[np.ndarray] = None,
        device: Optional[cuda.DeviceAllocation] = None,
    ):
        self.mem_dtype = mem_dtype
        self.host = host
        self.device = device


class InferenceModel:
    def __init__(
        self,
        onnx_path: Union[str, Path],
        engine_path: Union[str, Path],
        max_batch_size: int = 1,
        workspace_size: int = 1,
        fp16_mode: bool = True,
        force_rebuild: bool = False,
    ):
        self.onnx_path = Path(onnx_path)
        self.engine_path = Path(engine_path)
        self.max_batch_size = max_batch_size
        self.workspace_size = workspace_size
        self.fp16_mode = fp16_mode
        self.force_rebuild = force_rebuild
        self._init_engine()
        self._init_buffers()

    def _init_engine(self):
        print("Building TensorRT engine from %s..." % self.onnx_path, flush=True)

        self.logger = trt.Logger(trt.Logger.Severity.INFO)
        trt.init_libnvinfer_plugins(self.logger, "")

        if not self.force_rebuild and self.engine_path.exists():
            print("Loading %s" % self.engine_path, flush=True)
            with trt.Runtime(self.logger) as runtime:
                with open(self.engine_path, "rb") as f:
                    engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
            return

        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with ExitStack() as stack:
            builder = stack.enter_context(trt.Builder(self.logger))
            network = stack.enter_context(builder.create_network(explicit_batch))
            parser = stack.enter_context(trt.OnnxParser(network, self.logger))

            # configure builder
            builder.max_workspace_size = self.workspace_size << 30
            builder.max_batch_size = self.max_batch_size
            builder.fp16_mode = self.fp16_mode
            builder.strict_type_constraints = True

            with open(self.onnx_path, "rb") as model:
                success = parser.parse(model.read())
                if not success:
                    err = parser.get_error(0)
                    msg = "while parsing node number %i:\n" % err.node()
                    msg += "%s:%i In function %s:\n[%i] %s" % (
                        err.file(),
                        err.line(),
                        err.func(),
                        err.code(),
                        err.desc(),
                    )
                    raise RuntimeError(msg)

                self.engine = builder.build_cuda_engine(network)
                with open(self.engine_path, "wb") as f:
                    f.write(self.engine.serialize())

    def _init_buffers(self):
        print("Allocating memory buffers...", flush=True)

        self.stream = cuda.Stream()
        self.input_buffers = []
        self.output_buffers = []
        self.bindings = []

        for binding in self.engine:
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(binding)))
            binding_shape = self.engine.get_binding_shape(binding)
            size = trt.volume(binding_shape) * self.engine.max_batch_size
            device_mem = cuda.mem_alloc(size * dtype.itemsize)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.input_buffers.append(MemoryBuffer(dtype, None, device_mem))
            else:
                host_mem = cuda.pagelocked_empty(size, dtype)
                self.output_buffers.append(MemoryBuffer(dtype, host_mem, device_mem))

    def __call__(self):
        raise NotImplementedError
