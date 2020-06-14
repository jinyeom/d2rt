from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import tensorrt as trt
from pycuda import autoinit
from pycuda import driver as cuda


@dataclass
class MemoryBuffer:
    mem_dtype: np.dtype
    host: np.ndarray
    device: cuda.DeviceAllocation


class TensorrtModel:
    def __init__(
        self,
        onnx_path: Union[str, Path],
        engine_path: Union[str, Path],
        max_batch_size: int = 1,
        workspace_size: int = 4,
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
        print("initializing TensorRT engine...")
        self.logger = trt.Logger(trt.Logger.Severity.INFO)
        trt.init_libnvinfer_plugins(self.logger, "")

        if not self.force_rebuild and self.engine_path.exists():
            print("loading %s".format(self.engine_path))
            with trt.Runtime(self.logger) as runtime:
                with open(path, "rb") as f:
                    engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
            return

        with trt.Builder(self.logger) as builder:
            builder.max_workspace_size = self.workspace_size << 30
            builder.max_batch_size = self.batch_size
            builder.fp16_mode = self.fp16_mode
            builder.strict_type_constraints = True

            with builder.create_network() as network:
                with trt.OnnxParser(network, trt_logger) as parser:
                    with open(self.onnx_path, "rb") as model:
                        success = parser.parse(model.read())
                        if not success:
                            err = parser.get_error(0)
                            msg = "While parsing node number %i:\n" % err.node()
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
