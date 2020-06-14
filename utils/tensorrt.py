from contextlib import ExitStack
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import tensorrt as trt
from pycuda import autoinit
from pycuda import driver as cuda


class MemoryBuffer:
    def __init__(
        self,
        dtype: np.dtype,
        shape: Tuple[int, ...],
        host: Optional[np.ndarray] = None,
        device: Optional[cuda.DeviceAllocation] = None,
    ):
        self.dtype = dtype
        self.shape = shape
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

        self.logger = None
        self.engine = None
        self.context = None
        self.stream = None
        self.input_buffers = None
        self.output_buffers = None
        self.bindings = None

        self._init_logger()
        self._init_engine()
        self._init_buffers()

    def _init_logger(self):
        self.logger = trt.Logger(trt.Logger.Severity.INFO)
        trt.init_libnvinfer_plugins(self.logger, "")

    def _init_engine(self):
        if not self.force_rebuild and self.engine_path.exists():
            logger.info("Loading %s..." % self.engine_path)
            with trt.Runtime(self.logger) as runtime:
                with open(self.engine_path, "rb") as f:
                    engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
            return

        with ExitStack() as stack:
            explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            builder = stack.enter_context(trt.Builder(self.logger))
            network = stack.enter_context(builder.create_network(explicit_batch))
            parser = stack.enter_context(trt.OnnxParser(network, self.logger))

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
            self.context = self.engine.create_execution_context()
            with open(self.engine_path, "wb") as f:
                f.write(self.engine.serialize())

    def _init_buffers(self):
        self.stream = cuda.Stream()
        self.input_buffers = []
        self.output_buffers = []
        self.bindings = []

        for binding in self.engine:
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(binding)))
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape) * self.engine.max_batch_size
            device = cuda.mem_alloc(size * dtype.itemsize)
            self.bindings.append(int(device))

            if self.engine.binding_is_input(binding):
                buffer = MemoryBuffer(dtype, shape, None, device)
                self.input_buffers.append(buffer)
            else:
                host_mem = cuda.pagelocked_empty(size, dtype)
                buffer = MemoryBuffer(dtype, shape, host_mem, device)
                self.output_buffers.append(buffer)

    def _execute_inference(self):
        for ib in self.input_buffers:
            cuda.memcpy_htod_async(ib.device, ib.host, self.stream)
        self.context.execute_async(
            batch_size=self.max_batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        for ob in output_buffers:
            cuda.memcpy_dtoh_async(ob.host, ob.device, self.stream)
        self.stream.synchronize()

    def __call__(self, inputs: List[np.ndarray]):
        for i in range(0, len(inputs[0]), self.max_batch_size):
            for x, ib in zip(inputs, self.input_buffers):
                batch_input = x[i : i + self.max_batch_size]
                ib.host = np.ascontiguousarray(batch_input, dtype=ib.mem_dtype)

            self._execute_inference()

            batch_outputs = [
                np.reshape(ob.host, (self.max_batch_size, *ob.shape))
                for ob in self.output_buffers
            ]

        raise NotImplementedError
