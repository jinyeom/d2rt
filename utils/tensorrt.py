from contextlib import ExitStack
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

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


class InferenceEngine:
    def __init__(
        self,
        onnx_path: Union[str, Path],
        engine_path: Union[str, Path],
        max_batch_size: int = 1,
        workspace_size: int = 1,
        fp16_mode: bool = True,
        force_rebuild: bool = False,
    ):
        self.onnx_path: Union[str, Path] = Path(onnx_path)
        self.engine_path: Union[str, Path] = Path(engine_path)
        self.max_batch_size: int = max_batch_size
        self.workspace_size: int = workspace_size
        self.fp16_mode: bool = fp16_mode
        self.force_rebuild: bool = force_rebuild

        self._logger: trt.tensorrt.Logger = None
        self._engine: trt.tensorrt.ICudaEngine = None
        self._context: trt.tensorrt.IExecutionContext = None
        self._stream: cuda.Stream = None
        self._input_buffers: List[MemoryBuffer] = None
        self._output_buffers: List[MemoryBuffer] = None
        self._bindings: List[int] = None

        self._init_logger()
        self._init_engine()
        self._init_buffers()
        self._check_init()

    def _init_logger(self):
        self._logger = trt.Logger(trt.Logger.Severity.INFO)
        trt.init_libnvinfer_plugins(self._logger, "")

    def _init_engine(self):
        if not self.force_rebuild and self.engine_path.exists():
            with trt.Runtime(self._logger) as runtime:
                with open(self.engine_path, "rb") as f:
                    engine_data = f.read()
                self._engine = runtime.deserialize_cuda_engine(engine_data)
                self._context = self._engine.create_execution_context()
            return

        with ExitStack() as stack:
            explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            builder = stack.enter_context(trt.Builder(self._logger))
            network = stack.enter_context(builder.create_network(explicit_batch))
            parser = stack.enter_context(trt.OnnxParser(network, self._logger))

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

            self._engine = builder.build_cuda_engine(network)
            self._context = self._engine.create_execution_context()
            with open(self.engine_path, "wb") as f:
                f.write(self._engine.serialize())

    def _init_buffers(self):
        self._stream = cuda.Stream()
        self._input_buffers = []
        self._output_buffers = []
        self._bindings = []

        for binding in self._engine:
            dtype = np.dtype(trt.nptype(self._engine.get_binding_dtype(binding)))
            shape = self._engine.get_binding_shape(binding)
            size = trt.volume(shape) * self._engine.max_batch_size
            device = cuda.mem_alloc(size * dtype.itemsize)
            self._bindings.append(int(device))

            if self._engine.binding_is_input(binding):
                host = None  # will be added during inference
                buffer = MemoryBuffer(dtype, shape, host, device)
                self._input_buffers.append(buffer)
            else:
                host = cuda.pagelocked_empty(size, dtype)
                buffer = MemoryBuffer(dtype, shape, host, device)
                self._output_buffers.append(buffer)

    def _check_init(self):
        assert self._logger is not None
        assert self._engine is not None
        assert self._context is not None
        assert self._stream is not None
        assert self._input_buffers is not None
        assert self._output_buffers is not None
        assert self._bindings is not None

    def _validate_inputs(self, inputs: List[np.ndarray]):
        total_batch_size = inputs[0].shape[0]
        for x, ib in zip(inputs, self._input_buffers):
            assert x.shape[0] == total_batch_size, "inconsistent batch size"
            assert x.shape[1:] == ib.shape[1:], "invalid image shape"

    def _execute_inference(self):
        for ib in self._input_buffers:
            cuda.memcpy_htod_async(ib.device, ib.host, self._stream)

        self._context.execute_async(
            batch_size=self._engine.max_batch_size,
            bindings=self._bindings,
            stream_handle=self._stream.handle,
        )

        for ob in self._output_buffers:
            cuda.memcpy_dtoh_async(ob.host, ob.device, self._stream)

        self._stream.synchronize()

    def predict(self, inputs: List[np.ndarray]):
        self._validate_inputs(inputs)

        batch_outputs = []
        total_batch_size = inputs[0].shape[0]
        for i in range(0, total_batch_size, self._engine.max_batch_size):
            for x, ib in zip(inputs, self._input_buffers):
                batch = x[i : i + self._engine.max_batch_size]
                ib.host = np.ascontiguousarray(batch, dtype=ib.dtype)

            self._execute_inference()

            outputs = [
                np.reshape(ob.host, (self._engine.max_batch_size, *ob.shape))
                for ob in self._output_buffers
            ]
            batch_outputs.append(outputs)

        return [
            np.concatenate(output, axis=0)[:total_batch_size]
            for output in zip(*batch_outputs)
        ]

    def __call__(self, inputs: List[np.ndarray]):
        return self.predict(inputs)
