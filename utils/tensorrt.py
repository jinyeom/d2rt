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

        self.logger: trt.tensorrt.Logger = None
        self.engine: trt.tensorrt.ICudaEngine = None
        self.context: trt.tensorrt.IExecutionContext = None
        self.stream: cuda.Stream = None
        self.input_buffers: List[MemoryBuffer] = None
        self.output_buffers: List[MemoryBuffer] = None
        self.bindings: List[int] = None

        self._init_logger()
        self._init_engine()
        self._init_buffers()
        self._check_init()

    def _init_logger(self):
        self.logger = trt.Logger(trt.Logger.Severity.INFO)
        trt.init_libnvinfer_plugins(self.logger, "")

    def _init_engine(self):
        if not self.force_rebuild and self.engine_path.exists():
            with trt.Runtime(self.logger) as runtime:
                with open(self.engine_path, "rb") as f:
                    engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
                self.context = self.engine.create_execution_context()
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
                host = None  # will be added during inference
                buffer = MemoryBuffer(dtype, shape, host, device)
                self.input_buffers.append(buffer)
            else:
                host = cuda.pagelocked_empty(size, dtype)
                buffer = MemoryBuffer(dtype, shape, host, device)
                self.output_buffers.append(buffer)

    def _check_init(self):
        assert self.logger is not None
        assert self.engine is not None
        assert self.context is not None
        assert self.stream is not None
        assert self.input_buffers is not None
        assert self.output_buffers is not None
        assert self.bindings is not None

    def _validate_inputs(self, inputs: List[np.ndarray]):
        for x, ib in zip(inputs, self.input_buffers):
            assert x.shape == ib.shape

    def _execute_inference(self):
        # copy inputs: host -> device
        for ib in self.input_buffers:
            cuda.memcpy_htod_async(ib.device, ib.host, self.stream)

        # execute inference
        self.context.execute_async(
            batch_size=self.max_batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )

        # copy outputs: device -> host
        for ob in self.output_buffers:
            cuda.memcpy_dtoh_async(ob.host, ob.device, self.stream)

        # synchronize
        self.stream.synchronize()

    def __call__(self, inputs: List[np.ndarray]):
        self._validate_inputs(inputs)

        batch_outputs = []
        total_batch_size = inputs[0].shape[0]
        for i in range(0, total_batch_size, self.max_batch_size):
            # copy `self.max_batch_size` batch of images into each input buffer
            for x, ib in zip(inputs, self.input_buffers):
                batch = x[i : i + self.max_batch_size]
                ib.host = np.ascontiguousarray(batch, dtype=ib.dtype)

            # execute inference with the current batch
            self._execute_inference()

            # retrieve outputs in their desired shapes
            outputs = [
                np.reshape(ob.host, (self.max_batch_size, *ob.shape))
                for ob in self.output_buffers
            ]
            batch_outputs.append(outputs)

        batch_outputs = [
            np.concatenate(output, axis=0)[:total_batch_size]
            for output in zip(*batch_outputs)
        ]
        return batch_outputs
