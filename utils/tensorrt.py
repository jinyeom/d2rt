import contextlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Sequence

import torch
from torch import nn
import tensorrt as trt


def device_trt2torch(device: trt.TensorLocation) -> torch.device:
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        raise TypeError("invalid device: %s" % device)


def dtype_trt2torch(dtype: trt.DataType) -> torch.dtype:
    if dtype == trt.int8:
        return torch.int8
    elif dtype == trt.bool:
        assert trt_version() >= "7.0"
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("invalid dtype: %s" % dtype)


class TensorRTModule(nn.Module):
    def __init__(
        self,
        onnx_path: Union[str, Path],
        engine_path: Union[str, Path],
        max_batch_size: int = 1,
        max_workspace_size: int = 1 << 25,
        fp16_mode: bool = False,
        force_rebuild: bool = False,
    ):
        super().__init__()
        self.onnx_path: Path = Path(onnx_path)
        self.engine_path: Path = Path(engine_path)
        self.max_batch_size: int = max_batch_size
        self.max_workspace_size: int = max_workspace_size
        self.fp16_mode: bool = fp16_mode
        self.force_rebuild: bool = force_rebuild

        self._logger: trt.Logger = None
        self._engine: trt.ICudaEngine = None
        self._context: trt.IExecutionContext = None
        self._input_names: List[str] = None
        self._output_names: List[str] = None

        self._init_logger()
        self._init_engine()
        self._init_names()

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

        with contextlib.ExitStack() as stack:
            explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            builder = stack.enter_context(trt.Builder(self._logger))
            network = stack.enter_context(builder.create_network(explicit_batch))
            parser = stack.enter_context(trt.OnnxParser(network, self._logger))

            builder.max_workspace_size = self.max_workspace_size
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

    def _init_names(self):
        self._input_names = []
        self._output_names = []
        for index in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(index)
            if self._engine.binding_is_input(index):
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def forward(self, inputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor]:
        batch_size = inputs[0].shape[0]
        bindings = [None] * self._engine.num_bindings

        for i, input_name in enumerate(self._input_names):
            index = self._engine.get_binding_index(input_name)
            dtype = dtype_trt2torch(self._engine.get_binding_dtype(index))
            device = device_trt2torch(self._engine.get_location(index))
            x = inputs[i].to(dtype).to(device).contiguous()
            bindings[index] = x.data_ptr()

        outputs = []
        for output_name in self._output_names:
            index = self._engine.get_binding_index(output_name)
            shape = (batch_size, *self._engine.get_binding_shape(index)[1:])
            dtype = dtype_trt2torch(self._engine.get_binding_dtype(index))
            device = device_trt2torch(self._engine.get_location(index))
            output = torch.empty(shape, dtype=dtype, device=device)
            outputs.append(output)
            bindings[index] = output.data_ptr()

        stream = torch.cuda.current_stream()
        self._context.execute_async(
            batch_size=batch_size, bindings=bindings, stream_handle=stream.cuda_stream,
        )
        stream.synchronize()
        return outputs
