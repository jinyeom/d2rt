from typing import List, Tuple, Union, Optional, Sequence
from pathlib import Path

import numpy as np
import onnx
from onnxsim import simplify as onnx_simplify

import torch
from torch import Tensor
from torch import nn

from detectron2.config import CfgNode
from detectron2.modeling import build_model
from detectron2.modeling.meta_arch import RetinaNet


def permute_to_N_HWA_K(tensor: Tensor, K: int) -> Tensor:
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor


class _RetinaNet(nn.Module):
    def __init__(self, model: RetinaNet):
        super().__init__()
        self.pixel_mean = model.pixel_mean
        self.pixel_std = model.pixel_std
        self.backbone = model.backbone
        self.head = model.head
        self.anchor_generator = model.anchor_generator
        self.in_features = model.in_features
        self.num_classes = model.num_classes
        self.device = model.device

    def forward(self, x: Tensor) -> List[Tensor]:
        x = (x - self.pixel_mean) / self.pixel_std
        features = self.backbone(x)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        box_cls = [torch.sigmoid_(x) for x in box_cls]
        box_cls = [permute_to_N_HWA_K(c, self.num_classes) for c in box_cls]
        box_delta = [permute_to_N_HWA_K(d, 4) for d in box_delta]
        return [torch.cat([d, c], dim=-1) for d, c in zip(box_delta, box_cls)]


def deploy(
    model: RetinaNet,
    path: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 540, 960),
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    opset_version: int = 10,
    verbose: bool = True,
    simplify: bool = True,
) -> Path:
    path = Path(path)
    output_dir = path.parent
    # export the RetinaNet model as an ONNX file
    assert len(input_shape) == 4, "input_shape must be (N, C, H, W)"
    wrapped_model = _RetinaNet(model).eval().to(model.device)
    dummy_input = torch.rand(input_shape).to(wrapped_model.device)
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        verbose=verbose,
        keep_initializers_as_inputs=True,
    )
    if simplify:
        simple_onnx, success = onnx_simplify(str(path), perform_optimization=True)
        assert success, "failed to simplify the exported ONNX file"
        onnx.save(simple_onnx, str(path))
    # export anchors as a NumPy file
    anchors = []
    cell_anchors = wrapped_model.anchor_generator.cell_anchors
    strides = wrapped_model.anchor_generator.strides
    grid_offset = wrapped_model.anchor_generator.offset
    for cell_anchor, stride in zip(cell_anchors, strides):
        cell_anchor = cell_anchor.cpu().numpy()
        x = np.arange(0, input_shape[3], stride) + grid_offset * stride
        y = np.arange(0, input_shape[2], stride) + grid_offset * stride
        grid = np.meshgrid(x, y)
        grid = np.concatenate([grid, grid])
        grid = np.transpose(grid[None], [2, 3, 0, 1])
        anchors.append(np.reshape(grid + cell_anchor, [-1, 4]))
    anchors_path = path.parent / "{}_anchors.npy".format(path.stem)
    np.save(anchors_path, anchors, allow_pickle=True)
    return output_dir
