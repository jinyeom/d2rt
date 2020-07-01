from pathlib import Path
from typing import Union, List, Tuple
import torch
from torch import nn

from detectron2.config import CfgNode
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances, pairwise_iou

from utils.tensorrt import TensorRTModule


class RetinaNetRT(nn.Module):
    # TODO: clean up the interface a bit
    # TODO: get `image_size` elsewhere

    def __init__(
        self,
        cfg: CfgNode,
        onnx_path: Union[str, Path],
        engine_path: Union[str, Path],
        anchors_path: Union[str, Path],
        image_size: Tuple[int, int],
        max_batch_size: int = 1,
        max_workspace_size: int = 1 << 25,
        fp16_mode: bool = False,
        force_rebuild: bool = False,
    ):
        super().__init__()
        # fmt: off
        self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES
        self.score_threshold          = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.model = TensorRTModule(
            onnx_path,
            engine_path,
            max_batch_size=max_batch_size,
            max_workspace_size=max_workspace_size,
            fp16_mode=fp16_mode,
            force_rebuild=force_rebuild,
        )
        self.anchors = torch.load(anchors_path)
        self.anchors = [anchor.cuda() for anchor in self.anchors]
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS
        )
        self.image_size = image_size

    def forward(self, inputs: torch.Tensor):
        outputs = self.model([inputs])
        return self.postprocess(outputs)

    def postprocess(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        results = []
        for img_idx in range(len(outputs[0])):
            box_cls_per_image = []
            box_reg_per_image = []
            for o in outputs:
                box_cls_per_image.append(o[img_idx, ..., 4:])
                box_reg_per_image.append(o[img_idx, ..., :4])

            results_per_image = self.postprocess_single_image(
                box_cls_per_image, box_reg_per_image,
            )
            results.append(results_per_image)
        return results

    def postprocess_single_image(self, box_cls, box_delta):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, self.anchors):
            box_cls_i = box_cls_i.flatten()  # (HxWxAxK,)

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(self.image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result
