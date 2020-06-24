from typing import Tuple, List

import math
import numpy as np


# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


def postprocess(
    outputs: List[np.ndarray],
    anchors: List[np.ndarray],
    weights: List[float],
    nms_thresh: float,
    score_thresh: float,
    top_k: int,
    max_detections: int,
    class_nms: bool,
) -> np.ndarray:
    results = []
    for img_idx in range(len(outputs[0])):
        box_cls_per_image = []
        box_reg_per_image = []
        for o in outputs:
            box_cls_per_image.append(o[img_idx, ..., 4:])
            box_reg_per_image.append(o[img_idx, ..., :4])

        results_per_image = postprocess_single_image(
            box_cls=box_cls_per_image,
            box_reg=box_reg_per_image,
            anchors=anchors,
            weights=weights,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            top_k=top_k,
            max_detections=max_detections,
            class_nms=class_nms,
        )
        results.append(results_per_image)
    return np.asanyarray(results)


def postprocess_single_image(
    box_cls: List[np.ndarray],
    box_reg: List[np.ndarray],
    anchors: List[np.ndarray],
    weights: List[float],
    nms_thresh: float,
    score_thresh: float,
    top_k: int,
    max_detections: int,
    class_nms: bool,
) -> np.ndarray:
    num_classes = box_cls[0].shape[-1]

    boxes_all = []
    scores_all = []
    class_idxs_all = []

    for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_reg, anchors):
        box_cls_i = box_cls_i.flatten()  # (HxWxAxK,)

        num_topk = min(top_k, box_reg_i.shape[0])
        topk_idxs = np.argsort(box_cls_i)[::-1]
        topk_idxs = topk_idxs[:num_topk]
        predicted_prob = box_cls_i[topk_idxs]

        keep_idxs = predicted_prob > score_thresh
        predicted_prob = predicted_prob[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]

        anchor_idxs = topk_idxs // num_classes
        classes_idxs = topk_idxs % num_classes

        box_reg_i = box_reg_i[anchor_idxs]
        anchors_i = anchors_i[anchor_idxs]

        predicted_boxes = apply_deltas(box_reg_i, anchors_i, weights)

        boxes_all.append(predicted_boxes)
        scores_all.append(predicted_prob)
        class_idxs_all.append(classes_idxs)

    boxes_all = np.reshape(np.concatenate(boxes_all), [-1, 4])
    scores_all = np.reshape(np.concatenate(scores_all), [-1, 1])
    class_idxs_all = np.reshape(np.concatenate(class_idxs_all), [-1, 1])

    if class_nms:
        keep = nms_per_class(boxes_all, scores_all, class_idxs_all, nms_thresh)
    else:
        keep = nms(boxes_all, scores_all, nms_thresh)

    n_detections = len(keep)

    boxes_all[..., 2:4] -= boxes_all[..., :2]  # ltrb -> ltwh

    if n_detections >= max_detections:
        keep = keep[:max_detections]
        detections = np.concatenate(
            [boxes_all[keep], class_idxs_all[keep], scores_all[keep]], axis=-1
        )
    else:
        detections = np.full(shape=(max_detections, 6), fill_value=-1.0)
        detections[:n_detections] = np.concatenate(
            [boxes_all[keep], class_idxs_all[keep], scores_all[keep]], axis=-1
        )

    return detections


def apply_deltas(
    deltas: np.ndarray,
    anchors: np.ndarray,
    weights: List[float],
    scale_clamp: float = _DEFAULT_SCALE_CLAMP,
) -> np.ndarray:
    assert np.isfinite(deltas).all(), "Box regression deltas become infinite or NaN!"

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = deltas[:, 2] / ww
    dh = deltas[:, 3] / wh

    dw = np.clip(dw, a_min=None, a_max=scale_clamp)
    dh = np.clip(dh, a_min=None, a_max=scale_clamp)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    pred_boxes = np.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2
    return pred_boxes


def nms_per_class(
    boxes: np.ndarray, scores: np.ndarray, class_indices: np.ndarray, iou_thresh: float
) -> np.ndarray:
    assert boxes.shape[-1] == 4
    result_mask = np.zeros_like(scores, dtype=np.bool)
    for i in np.unique(class_indices):
        mask = np.nonzero(class_indices == i)[0]
        keep = nms(boxes[mask], scores[mask], iou_thresh)
        result_mask[mask[keep]] = True
    keep = np.nonzero(result_mask)[0]
    keep = keep[np.argsort(scores[keep], axis=0)[::-1]]
    return keep.ravel()


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores, axis=0)[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i[0])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maximum height
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep
