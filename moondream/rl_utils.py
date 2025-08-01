import torch
from typing import Tuple, List
import numpy as np

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2) – in proportion form


def calculate_overlap(predicted_box, true_box):
    x_min = max(predicted_box["x_min"], true_box["x_min"])
    y_min = max(predicted_box["y_min"], true_box["y_min"])
    x_max = min(predicted_box["x_max"], true_box["x_max"])
    y_max = min(predicted_box["y_max"], true_box["y_max"])
    overlap = max(0, x_max - x_min) * max(0, y_max - y_min)
    return (
        overlap
        / (predicted_box["x_max"] - predicted_box["x_min"])
        * (predicted_box["y_max"] - predicted_box["y_min"])
    )


def calculate_object_center(predicted_box, true_box):
    # finds the distance between the centers of the predicted and true boxes
    predicted_center_x = (predicted_box["x_min"] + predicted_box["x_max"]) / 2
    predicted_center_y = (predicted_box["y_min"] + predicted_box["y_max"]) / 2
    true_center_x = (true_box["x_min"] + true_box["x_max"]) / 2
    true_center_y = (true_box["y_min"] + true_box["y_max"]) / 2

    distance = np.sqrt(
        (predicted_center_x - true_center_x) ** 2
        + (predicted_center_y - true_center_y) ** 2
    )
    # return 1 / distance but normalized to 0-1
    if distance == 0:
        return 1.0

    return 1 - distance


def calculate_single_reward(trajectory_detection, sample):
    trajectory_reward = float(0)
    predicted_boxes = trajectory_detection["objects"]
    true_boxes = sample[2]
    if len(predicted_boxes) != len(true_boxes):
        trajectory_reward -= 1
    else:
        trajectory_reward += 1
    for predicted_box, true_box in zip(predicted_boxes, true_boxes):
        overlap = calculate_overlap(predicted_box, true_box)
        center_distance = calculate_object_center(predicted_box, true_box)
        trajectory_reward += overlap + center_distance
    return trajectory_reward


def calculate_rewards(trajectory_detections, sample):
    total_rewards = []
    for trajectory_detection in trajectory_detections:
        reward = calculate_single_reward(trajectory_detection, sample)
        total_rewards.append(reward)

    return total_rewards


def calculate_grpo_loss(
    new_logprobs, old_logprobs, advantages, attention_mask, clip_epsilon=0.2
):
    importance_sampling_ratio = torch.exp(new_logprobs - old_logprobs)

    clipped = importance_sampling_ratio * advantages

    unclipped = (
        torch.clamp(importance_sampling_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        * advantages
    )
    loss = -torch.min(clipped, unclipped)

    # gpro loss is to take the sum where attention mask is 1, set to 0 otherwise

    grpo_loss = (loss * attention_mask).mean()
    return grpo_loss


def iou(a: Box, b: Box) -> float:
    """Corner-format IoU. Returns 0 when either box has zero area."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def match(gt: List[Box], pr: List[Box], iou_thr: float) -> Tuple[int, int, int]:
    """
    Greedy one-to-one matching with no confidences.
    Predictions are taken in the order produced by the model.
    """
    tp = fp = 0
    seen = [False] * len(gt)

    for p in pr:
        best, best_i = 0.0, -1
        for i, g in enumerate(gt):
            if seen[i]:
                continue
            iou_ = iou(p, g)
            if iou_ > best:
                best, best_i = iou_, i
        if best >= iou_thr:
            tp += 1
            seen[best_i] = True
        else:
            fp += 1

    fn = len(gt) - tp
    return tp, fp, fn


def calculate_detection_reward(trajectory_detections, sample):
    true_boxes = [(b["x_min"], b["y_min"], b["x_max"], b["y_max"]) for b in sample[2]]

    predicted_boxes = [
        (o["x_min"], o["y_min"], o["x_max"], o["y_max"])
        for o in trajectory_detections["objects"]
    ]
    tp, fp, fn = match(true_boxes, predicted_boxes, 0.5)
    reward = tp - fp - fn
    if len(true_boxes) == len(predicted_boxes):
        reward += 1
    else:
        reward -= 1
    return reward
