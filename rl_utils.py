import torch


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
        trajectory_reward += overlap
    return trajectory_reward


def calculate_rewards(trajectory_detections, sample):
    total_rewards = []
    for trajectory_detection in trajectory_detections:
        reward = calculate_single_reward(trajectory_detection, sample)
        total_rewards.append(reward)

    return total_rewards


def calculate_gpro_loss(
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

    gpro_loss = (loss * attention_mask).mean()
    return gpro_loss
