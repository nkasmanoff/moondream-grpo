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


def calculate_reward(trajectory_detections, sample):
    # For each trajectory, calculate the reward for the object detection model

    # Things like: same number of objects
    # overlap in bounding box corresponding

    total_rewards = []
    for i in range(len(trajectory_detections)):
        trajectory_reward = float(0)
        predicted_boxes = trajectory_detections[i]["objects"]
        true_boxes = sample[2]
        if len(predicted_boxes) != len(true_boxes):
            trajectory_reward -= 1
        else:
            trajectory_reward += 1
        for j in range(
            min(len(predicted_boxes), len(true_boxes))
        ):  # loop through the smaller of the two
            predicted_box = predicted_boxes[j]
            true_box = true_boxes[j]
            overlap = calculate_overlap(predicted_box, true_box)
            trajectory_reward += overlap
        total_rewards.append(trajectory_reward)

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
