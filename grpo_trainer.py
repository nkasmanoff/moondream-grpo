"""
GRPO (Group Relative Policy Optimization) trainer for Moondream.

This trainer uses reinforcement learning to fine-tune the region head by:
- Collecting rollouts with multiple detections per sample
- Computing rewards based on detection quality
- Using GRPO loss to update the model based on advantages

Usage Examples:
---------------
Basic training with default hyperparameters:
    python grpo_trainer.py

Training with custom learning rate and batch size:
    python grpo_trainer.py --learning_rate=5e-5 --batch_size=5

Training with more rollouts:
    python grpo_trainer.py --num_rollouts=5 --num_epochs=3

Training with overfitting mode:
    python grpo_trainer.py --overfit_train=True --num_epochs=50

Training with Moondream 3:
    python grpo_trainer.py --md_version=3
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from datasets.basketball_dataset import load_object_detection_dataset
from moondream2.rl_utils import (
    calculate_rewards,
    match_boxes_score,
    calculate_grpo_loss,
)
from trainer_helpers import (
    lr_schedule,
    validate,
    validate_with_gt,
)
from safetensors.torch import load_file, save_file
import wandb
import logging
import os
import fire

# Default imports for Moondream 2 (can be overridden in main)
from moondream2.moondream_functions import (
    detect as _detect_md2,
    detect_grad as _detect_grad_md2,
)

# Module-level variables that can be reassigned based on md_version
detect = _detect_md2
detect_grad = _detect_grad_md2

device = "cuda" if torch.cuda.is_available() else "mps"


def collect_experience(train_ds, model, start_idx, batch_size, num_rollouts):
    experience = []
    for i in range(batch_size):
        sample = train_ds[start_idx + i]

        trajectory_detections = []

        for _ in range(num_rollouts):
            detections = detect(model, sample[0], sample[1], None, temperature=1.5)
            if len(detections["objects"]) == 0:
                # if no objects detected, skip this trajectory
                continue
            trajectory_detections.append(detections)

        if not trajectory_detections:
            experience.append([])
            continue

        rewards = calculate_rewards(trajectory_detections, sample)

        advantages = rewards - np.mean(rewards)

        advantages = advantages / max(np.std(advantages), 1e-6)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(model.device)
        advantages = advantages.unsqueeze(1)

        group_experience = []
        for j, trajectory in enumerate(trajectory_detections):
            predictions = trajectory["objects"]
            advantage = advantages[j].detach().cpu()
            logprobs = []
            for obj in predictions:
                x_logprob = obj["x_logprob"].detach().cpu()
                y_logprob = obj["y_logprob"].detach().cpu()
                w_logprob = obj["w_logprob"].detach().cpu()
                h_logprob = obj["h_logprob"].detach().cpu()
                logprobs.extend([x_logprob, y_logprob, w_logprob, h_logprob])

            # convert logits list to tensor
            logprobs = torch.tensor(logprobs, dtype=torch.float32)
            group_experience.append({"logprobs": logprobs, "advantage": advantage})
        experience.append(group_experience)

    return experience, trajectory_detections


def train_step(
    experience,
    model,
    optimizer,
    train_ds,
    start_idx,
    batch_size,
    learning_rate,
    num_epochs,
    train_ds_len,
    constant_lr,
    num_steps=0,
):
    optimizer.zero_grad()
    total_loss = 0

    for i in range(batch_size):
        sample = train_ds[start_idx + i]

        new_predictions = detect_grad(model, sample[0], sample[1], None, temperature=0)
        if len(new_predictions["out_logprobs"]) == 0:
            # if no objects detected, skip this sample
            continue
        new_logprobs = torch.stack(new_predictions["out_logprobs"]).reshape(
            -1, len(new_predictions["out_logprobs"])
        )

        old_logprobs_stack = []  # the prior log probs for the same corresponding sample
        advantages_stack = []
        # truncate experience to only be for this sample
        trajectory_experience = experience[i]
        if not trajectory_experience:
            continue

        attention_mask = torch.ones(
            len(trajectory_experience),
            new_logprobs.shape[-1],
            device=model.device,
        )

        for j, trajectory in enumerate(trajectory_experience):
            group_experience_logprobs = trajectory["logprobs"].to(model.device)
            # todo add padding and some kind of masking during gradient calculation
            orig_len = len(group_experience_logprobs)
            # pad right with 0s to match new_logprobs.shape[-1]
            if orig_len < new_logprobs.shape[-1]:
                group_experience_logprobs = torch.cat(
                    [
                        group_experience_logprobs,
                        torch.zeros(
                            new_logprobs.shape[-1] - orig_len, device=model.device
                        ),
                    ]
                )
                # set attention mask to 0 for the padded tokens
                attention_mask[j, orig_len:] = 0
            elif orig_len > new_logprobs.shape[-1]:
                group_experience_logprobs = group_experience_logprobs[
                    : new_logprobs.shape[-1]
                ]  # truncate
            old_logprobs_stack.append(group_experience_logprobs)
            advantages_stack.append(
                trajectory_experience[j]["advantage"].to(model.device)
            )

        if not advantages_stack:
            continue

        advantages = torch.stack(advantages_stack)
        old_logprobs = torch.stack(old_logprobs_stack)

        grpo_loss = calculate_grpo_loss(
            new_logprobs, old_logprobs, advantages, attention_mask
        )
        loss = grpo_loss / batch_size
        total_loss += loss

    if isinstance(total_loss, int):
        logging.error("Loss is an integer, skipping step")
        return 0
    if isinstance(total_loss, torch.Tensor) and torch.isnan(total_loss).any():
        logging.error("Loss is NaN, skipping step")
        return 0
    total_loss.backward()
    # apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.region.parameters(), 1.0)
    optimizer.step()
    lr_val = lr_schedule(
        num_steps,
        num_epochs * train_ds_len / batch_size,
        base_lr=learning_rate,
        constant=constant_lr,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_val

    return total_loss.item() / batch_size


def main(
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    batch_size: int = 3,
    num_rollouts: int = 3,
    train_steps: int = 1,
    validation_samples: int = 250,
    max_plot_samples: int = 25,
    overfit_train: bool = False,
    weight_decay: float = 1e-8,
    eval_interval: int = 1,
    constant_lr: bool = False,
    md_version: str = "2",
    wandb_project: str = "moondream-basketball-detection",
    dataset_name: str = "basketball-ball-detection",
):
    """
    Main GRPO training function with configurable hyperparameters via Fire CLI.

    Args:
        learning_rate: Learning rate (default: 1e-4)
        num_epochs: Number of training epochs (default: 1, or 50 if overfit_train=True)
        batch_size: Batch size for training (default: 3)
        num_rollouts: Number of rollouts per sample (default: 3, must be > 1)
        train_steps: Number of training steps per batch (default: 1)
        validation_samples: Number of samples to use for validation (default: 250)
        max_plot_samples: Maximum number of samples to plot during validation (default: 25)
        overfit_train: Whether to overfit on training set (default: False)
        weight_decay: Weight decay for optimizer (default: 1e-8)
        eval_interval: Evaluate every N steps (default: 1)
        constant_lr: Whether to use constant learning rate (default: False)
        md_version: Moondream version ("2" or "3") (default: "2")
        wandb_project: Weights & Biases project name (default: "moondream-basketball-detection")
        dataset_name: Dataset name for wandb logging (default: "basketball-ball-detection")
    """
    # Rollout warning
    if num_rollouts == 1:
        raise ValueError("num_rollouts must be greater than 1")

    # Adjust epochs based on overfit_train if not explicitly set
    if overfit_train and num_epochs == 1:
        num_epochs = 50

    # Adjust constant_lr based on overfit_train if not explicitly set
    if overfit_train and not constant_lr:
        constant_lr = True

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    os.makedirs("predictions", exist_ok=True)
    wandb.init(
        project=wandb_project,
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "num_rollouts": num_rollouts,
            "train_steps_per_batch": train_steps,
            "validation_samples": validation_samples,
            "overfit_train": overfit_train,
            "weight_decay": weight_decay,
            "eval_interval": eval_interval,
            "constant_lr": constant_lr,
            "dataset": dataset_name,
            "md_version": md_version,
        },
    )
    num_steps = 0

    # Import appropriate moondream version and update module-level detect/detect_grad
    global detect, detect_grad
    if md_version == "3":
        from moondream3.moondream import MoondreamModel, MoondreamConfig
        from moondream3.moondream_functions import (
            detect as detect_md3,
            detect_grad as detect_grad_md3,
        )

        detect = detect_md3
        detect_grad = detect_grad_md3
        safetensors_path = "models/model_md3.safetensors"
        setup_caches = False
    elif md_version == "2":
        from moondream2.moondream import MoondreamModel, MoondreamConfig
        from moondream2.moondream_functions import (
            detect as detect_md2,
            detect_grad as detect_grad_md2,
        )

        detect = detect_md2
        detect_grad = detect_grad_md2
        safetensors_path = "moondream2/model.safetensors"
        setup_caches = True
    else:
        raise ValueError(f"Invalid md_version: {md_version}. Must be '2' or '3'")

    model = MoondreamModel(config=MoondreamConfig(), setup_caches=setup_caches)

    # Load weights before moving to device
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)

    # Move model to device
    model.to(device)

    if md_version == "3":
        model._setup_caches()

    # Ensure all buffers and parameters are on the correct device
    for name, buffer in model.named_buffers():
        buffer.data = buffer.data.to(device)

    # Also explicitly move submodules
    model.text.to(device)
    model.vision.to(device)
    model.region.to(device)

    # Force all parameters to device
    for param in model.parameters():
        param.data = param.data.to(device)
        if param._grad is not None:
            param._grad.data = param._grad.data.to(device)

    # Verify all tensors are on the correct device
    device_str = str(device)
    for name, param in model.named_parameters():
        if str(param.device) != device_str:
            logging.warning(
                f"Parameter {name} is on {param.device}, moving to {device}"
            )
            param.data = param.data.to(device)

    for name, buffer in model.named_buffers():
        if str(buffer.device) != device_str:
            logging.warning(f"Buffer {name} is on {buffer.device}, moving to {device}")
            buffer.data = buffer.data.to(device)

    logging.info(f"Model successfully loaded and moved to {device}")

    optimizer = AdamW(
        [{"params": model.region.parameters()}],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    num_params = sum(p.numel() for p in model.region.parameters())
    logging.info(f"Number of parameters: {num_params:,}")

    train_ds = load_object_detection_dataset("train")
    val_split = "val" if not overfit_train else "train"
    val_ds = load_object_detection_dataset(val_split)

    logging.info(f"Train dataset size: {len(train_ds)}")
    logging.info(f"Val dataset size: {len(val_ds)}")
    logging.info(f"Validation will use {min(validation_samples, len(val_ds))} samples")
    gt_validation_score = validate_with_gt(val_ds, max_samples=validation_samples)
    logging.info(f"GT validation f1: {round(gt_validation_score['f1'], 4)}")
    logging.info(
        f"Validation will use {min(validation_samples, len(val_ds))} samples. Now running initial validation."
    )
    initial_validation_score = validate(
        model, val_ds, step=num_steps, max_samples=validation_samples
    )
    best_validation_score = initial_validation_score["f1"]

    logging.info(f"Initial validation f1: {round(initial_validation_score['f1'], 4)}")

    wandb.log(
        {
            "gt_validation_f1": gt_validation_score["f1"],
            "gt_validation_precision": gt_validation_score["precision"],
            "gt_validation_recall": gt_validation_score["recall"],
            "initial_validation_f1": initial_validation_score["f1"],
            "initial_validation_precision": initial_validation_score["precision"],
            "initial_validation_recall": initial_validation_score["recall"],
        },
        step=num_steps,
    )

    for epoch in range(num_epochs):
        num_samples = len(train_ds)
        for start_idx in range(0, num_samples, batch_size):
            with torch.no_grad():
                experience, _ = collect_experience(
                    train_ds, model, start_idx, batch_size, num_rollouts
                )

            train_loss = train_step(
                experience,
                model,
                optimizer,
                train_ds,
                start_idx,
                batch_size,
                learning_rate,
                num_epochs,
                len(train_ds),
                constant_lr,
                num_steps,
            )
            num_steps += 1
            logging.info(f"Step {num_steps} complete")
            lr_val = lr_schedule(
                num_steps,
                num_epochs * len(train_ds) / batch_size,
                base_lr=learning_rate,
                constant=constant_lr,
            )

            wandb.log(
                {"train_loss": train_loss, "epoch": epoch, "lr": lr_val},
                step=num_steps,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if num_steps % eval_interval == 0:
                logging.info(f"Evaluating at step {num_steps}")
                validation_score = validate(
                    model,
                    val_ds,
                    step=num_steps,
                    max_samples=validation_samples,
                    max_plot_samples=max_plot_samples,
                )
                logging.info(f"Validation f1: {round(validation_score['f1'], 4)}")
                wandb.log(
                    {
                        "validation_f1": validation_score["f1"],
                        "validation_precision": validation_score["precision"],
                        "validation_recall": validation_score["recall"],
                    },
                    step=num_steps,
                )
                if validation_score["f1"] > best_validation_score:
                    if not os.path.exists("models"):
                        os.makedirs("models")

                    model_path = f"models/grpo_model_{num_steps}.safetensors"
                    save_file(model.state_dict(), model_path)
                    logging.info(f"Saved model to {model_path}")
            logging.info(
                f"Epoch {epoch} batch {start_idx} loss: {round(train_loss, 4)}"
            )
            if overfit_train:
                break

    wandb.finish()


if __name__ == "__main__":
    """
    Run with Fire CLI. Examples:
        python grpo_trainer.py
        python grpo_trainer.py --learning_rate=5e-5 --batch_size=5
        python grpo_trainer.py --num_rollouts=5 --num_epochs=3
    """
    fire.Fire(main)
