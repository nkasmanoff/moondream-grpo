import numpy as np
import torch
from torch.optim import AdamW
from moondream_functions import detect, detect_grad
from refcoco_dataset import load_object_detection_dataset
from rl_utils import calculate_rewards, calculate_single_reward
from moondream import MoondreamModel, MoondreamConfig
from safetensors.torch import load_file
from rl_utils import calculate_grpo_loss
from safetensors.torch import save_file
import wandb
import logging
from visualization_utils import plot_prediction
import os
import math

# set. VIPS_WARNING=0 in environment
os.environ["VIPS_WARNING"] = "0"
os.environ["VIPS_INFO"] = "0"

NUM_EPOCHS = 1
BATCH_SIZE = 1
NUM_ROLLOUTS = 32
LEARNING_RATE = 5e-5
TRAIN_STEPS = 1

EVAL_INTERVAL = 50
VALIDATION_SAMPLES = 100
MAX_PLOT_SAMPLES = 27
safetensors_path = "model.safetensors"
device = "cuda" if torch.cuda.is_available() else "mps"


def lr_schedule(step, max_steps, constant=True):
    if constant:
        return LEARNING_RATE
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LEARNING_RATE + 0.9 * LEARNING_RATE * x / 0.1
    else:
        return (
            0.1 * LEARNING_RATE
            + 0.9 * LEARNING_RATE * (1 + math.cos(math.pi * (x - 0.1))) / 2
        )


def collect_experience(train_ds, model, start_idx):
    experience = []
    for i in range(BATCH_SIZE):
        sample = train_ds[start_idx + i]

        trajectory_detections = []

        for _ in range(NUM_ROLLOUTS):
            detections = detect(model, sample[0], sample[1], None, temperature=1.0)
            if len(detections["objects"]) == 0:
                # if no objects detected, skip this trajectory
                continue
            trajectory_detections.append(detections)

        if not trajectory_detections:
            experience.append([])
            continue

        rewards = calculate_rewards(trajectory_detections, sample)

        advantages = rewards - np.mean(rewards)

        advantages = advantages / np.std(advantages)

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


def train_step(experience, model, optimizer, train_ds, start_idx, num_steps=0):
    optimizer.zero_grad()
    total_loss = 0

    for i in range(BATCH_SIZE):
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
        loss = grpo_loss / BATCH_SIZE
        total_loss += loss

    if isinstance(total_loss, int):
        logging.error("Loss is an integer, skipping step")
        return 0
    if isinstance(total_loss, torch.Tensor) and torch.isnan(total_loss).any():
        logging.error("Loss is NaN, skipping step")
        return 0
    total_loss.backward()
    optimizer.step()
    lr_val = lr_schedule(num_steps, NUM_EPOCHS * len(train_ds) / BATCH_SIZE)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_val

    return total_loss.item() / BATCH_SIZE


def validate(model, val_ds, step, max_samples=VALIDATION_SAMPLES):
    model.eval()
    total_rewards = 0
    images = []
    with torch.no_grad():
        for i in range(max_samples):
            sample = val_ds[i]
            detections = detect(model, sample[0], sample[1], None, temperature=0)
            reward = calculate_single_reward(detections, sample)
            # plot sample
            if i < MAX_PLOT_SAMPLES:
                fig = plot_prediction(detections, sample)
                fig.savefig(f"predictions/prediction_{i}.png")
                images.append(wandb.Image(f"predictions/prediction_{i}.png"))
            total_rewards += reward
    wandb.log({"predictions": images[-10:]}, step=step)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total_rewards / max_samples


def validate_with_gt(val_ds, max_samples=VALIDATION_SAMPLES):
    total_rewards = 0
    for i in range(max_samples):
        sample = val_ds[i]
        detections = {"objects": sample[2]}
        reward = calculate_single_reward(detections, sample)
        total_rewards += reward
    return total_rewards / max_samples


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    os.makedirs("predictions", exist_ok=True)
    wandb.init(
        project="moondream-refcoco-detection",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_rollouts": NUM_ROLLOUTS,
            "train_steps_per_batch": TRAIN_STEPS,
            "validation_samples": VALIDATION_SAMPLES,
        },
    )
    num_steps = 0
    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    model.to(device)

    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
    optimizer = AdamW(
        [{"params": model.parameters()}],
        lr=LEARNING_RATE,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters: {num_params:,}")

    train_ds = load_object_detection_dataset("train")
    val_ds = load_object_detection_dataset("val")
    best_validation_score = float("-inf")
    gt_validation_score = validate_with_gt(val_ds, max_samples=VALIDATION_SAMPLES)
    logging.info(f"GT validation score: {round(gt_validation_score, 4)}")
    initial_validation_score = validate(
        model, val_ds, step=num_steps, max_samples=VALIDATION_SAMPLES
    )
    logging.info(f"Initial validation score: {round(initial_validation_score, 4)}")

    wandb.log(
        {
            "gt_validation_score": gt_validation_score,
            "initial_validation_score": initial_validation_score,
        },
        step=num_steps,
    )

    for epoch in range(NUM_EPOCHS):
        num_samples = len(train_ds)
        for start_idx in range(0, num_samples, BATCH_SIZE):
            with torch.no_grad():
                experience, _ = collect_experience(train_ds, model, start_idx)

            train_loss = train_step(
                experience, model, optimizer, train_ds, start_idx, num_steps
            )
            num_steps += 1
            logging.info(f"Step {num_steps} complete")
            lr_val = lr_schedule(num_steps, NUM_EPOCHS * len(train_ds) / BATCH_SIZE)

            wandb.log(
                {"train_loss": train_loss, "epoch": epoch, "lr": lr_val},
                step=num_steps,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if num_steps % EVAL_INTERVAL == 0:
                logging.info(f"Evaluating at step {num_steps}")
                validation_score = validate(
                    model, val_ds, step=num_steps, max_samples=VALIDATION_SAMPLES
                )
                logging.info(f"Validation score: {round(validation_score, 4)}")
                wandb.log({"validation_score": validation_score}, step=num_steps)
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    if not os.path.exists("models"):
                        os.makedirs("models")

                    model_path = f"models/grpo_model_{num_steps}.safetensors"
                    save_file(
                        model.state_dict(),
                        model_path,
                    )
                    #                    artifact = wandb.Artifact(f"grpo-model-{num_steps}", type="model")
                    #                    artifact.add_file(model_path)
                    #                    wandb.log_artifact(artifact)
                    logging.info(f"Saved model to {model_path}")
            logging.info(
                f"Epoch {epoch} batch {start_idx} loss: {round(train_loss, 4)}"
            )


    wandb.finish()


if __name__ == "__main__":
    main()
