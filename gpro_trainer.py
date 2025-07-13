import numpy as np
import torch
from torch.optim import AdamW
from moondream_functions import detect, detect_grad
from dataset import load_object_detection_dataset
from rl_utils import calculate_rewards, calculate_single_reward
from moondream import MoondreamModel, MoondreamConfig
from safetensors.torch import load_file
from rl_utils import calculate_gpro_loss
from safetensors.torch import save_file


NUM_EPOCHS = 1
BATCH_SIZE = 2
NUM_ROLLOUTS = 6
LEARNING_RATE = 1e-4
TRAIN_STEPS = 2
EVAL_INTERVAL = 4
safetensors_path = "model.safetensors"
device = "cuda" if torch.cuda.is_available() else "mps"
torch.autograd.set_detect_anomaly(True)


def collect_experience(train_ds, model, start_idx):
    experience = []
    for i in range(BATCH_SIZE):
        sample = train_ds[start_idx + i]

        trajectory_detections = []

        for _ in range(NUM_ROLLOUTS):
            detections = detect(model, sample[0], sample[1], None, temperature=1)
            trajectory_detections.append(detections)

        rewards = calculate_rewards(trajectory_detections, sample)

        advantages = rewards - np.mean(rewards)

        advantages = advantages / np.std(advantages)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(model.device)
        advantages = advantages.unsqueeze(1)

        group_experience = []
        for trajectory in trajectory_detections:
            predictions = trajectory["objects"]
            advantage = advantages[i]
            logprobs = []
            for obj in predictions:
                x_logprob = obj["x_logprob"]
                y_logprob = obj["y_logprob"]
                w_logprob = obj["w_logprob"]
                h_logprob = obj["h_logprob"]
                logprobs.extend([x_logprob, y_logprob, w_logprob, h_logprob])

            # convert logits list to tensor
            logprobs = torch.tensor(logprobs, dtype=torch.float32).to(model.device)
            group_experience.append({"logprobs": logprobs, "advantage": advantage})
        experience.extend(group_experience)

    return experience, trajectory_detections


def train_step(experience, model, optimizer, train_ds, start_idx):

    optimizer.zero_grad()
    total_loss = 0

    for i in range(BATCH_SIZE):
        sample = train_ds[start_idx + i]

        new_predictions = detect_grad(model, sample[0], sample[1], None, temperature=0)
        new_logprobs = torch.stack(new_predictions["out_logprobs"]).reshape(
            -1, len(new_predictions["out_logprobs"])
        )

        old_logprobs_stack = []  # the prior log probs for the same corresponding sample
        advantages_stack = []
        # truncate experience to only be for this sample
        trajectory_experience = experience[i * NUM_ROLLOUTS : (i + 1) * NUM_ROLLOUTS]

        attention_mask = torch.ones(
            len(trajectory_experience),
            new_logprobs.shape[-1],
            device=model.device,
        )

        for j, trajectory in enumerate(trajectory_experience):
            group_experience_logprobs = trajectory["logprobs"]
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
            advantages_stack.append(trajectory_experience[j]["advantage"])

        advantages = torch.stack(advantages_stack)
        old_logprobs = torch.stack(old_logprobs_stack)

        gpro_loss = calculate_gpro_loss(
            new_logprobs, old_logprobs, advantages, attention_mask
        )
        total_loss += gpro_loss

    if BATCH_SIZE > 0:
        final_loss = total_loss / BATCH_SIZE
        final_loss.backward()
        optimizer.step()

    return final_loss.item(), new_predictions


def validate(model, val_ds, max_samples=10):
    model.eval()
    total_rewards = 0
    for i in range(max_samples):
        sample = val_ds[i]
        detections = detect(model, sample[0], sample[1], None, temperature=0)
        reward = calculate_single_reward(detections, sample)
        total_rewards += reward
    model.train()
    return total_rewards / max_samples


def main():
    num_steps = 0
    model = MoondreamModel(config=MoondreamConfig)
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
    model.to(device)

    optimizer = AdamW(model.region.parameters(), lr=LEARNING_RATE)

    num_params = sum(p.numel() for p in model.region.parameters())
    print(f"Number of parameters: {num_params:,}")

    train_ds = load_object_detection_dataset("train")
    val_ds = load_object_detection_dataset("test")
    best_validation_score = float("-inf")
    for epoch in range(NUM_EPOCHS):
        num_samples = len(train_ds)
        for start_idx in range(0, num_samples, BATCH_SIZE):
            with torch.no_grad():
                experience, _ = collect_experience(train_ds, model, start_idx)
            # take 2 steps for each batch
            for _ in range(TRAIN_STEPS):
                train_loss, _ = train_step(
                    experience, model, optimizer, train_ds, start_idx
                )
            num_steps += 1
            print(f"Step {num_steps} complete")

            if num_steps % EVAL_INTERVAL == 0:
                print(f"Evaluating at step {num_steps}")
                validation_score = validate(model, val_ds, max_samples=2)
                print(f"Validation score: {round(validation_score, 4)}")
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    save_file(
                        model.state_dict(),
                        f"gpro_model_{num_steps}.safetensors",
                    )
            print(f"Epoch {epoch} batch {start_idx} loss: {round(train_loss, 4)}")


if __name__ == "__main__":
    main()
