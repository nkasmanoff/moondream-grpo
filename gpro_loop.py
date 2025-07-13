from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np
import torch
from torch.optim import AdamW
from visualization_utils import plot_sample, plot_prediction
from moondream_functions import detect, detect_grad
from dataset import load_object_detection_dataset
from moondream_cache import setup_caches
from rl_utils import calculate_reward
from moondream import MoondreamModel, MoondreamConfig
from safetensors.torch import load_file
from rl_utils import calculate_gpro_loss


BATCH_SIZE = 1
NUM_ROLLOUTS = 6
safetensors_path = "model.safetensors"


torch.autograd.set_detect_anomaly(True)


def collect_experience(train_ds, model):
    experience = []
    for i in range(BATCH_SIZE):
        sample = train_ds[11]
        #    hf_model.encode_image(sample[0], None); # initalize the  kv_cache for this sample

        trajectory_detections = []

        for i in range(NUM_ROLLOUTS):
            detections = detect(model, sample[0], sample[1], None, temperature=1)
            trajectory_detections.append(detections)

        rewards = calculate_reward(trajectory_detections, sample)
        advantages = rewards - np.mean(rewards)

        advantages = advantages / np.std(advantages)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(model.device)
        advantages = advantages.unsqueeze(1)

        group_experience = []
        for i, traj in enumerate(trajectory_detections):
            predictions = traj["objects"]
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

    return experience


def train_step(experience, model, optimizer, train_ds):

    optimizer.zero_grad()
    total_loss = 0

    for i in range(BATCH_SIZE):
        sample = train_ds[11]

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

        for j in range(len(trajectory_experience)):

            # todo add padding and some kind of masking during gradient calculation
            group_experience_logprobs = trajectory_experience[j]["logprobs"]
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
        print(total_loss)
        final_loss = total_loss / BATCH_SIZE
        final_loss.backward()
        optimizer.step()


def main():
    model = MoondreamModel(config=MoondreamConfig)
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
    model.to("mps")

    optimizer = AdamW(model.region.parameters(), lr=1e-4)

    num_params = sum(p.numel() for p in model.region.parameters())
    print(f"Number of parameters: {num_params:,}")

    train_ds = load_object_detection_dataset("train")

    print("Collecting experience")
    with torch.no_grad():
        experience = collect_experience(train_ds, model)
    print("Training step")
    train_step(experience, model, optimizer, train_ds)
    print("Training step 2")
    train_step(experience, model, optimizer, train_ds)


if __name__ == "__main__":
    main()
