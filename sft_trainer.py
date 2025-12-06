"""
Graciously adapted from https://github.com/moondream-ai/moondream/blob/main/finetune_region.py
"""

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
from safetensors.torch import save_file
import datasets
import logging
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.optim import AdamW
import wandb
from safetensors.torch import load_file

# Import basketball dataset
from datasets.basketball_dataset import BasketballCocoDataset

# Set moondream version
MD_VERSION = "2"
if MD_VERSION == "3":
    from moondream3.moondream import MoondreamModel, MoondreamConfig
    from moondream3.moondream_functions import detect
    from moondream3.moondream import text_encoder
    from moondream3.text import _produce_hidden
    from moondream3.region import (
        decode_coordinate,
        decode_size,
        encode_coordinate,
        encode_size,
    )

    MODEL_PATH = "models/model_md3.safetensors"
elif MD_VERSION == "2":
    from moondream2.moondream import MoondreamModel, MoondreamConfig, text_encoder
    from moondream2.moondream_functions import detect
    from moondream2.text import _produce_hidden
    from moondream2.region import (
        decode_coordinate,
        decode_size,
        encode_coordinate,
        encode_size,
    )

    MODEL_PATH = "moondream2/model.safetensors"
else:
    raise ValueError(f"Invalid MD_VERSION: {MD_VERSION}")

# Import validation utilities
from moondream2.rl_utils import match_boxes_score
from moondream2.visualization_utils import plot_prediction


# This is a intended to be a basic starting point. Your optimal hyperparams and data may be different.
LR = 1e-5
EPOCHS = 1
GRAD_ACCUM_STEPS = 128
VALIDATION_SAMPLES = 250
MAX_PLOT_SAMPLES = 25
EVAL_INTERVAL = 100  # Evaluate every N gradient accumulation steps
device = "cuda" if torch.cuda.is_available() else "mps"


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


def region_loss(
    hidden_states: torch.Tensor,
    w,
    labels: torch.Tensor,
    c_idx: torch.Tensor,
    s_idx: torch.Tensor,
):
    l_idx = torch.arange(len(labels))

    c_idx = c_idx - 1
    c_hidden = hidden_states[:, c_idx, :]
    c_logits = decode_coordinate(c_hidden, w)
    c_labels = labels[(l_idx % 4) < 2]

    c_loss = F.cross_entropy(
        c_logits.view(-1, c_logits.size(-1)),
        c_labels,
    )

    s_idx = s_idx - 1
    s_hidden = hidden_states[:, s_idx, :]
    s_logits = decode_size(s_hidden, w).view(-1, 1024)
    s_labels = labels[(l_idx % 4) >= 2]

    s_loss = F.cross_entropy(s_logits, s_labels)

    return c_loss + s_loss


class WasteDetection(Dataset):
    def __init__(self, split: str = "train"):
        self.dataset: datasets.Dataset = datasets.load_dataset(
            "moondream/waste_detection", split=split
        )
        self.dataset = self.dataset.shuffle(seed=111)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"]
        boxes = row["boxes"]
        labels = row["labels"]

        objects = {}
        for box, label in zip(boxes, labels):
            objects.setdefault(label, []).append(box)

        flat_boxes = []
        class_names = []
        for label, box_list in objects.items():
            for b in box_list:
                flat_boxes.append(b)  # x, y, w, h , normalized to 0-1
                class_names.append(label)

        flat_boxes = torch.as_tensor(flat_boxes, dtype=torch.float16)
        image_id = torch.tensor([idx], dtype=torch.int64)

        return {
            "image": image,
            "boxes": flat_boxes,
            "class_names": class_names,
            "image_id": image_id,
        }

    def get_sample_for_validation(self, idx):
        """Get sample in format compatible with validation functions"""
        row = self.dataset[idx]
        image = row["image"]
        boxes = row["boxes"]
        labels = row["labels"]

        # Convert to normalized boxes format expected by validation
        normalized_boxes = []
        for box in boxes:
            # box is already [x, y, w, h] normalized to 0-1
            x, y, w, h = box
            normalized_boxes.append(
                {
                    "x_min": x,
                    "y_min": y,
                    "x_max": x + w,
                    "y_max": y + h,
                }
            )

        # Use first label as the query
        label = labels[0] if labels else "object"

        return image, label, normalized_boxes


class BasketballDetection(Dataset):
    def __init__(self, split: str = "train"):
        """Wrapper for basketball dataset to match SFT trainer format"""
        dataset_root = "datasets/basketball-player-detection-3.v1i.coco"

        if split == "train":
            self.dataset = BasketballCocoDataset(
                dataset_root, split="train", categories_to_use=["ball"]
            )
        elif split == "val":
            self.dataset = BasketballCocoDataset(
                dataset_root, split="valid", categories_to_use=["ball"]
            )
        else:
            self.dataset = BasketballCocoDataset(
                dataset_root, split="test", categories_to_use=["ball"]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, boxes = self.dataset[idx]

        # Convert from normalized [x_min, y_min, x_max, y_max] format
        # to [x, y, w, h] format expected by SFT trainer
        flat_boxes = []
        class_names = []
        for box in boxes:
            x = box["x_min"]
            y = box["y_min"]
            w = box["x_max"] - box["x_min"]
            h = box["y_max"] - box["y_min"]
            flat_boxes.append([x, y, w, h])
            class_names.append(label)

        flat_boxes = torch.as_tensor(flat_boxes, dtype=torch.float16)
        image_id = torch.tensor([idx], dtype=torch.int64)

        return {
            "image": image,
            "boxes": flat_boxes,
            "class_names": class_names,
            "image_id": image_id,
        }

    def get_sample_for_validation(self, idx):
        """Get sample in format compatible with validation functions"""
        # Just use the underlying dataset's format which is already correct
        return self.dataset[idx]


class RefCocoDetection(Dataset):
    def __init__(self, split: str = "train"):
        self.dataset: datasets.Dataset = datasets.load_dataset(
            "lmms-lab/RefCOCO", split="val" if split == "train" else "test"
        )
        self.dataset = self.dataset.shuffle(seed=111)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"]
        labels = row["answer"]
        boxes = [row["bbox"]]

        flat_boxes = []
        class_names = []
        for label, box in zip(labels, boxes):
            x, y, w, h = box
            x = x / image.width
            y = y / image.height
            w = w / image.width
            h = h / image.height
            flat_boxes.append([x, y, w, h])
            class_names.append(label)

        flat_boxes = torch.as_tensor(flat_boxes, dtype=torch.float16)
        image_id = torch.tensor([idx], dtype=torch.int64)

        return {
            "image": image,
            "boxes": flat_boxes,
            "class_names": class_names,
            "image_id": image_id,
        }

    def get_sample_for_validation(self, idx):
        """Get sample in format compatible with validation functions"""
        row = self.dataset[idx]
        image = row["image"]
        labels = row["answer"]
        boxes = [row["bbox"]]

        # Convert to normalized boxes format expected by validation
        normalized_boxes = []
        for box in boxes:
            x, y, w, h = box
            x = x / image.width
            y = y / image.height
            w = w / image.width
            h = h / image.height
            normalized_boxes.append(
                {
                    "x_min": x,
                    "y_min": y,
                    "x_max": x + w,
                    "y_max": y + h,
                }
            )

        # Use first label as the query
        label = labels[0] if labels else "object"

        return image, label, normalized_boxes


def validate(model, val_ds, step, max_samples=VALIDATION_SAMPLES):
    """Validate the model on the validation set"""
    model.eval()
    TP = FP = FN = 0

    # Use the minimum of max_samples and the actual dataset size
    num_samples = min(max_samples, len(val_ds))

    images = []
    with torch.no_grad():
        for i in range(num_samples):
            sample = val_ds.get_sample_for_validation(i)
            detections = detect(model, sample[0], sample[1], None, temperature=0)
            tp, fp, fn = match_boxes_score(detections["objects"], sample[2])

            # plot sample
            if i < MAX_PLOT_SAMPLES:
                try:
                    fig = plot_prediction(detections, sample)
                    fig_path = f"predictions/prediction_{i}.png"
                    fig.savefig(fig_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)  # Free up memory
                    images.append(wandb.Image(fig_path))
                except Exception as e:
                    logging.warning(f"Failed to save prediction image {i}: {e}")
            TP += tp
            FP += fp
            FN += fn

    # Only log images if we have any
    if images:
        try:
            wandb.log({"predictions": images[-MAX_PLOT_SAMPLES:]}, step=step)
        except Exception as e:
            logging.warning(f"Failed to log prediction images to wandb: {e}")
    
    model.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if TP + FP == 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def validate_with_gt(val_ds, max_samples=VALIDATION_SAMPLES):
    """Validate with ground truth (perfect detection)"""
    # Use the minimum of max_samples and the actual dataset size
    num_samples = min(max_samples, len(val_ds))

    TP = FP = FN = 0
    for i in range(num_samples):
        sample = val_ds.get_sample_for_validation(i)
        detections = {"objects": sample[2]}
        tp, fp, fn = match_boxes_score(detections["objects"], sample[2])
        TP += tp
        FP += fp
        FN += fn

    if TP + FP == 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    os.makedirs("predictions", exist_ok=True)

    wandb.init(
        project="moondream-basketball-ft",
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
            "VALIDATION_SAMPLES": VALIDATION_SAMPLES,
            "EVAL_INTERVAL": EVAL_INTERVAL,
            "dataset": "basketball-ball-detection",
            "md_version": MD_VERSION,
        },
    )

    # Setup model with appropriate cache settings based on version
    if MD_VERSION == "3":
        setup_caches = False
    else:
        setup_caches = True

    model = MoondreamModel(config=MoondreamConfig(), setup_caches=setup_caches)
    model.to(device)

    state_dict = load_file(MODEL_PATH)
    model.load_state_dict(state_dict)

    # For MD3, setup caches after loading weights
    if MD_VERSION == "3":
        model._setup_caches()

    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters: {num_params:,}")

    optimizer = AdamW(
        [{"params": model.parameters()}],
        lr=LR,
    )

    # Load basketball dataset (filtered to "ball" category)
    dataset = BasketballDetection(split="train")
    val_dataset = BasketballDetection(split="val")

    logging.info(f"Train dataset size: {len(dataset)}")
    logging.info(f"Val dataset size: {len(val_dataset)}")
    logging.info(
        f"Validation will use {min(VALIDATION_SAMPLES, len(val_dataset))} samples"
    )

    # Run initial validation
    gt_validation_score = validate_with_gt(val_dataset, max_samples=VALIDATION_SAMPLES)
    logging.info(f"GT validation f1: {round(gt_validation_score['f1'], 4)}")

    initial_validation_score = validate(
        model, val_dataset, step=0, max_samples=VALIDATION_SAMPLES
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
        step=0,
    )

    total_steps = EPOCHS * len(dataset) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    i = 0
    for epoch in range(EPOCHS):
        for sample in dataset:
            i += 1

            with torch.no_grad():
                img_emb = model._run_vision_encoder(sample["image"])
                bos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.bos_id]], device=model.device
                    ),
                    model.text,
                )
                eos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.eos_id]], device=model.device
                    ),
                    model.text,
                )

            boxes_by_class = {}
            for box, cls in zip(sample["boxes"], sample["class_names"]):
                boxes_by_class.setdefault(cls, []).append(box)

            total_loss = 0.0
            for class_name, boxes_list in boxes_by_class.items():
                with torch.no_grad():
                    instruction = f"\n\nDetect: {class_name}\n\n"
                    instruction_tokens = model.tokenizer.encode(instruction).ids
                    instruction_emb = text_encoder(
                        torch.tensor([[instruction_tokens]], device=model.device),
                        model.text,
                    ).squeeze(0)

                cs_emb = []
                cs_labels = []
                c_idx = []
                s_idx = []
                for bb in boxes_list:
                    # set device of bb to model.device
                    bb = bb.to(dtype=torch.bfloat16, device=model.device)
                    l_cs = len(cs_emb)
                    cs_emb.extend(
                        [
                            encode_coordinate(bb[0].unsqueeze(0), model.region),
                            encode_coordinate(bb[1].unsqueeze(0), model.region),
                            encode_size(bb[2:4], model.region),
                        ]
                    )
                    c_idx.extend([l_cs, l_cs + 1])
                    s_idx.append(l_cs + 2)

                    # Create coordinate bin labels
                    coord_labels = [
                        min(max(torch.round(p * 1023), 0), 1023).item() for p in bb[:2]
                    ]

                    # Create size bin labels using log-scale mapping
                    s_log2_bins = []
                    for s_val in bb[2:4]:
                        s_val = float(s_val)
                        s_clamped = max(s_val, 1 / 1024)
                        s_log2 = math.log2(s_clamped)
                        mapped = (s_log2 + 10.0) / 10.0 * 1023.0
                        s_bin = int(round(mapped))
                        s_bin = max(min(s_bin, 1023), 0)
                        s_log2_bins.append(s_bin)

                    # Combine coordinate and size bin labels
                    cs_labels.extend(coord_labels + s_log2_bins)

                if len(cs_emb) == 0:
                    continue
                cs_emb = torch.stack(cs_emb)

                inputs_embeds = torch.cat(
                    [bos_emb, img_emb[None], instruction_emb, cs_emb[None], eos_emb],
                    dim=1,
                )
                prefix = inputs_embeds.size(1) - cs_emb.size(0)
                c_idx = torch.tensor(c_idx, device=model.device) + prefix
                s_idx = torch.tensor(s_idx, device=model.device) + prefix

                hidden = _produce_hidden(
                    inputs_embeds=inputs_embeds, w=model.text, config=model.config.text
                )

                loss = region_loss(
                    hidden_states=hidden,
                    w=model.region,
                    labels=torch.tensor(
                        cs_labels, dtype=torch.int64, device=model.device
                    ),
                    c_idx=c_idx,
                    s_idx=s_idx,
                )
                total_loss += loss

            total_loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr_val = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_val

                current_step = i // GRAD_ACCUM_STEPS
                pbar.set_postfix({"step": current_step, "loss": total_loss.item()})
                pbar.update(1)

                wandb.log(
                    {
                        "loss/train": total_loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=current_step,
                )

                # Periodic validation
                if current_step % EVAL_INTERVAL == 0 and current_step > 0:
                    logging.info(f"Evaluating at step {current_step}")
                    validation_score = validate(
                        model,
                        val_dataset,
                        step=current_step,
                        max_samples=VALIDATION_SAMPLES,
                    )
                    logging.info(f"Validation f1: {round(validation_score['f1'], 4)}")

                    wandb.log(
                        {
                            "validation_f1": validation_score["f1"],
                            "validation_precision": validation_score["precision"],
                            "validation_recall": validation_score["recall"],
                        },
                        step=current_step,
                    )

                    # Save best model
                    if validation_score["f1"] > best_validation_score:
                        best_validation_score = validation_score["f1"]
                        save_file(
                            model.state_dict(),
                            f"moondream_best_step_{current_step}.safetensors",
                        )
                        logging.info(
                            f"Saved best model with F1: {round(best_validation_score, 4)}"
                        )
    wandb.finish()

    # Replace with your desired output location.
    save_file(
        model.state_dict(),
        "moondream_finetune.safetensors",
    )


if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_region
    """
    main()
