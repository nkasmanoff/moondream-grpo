"""
Graciously adapted from https://github.com/moondream-ai/moondream/blob/main/finetune_region.py

LoRA Fine-Tuning:
-----------------
This trainer supports both full fine-tuning and LoRA (Low-Rank Adaptation) fine-tuning.

LoRA Benefits:
- Much fewer trainable parameters (~1-2% of full model)
- Faster training and lower memory usage
- Smaller checkpoint files (only LoRA weights saved)
- Can be merged with base model or swapped at inference time

To enable LoRA, set USE_LORA = True and configure:
- LORA_RANK: Rank of LoRA matrices (8-64, default 16)
- LORA_ALPHA: Scaling factor (typically 2x rank, default 32)
- LORA_DROPOUT: Dropout probability (default 0.1)
- LORA_TARGET_MODULES: Which layers to apply LoRA to

LoRA adapters are saved separately as "moondream_lora_*.safetensors"

Usage Examples:
---------------
Basic training with default hyperparameters:
    python sft_trainer.py

Training with custom learning rate and epochs:
    python sft_trainer.py --lr=1e-3 --epochs=20

Training with LoRA disabled (full fine-tuning):
    python sft_trainer.py --use_lora=False

Training with custom LoRA parameters:
    python sft_trainer.py --lora_rank=32 --lora_alpha=64 --lora_dropout=0.2

Training with overfitting mode (small batch):
    python sft_trainer.py --overfit_batch_size=8 --epochs=5

Training with Moondream 3:
    python sft_trainer.py --md_version=3

Combining multiple parameters:
    python sft_trainer.py --lr=1e-3 --epochs=15 --grad_accum_steps=2 --eval_interval=100 --validation_samples=500
"""

import torch
from torch.utils.data import Dataset
from safetensors.torch import save_file
import datasets
import logging
import os
import math

from tqdm import tqdm
from torch.optim import AdamW
import wandb
from safetensors.torch import load_file
import fire

# Import basketball dataset
from datasets.basketball_dataset import BasketballCocoDataset

# Import shared helper functions
from trainer_helpers import (
    LoRALinear,
    inject_lora_into_model,
    get_lora_state_dict,
    lr_schedule,
    region_loss,
    validate,
    validate_with_gt,
)


device = "cuda" if torch.cuda.is_available() else "mps"


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

        # Use float32 for better numerical stability and compatibility
        flat_boxes = torch.as_tensor(flat_boxes, dtype=torch.float32)
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
                dataset_root, split="train", categories_to_use=["player"]
            )
        elif split == "val":
            self.dataset = BasketballCocoDataset(
                dataset_root, split="valid", categories_to_use=["player"]
            )
        else:
            self.dataset = BasketballCocoDataset(
                dataset_root, split="test", categories_to_use=["player"]
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

        # Use float32 for better numerical stability and compatibility
        flat_boxes = torch.as_tensor(flat_boxes, dtype=torch.float32)
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

        # Use float32 for better numerical stability and compatibility
        flat_boxes = torch.as_tensor(flat_boxes, dtype=torch.float32)
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


def main(
    lr: float = 5e-4,
    epochs: int = 10,
    grad_accum_steps: int = 1,
    validation_samples: int = 250,
    max_plot_samples: int = 25,
    eval_interval: int = 50,
    overfit_batch_size: int = 4,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_modules: list = None,
    md_version: str = "2",
    wandb_project: str = "moondream-basketball-ft",
    dataset_name: str = "basketball-ball-detection",
):
    """
    Main training function with configurable hyperparameters via Fire CLI.

    Args:
        lr: Learning rate (default: 5e-4)
        epochs: Number of training epochs (default: 10)
        grad_accum_steps: Gradient accumulation steps (default: 1)
        validation_samples: Number of samples to use for validation (default: 250)
        max_plot_samples: Maximum number of samples to plot during validation (default: 25)
        eval_interval: Evaluate every N gradient accumulation steps (default: 50)
        overfit_batch_size: Set to > 0 to overfit on a small batch (default: 4)
        use_lora: Whether to use LoRA instead of full fine-tuning (default: True)
        lora_rank: Rank of LoRA matrices (default: 16)
        lora_alpha: Scaling factor for LoRA, typically 2x rank (default: 32)
        lora_dropout: Dropout for LoRA layers (default: 0.1)
        lora_target_modules: Which layers to apply LoRA to (default: ["qkv", "proj", "fc1", "fc2"])
        md_version: Moondream version ("2" or "3") (default: "2")
        wandb_project: Weights & Biases project name (default: "moondream-basketball-ft")
        dataset_name: Dataset name for wandb logging (default: "basketball-ball-detection")
    """
    # Set default lora_target_modules if None
    if lora_target_modules is None:
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"]

    # Import appropriate moondream version
    if md_version == "3":
        from moondream3.moondream import MoondreamModel, MoondreamConfig
        from moondream3.moondream_functions import detect
        from moondream3.moondream import text_encoder
        from moondream3.text import _produce_hidden
        from moondream3.region import (
            encode_coordinate,
            encode_size,
        )

        model_path = "models/model_md3.safetensors"
    elif md_version == "2":
        from moondream2.moondream import MoondreamModel, MoondreamConfig, text_encoder
        from moondream2.moondream_functions import detect
        from moondream2.text import _produce_hidden
        from moondream2.region import (
            encode_coordinate,
            encode_size,
        )

        model_path = "moondream2/model.safetensors"
    else:
        raise ValueError(f"Invalid md_version: {md_version}. Must be '2' or '3'")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    os.makedirs("predictions", exist_ok=True)

    wandb.init(
        project=wandb_project,
        config={
            "EPOCHS": epochs,
            "GRAD_ACCUM_STEPS": grad_accum_steps,
            "LR": lr,
            "VALIDATION_SAMPLES": validation_samples,
            "EVAL_INTERVAL": eval_interval,
            "OVERFIT_BATCH_SIZE": overfit_batch_size,
            "USE_LORA": use_lora,
            "LORA_RANK": lora_rank if use_lora else None,
            "LORA_ALPHA": lora_alpha if use_lora else None,
            "LORA_DROPOUT": lora_dropout if use_lora else None,
            "dataset": dataset_name,
            "md_version": md_version,
        },
    )

    # Setup model with appropriate cache settings based on version
    if md_version == "3":
        setup_caches = False
    else:
        setup_caches = True

    model = MoondreamModel(config=MoondreamConfig(), setup_caches=setup_caches)

    # Load weights before moving to device
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)

    # Move model to device
    model.to(device)

    # For MD3, setup caches after loading weights
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

    # Apply LoRA if enabled
    if use_lora:
        logging.info("Applying LoRA to model...")
        lora_params = inject_lora_into_model(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )

        # Freeze all parameters, then unfreeze only LoRA weights.
        # This ensures we're really doing LoRA-style fine-tuning, not (almost) full fine-tuning.
        for param in model.parameters():
            param.requires_grad = False

        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

        # Count total and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable LoRA parameters: {trainable_params:,}")
        logging.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        # Optimizer only for LoRA parameters
        optimizer = AdamW(
            [{"params": [p for p in model.parameters() if p.requires_grad]}],
            lr=lr,
        )
    else:
        num_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Number of parameters: {num_params:,}")

        optimizer = AdamW(
            [{"params": model.parameters()}],
            lr=lr,
        )

    # Load basketball dataset (filtered to "player" category)
    dataset = BasketballDetection(split="train")
    val_dataset = BasketballDetection(split="val")

    # Handle overfit batch size
    if overfit_batch_size is not None and overfit_batch_size > 0:
        logging.info(
            f"Overfitting mode enabled: using first {overfit_batch_size} training samples for both train and val"
        )
        # Limit the dataset to first N samples by modifying the underlying dataset
        train_full = BasketballDetection(split="train")
        # Truncate the underlying COCO dataset
        train_full.dataset.image_ids = train_full.dataset.image_ids[:overfit_batch_size]
        dataset = train_full

        # Use the same subset for validation
        val_full = BasketballDetection(split="train")
        val_full.dataset.image_ids = val_full.dataset.image_ids[:overfit_batch_size]
        val_dataset = val_full

    logging.info(f"Train dataset size: {len(dataset)}")
    logging.info(f"Val dataset size: {len(val_dataset)}")
    logging.info(
        f"Validation will use {min(validation_samples, len(val_dataset))} samples"
    )

    # Run initial validation
    gt_validation_score = validate_with_gt(val_dataset, max_samples=validation_samples)
    logging.info(f"GT validation f1: {round(gt_validation_score['f1'], 4)}")

    initial_validation_score = validate(
        model, val_dataset, step=0, max_samples=validation_samples
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

    total_steps = epochs * len(dataset) // grad_accum_steps
    pbar = tqdm(total=total_steps)

    i = 0
    for epoch in range(epochs):
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

            total_loss = None
            for class_name, boxes_list in boxes_by_class.items():
                with torch.no_grad():
                    # Build the instruction using the same detect template as inference
                    detect_template = model.config.tokenizer.templates["detect"]
                    object_tokens = model.tokenizer.encode(" " + class_name).ids
                    instruction_token_ids = (
                        detect_template["prefix"]
                        + object_tokens
                        + detect_template["suffix"]
                    )
                    instruction_tokens = torch.tensor(
                        [instruction_token_ids],
                        dtype=torch.long,
                        device=model.device,
                    )
                    instruction_emb = text_encoder(instruction_tokens, model.text)

                cs_emb = []
                cs_labels = []
                c_idx = []
                s_idx = []
                for bb in boxes_list:
                    # Move boxes to model device (already float32 from dataset)
                    bb = bb.to(device=model.device)

                    # Interpret bb as [x_min, y_min, w, h] and convert to center coords
                    x_min, y_min, w_box, h_box = bb
                    x_center = x_min + w_box / 2.0
                    y_center = y_min + h_box / 2.0
                    l_cs = len(cs_emb)
                    cs_emb.extend(
                        [
                            encode_coordinate(x_center.unsqueeze(0), model.region),
                            encode_coordinate(y_center.unsqueeze(0), model.region),
                            encode_size(bb[2:4], model.region),
                        ]
                    )
                    c_idx.extend([l_cs, l_cs + 1])
                    s_idx.append(l_cs + 2)

                    # Create coordinate bin labels using center coordinates
                    coord_labels = [
                        int(min(max(torch.round(p * 1023), 0), 1023).item())
                        for p in (x_center, y_center)
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
                c_idx = (
                    torch.tensor(c_idx, dtype=torch.long, device=model.device) + prefix
                )
                s_idx = (
                    torch.tensor(s_idx, dtype=torch.long, device=model.device) + prefix
                )

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
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss

            if total_loss is not None:
                total_loss.backward()

            if i % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr_val = lr_schedule(i / grad_accum_steps, total_steps, base_lr=lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_val

                current_step = i // grad_accum_steps
                loss_val = total_loss.item() if total_loss is not None else 0.0
                pbar.set_postfix({"step": current_step, "loss": loss_val})
                pbar.update(1)

                wandb.log(
                    {
                        "loss/train": loss_val,
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    },
                    step=current_step,
                )

                # Periodic validation
                if current_step % eval_interval == 0 and current_step > 0:
                    logging.info(f"Evaluating at step {current_step}")
                    validation_score = validate(
                        model,
                        val_dataset,
                        step=current_step,
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
                        step=current_step,
                    )

                    # Save best model
                    if validation_score["f1"] > best_validation_score:
                        best_validation_score = validation_score["f1"]
                        if use_lora:
                            # Save only LoRA parameters
                            save_file(
                                get_lora_state_dict(model),
                                f"moondream_lora_best_step_{current_step}.safetensors",
                            )
                            logging.info(
                                f"Saved best LoRA adapter with F1: {round(best_validation_score, 4)}"
                            )
                        else:
                            save_file(
                                model.state_dict(),
                                f"moondream_best_step_{current_step}.safetensors",
                            )
                            logging.info(
                                f"Saved best model with F1: {round(best_validation_score, 4)}"
                            )
    wandb.finish()

    # Replace with your desired output location.
    if use_lora:
        # Save only LoRA parameters
        save_file(
            get_lora_state_dict(model),
            "moondream_lora_finetune.safetensors",
        )
        logging.info("Saved final LoRA adapter to moondream_lora_finetune.safetensors")
    else:
        save_file(
            model.state_dict(),
            "moondream_finetune.safetensors",
        )
        logging.info("Saved final model to moondream_finetune.safetensors")


if __name__ == "__main__":
    """
    Run with Fire CLI. Examples:
        python sft_trainer.py
        python sft_trainer.py --lr=1e-3 --epochs=20
        python sft_trainer.py --use_lora=False
    """
    fire.Fire(main)
