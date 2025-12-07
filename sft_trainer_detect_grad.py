"""
Teacher-forced region fine-tuning that follows the generative detection path.

Instead of supervising the region head on hidden states from a synthetic sequence
(`_produce_hidden`), this trainer:
- Uses `encode_image_grad` + `_prefill_prompt_grad` to set up caches exactly
  like `detect` / `detect_grad`.
- Steps the decoder with teacher-forced ground-truth centers and sizes,
  and applies cross-entropy on the coordinate / size logits at the same
  points where inference uses them.

This is designed for Moondream 2 and the basketball detection dataset, and
supports LoRA-only fine-tuning via the same adapters as `sft_trainer.py`.
"""

import math
import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from safetensors.torch import load_file, save_file
import wandb

from sft_trainer import (
    LoRALinear,
    inject_lora_into_model,
    get_lora_state_dict,
    BasketballDetection,
    validate,
    validate_with_gt,
)

from moondream2.moondream import MoondreamModel, MoondreamConfig
from moondream2.moondream_functions import encode_image_grad, _prefill_prompt_grad
from moondream2.region import (
    decode_coordinate,
    decode_size,
    encode_coordinate,
    encode_size,
)


# Hyperparameters (tuned for stability; feel free to adjust)
LR = 1e-5
EPOCHS = 50
GRAD_ACCUM_STEPS = 1
VALIDATION_SAMPLES = 250
EVAL_INTERVAL = 1
OVERFIT_BATCH_SIZE: Optional[int] = 1  # Set >0 to overfit on a tiny subset

# LoRA configuration
USE_LORA = True
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["qkv", "proj", "fc1", "fc2"]

MODEL_PATH = "moondream2/model.safetensors"

device = "cuda" if torch.cuda.is_available() else "mps"


def lr_schedule(step: int, max_steps: int) -> float:
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


def coord_to_bin(center: float, n_bins: int = 1024) -> int:
    """Map normalized coord in [0,1] to discrete bin [0, n_bins-1]."""
    return int(min(max(round(center * (n_bins - 1)), 0), n_bins - 1))


def size_to_bin(s: float, n_bins: int = 1024) -> int:
    """
    Map normalized size to log-scale bin as in the region head:
    bin = (log2(size) + 10.0) / 10.0 * 1023.0, clamped to [0, 1023].
    """
    s_clamped = max(float(s), 1.0 / 1024.0)
    s_log2 = math.log2(s_clamped)
    mapped = (s_log2 + 10.0) / 10.0 * (n_bins - 1)
    b = int(round(mapped))
    return int(min(max(b, 0), n_bins - 1))


def bin_to_size(bin_idx: int, n_bins: int = 1024) -> float:
    """Inverse mapping from bin index back to size value."""
    return float(2.0 ** ((bin_idx / (n_bins - 1.0)) * 10.0 - 10.0))


def teacher_forced_region_loss(
    model: MoondreamModel,
    image,
    class_name: str,
    boxes: torch.Tensor,
    max_objects: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute region loss by following the same decode path as `detect` but with
    teacher-forced ground-truth centers and sizes.

    Args:
        model: MoondreamModel (Moondream 2).
        image: PIL.Image for the sample.
        class_name: Object string used for the detect prompt.
        boxes: Tensor of shape (N, 4) with [x_min, y_min, w, h] in [0, 1].
        max_objects: Optional cap on number of GT boxes to supervise.
    """
    if boxes.numel() == 0:
        return torch.zeros([], device=model.device)

    # Use the same detect template as inference
    detect_template = model.config.tokenizer.templates["detect"]
    object_tokens = model.tokenizer.encode(" " + class_name).ids
    instruction_token_ids = (
        detect_template["prefix"] + object_tokens + detect_template["suffix"]
    )

    # Encode image and prefill caches with BOS + image
    with torch.no_grad():
        encoded_image = encode_image_grad(model, image, settings=None)
    model.load_encoded_image(encoded_image)
    pos = encoded_image.pos

    # Prefill detect prompt (no teacher forcing yet)
    prompt_tokens = torch.tensor(
        [instruction_token_ids],
        dtype=torch.long,
        device=model.device,
    )
    logits_BV, hidden_BC, next_token, pos = _prefill_prompt_grad(
        model,
        prompt_tokens,
        pos,
        temperature=0.0,
        top_p=0.0,
        spatial_refs=None,
        attn_mask=None,
        lora=None,
    )
    # Hidden state corresponding to the last prompt token
    hidden = hidden_BC[:, -1:, :]

    # Attention mask used for subsequent one-token decode steps
    mask = torch.zeros(1, 1, 2048, device=model.device, dtype=torch.bool)
    mask[:, :, :pos] = 1

    if max_objects is None:
        max_objects = boxes.size(0)
    n_objects = min(boxes.size(0), max_objects)

    total_loss = torch.zeros([], device=model.device)
    n_terms = 0

    for obj_idx in range(n_objects):
        bb = boxes[obj_idx].to(model.device)  # [x_min, y_min, w, h]
        x_min, y_min, w_box, h_box = bb

        # Convert to center coordinates
        x_center = torch.clamp(x_min + w_box / 2.0, 0.0, 1.0)
        y_center = torch.clamp(y_min + h_box / 2.0, 0.0, 1.0)

        # ----- X coordinate step -----
        x_logits = decode_coordinate(hidden, model.region)  # [1, 1, 1024]
        x_bin = coord_to_bin(float(x_center))
        x_target = torch.tensor([x_bin], dtype=torch.long, device=model.device)
        loss_x = F.cross_entropy(
            x_logits.view(-1, x_logits.size(-1)),
            x_target,
        )
        total_loss = total_loss + loss_x
        n_terms += 1

        # Teacher-force x embedding and advance decoder (y will see this)
        # Shape here should match Moondream's own `_generate_points_grad`:
        # x_center.unsqueeze(-1) -> (1, 1, 1), then encode_coordinate -> (1, 1, dim)
        x_center_tensor = (
            x_center.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(dtype=x_logits.dtype, device=model.device)
        )
        next_emb = encode_coordinate(x_center_tensor, model.region)
        mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        _, hidden = model._decode_one_tok(next_emb, mask, pos_ids, lora=None)
        pos += 1

        # ----- Y coordinate step -----
        y_logits = decode_coordinate(hidden, model.region)
        y_bin = coord_to_bin(float(y_center))
        y_target = torch.tensor([y_bin], dtype=torch.long, device=model.device)
        loss_y = F.cross_entropy(
            y_logits.view(-1, y_logits.size(-1)),
            y_target,
        )
        total_loss = total_loss + loss_y
        n_terms += 1

        # Teacher-force y embedding and advance decoder (size will see this)
        y_center_tensor = (
            y_center.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(dtype=y_logits.dtype, device=model.device)
        )
        next_emb = encode_coordinate(y_center_tensor, model.region)
        mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        _, hidden = model._decode_one_tok(next_emb, mask, pos_ids, lora=None)
        pos += 1

        # ----- Size step (log-scale bins for w, h) -----
        size_logits = decode_size(hidden, model.region)  # (2, 1024)
        w_bin = size_to_bin(float(w_box))
        h_bin = size_to_bin(float(h_box))
        w_target = torch.tensor([w_bin], dtype=torch.long, device=model.device)
        h_target = torch.tensor([h_bin], dtype=torch.long, device=model.device)

        loss_w = F.cross_entropy(size_logits[0].unsqueeze(0), w_target)
        loss_h = F.cross_entropy(size_logits[1].unsqueeze(0), h_target)
        total_loss = total_loss + loss_w + loss_h
        n_terms += 2

        # Teacher-force size embedding and take one LM step to start next object
        w_val = bin_to_size(w_bin)
        h_val = bin_to_size(h_bin)
        size_tensor = torch.tensor(
            [w_val, h_val],
            device=model.device,
            dtype=size_logits.dtype,
        )
        next_emb = encode_size(size_tensor, model.region).unsqueeze(0).unsqueeze(0)
        mask[:, :, pos] = 1
        pos_ids = torch.tensor([pos], device=model.device, dtype=torch.long)
        _, hidden = model._decode_one_tok(next_emb, mask, pos_ids, lora=None)
        pos += 1

    if n_terms == 0:
        return torch.zeros([], device=model.device)
    return total_loss / n_terms


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    os.makedirs("predictions", exist_ok=True)

    wandb.init(
        project="moondream-basketball-ft-detect-grad",
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
            "VALIDATION_SAMPLES": VALIDATION_SAMPLES,
            "EVAL_INTERVAL": EVAL_INTERVAL,
            "OVERFIT_BATCH_SIZE": OVERFIT_BATCH_SIZE,
            "USE_LORA": USE_LORA,
            "LORA_RANK": LORA_RANK if USE_LORA else None,
            "LORA_ALPHA": LORA_ALPHA if USE_LORA else None,
            "LORA_DROPOUT": LORA_DROPOUT if USE_LORA else None,
            "dataset": "basketball-player-detection",
            "md_version": "2",
            "trainer": "detect_grad_teacher_forced",
        },
    )

    # Build and load Moondream 2
    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    state_dict = load_file(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.to(device)

    # Ensure all buffers are on the right device
    for _, buffer in model.named_buffers():
        buffer.data = buffer.data.to(device)

    # Apply LoRA if enabled
    if USE_LORA:
        logging.info("Applying LoRA adapters to text model...")
        inject_lora_into_model(
            model,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
        )

        # Freeze all base parameters, then unfreeze only LoRA weights
        for param in model.parameters():
            param.requires_grad = False
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable LoRA parameters: {trainable_params:,}")
        logging.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        optimizer = AdamW(
            [{"params": [p for p in model.parameters() if p.requires_grad]}],
            lr=LR,
        )
    else:
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Total trainable parameters (full fine-tuning): {total_params:,}")
        optimizer = AdamW(model.parameters(), lr=LR)

    # Datasets (basketball player detection)
    dataset = BasketballDetection(split="train")
    val_dataset = BasketballDetection(split="val")

    # Overfit subset handling
    if OVERFIT_BATCH_SIZE is not None and OVERFIT_BATCH_SIZE > 0:
        logging.info(
            f"Overfitting mode: using first {OVERFIT_BATCH_SIZE} training samples for both train and val"
        )
        train_full = BasketballDetection(split="train")
        train_full.dataset.image_ids = train_full.dataset.image_ids[:OVERFIT_BATCH_SIZE]
        dataset = train_full

        val_full = BasketballDetection(split="train")
        val_full.dataset.image_ids = val_full.dataset.image_ids[:OVERFIT_BATCH_SIZE]
        val_dataset = val_full

    logging.info(f"Train dataset size: {len(dataset)}")
    logging.info(f"Val dataset size: {len(val_dataset)}")

    # Initial validation
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

    model.train()
    i = 0
    for epoch in range(EPOCHS):
        for sample in dataset:
            i += 1

            boxes_by_class = {}
            for box, cls in zip(sample["boxes"], sample["class_names"]):
                boxes_by_class.setdefault(cls, []).append(box)

            total_loss = None
            for class_name, boxes_list in boxes_by_class.items():
                boxes_tensor = torch.stack(
                    [bb.to(dtype=torch.float32) for bb in boxes_list]
                )
                loss = teacher_forced_region_loss(
                    model,
                    sample["image"],
                    class_name,
                    boxes_tensor,
                )
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss

            if total_loss is not None:
                (total_loss / max(len(boxes_by_class), 1)).backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr_val = lr_schedule(i // GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_val

                current_step = i // GRAD_ACCUM_STEPS
                loss_val = (
                    total_loss.item() / max(len(boxes_by_class), 1)
                    if total_loss is not None
                    else 0.0
                )
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

                    # Save best model (LoRA-only if enabled)
                    if validation_score["f1"] > best_validation_score:
                        best_validation_score = validation_score["f1"]
                        if USE_LORA:
                            save_file(
                                get_lora_state_dict(model),
                                f"moondream_lora_best_step_{current_step}_detect_grad.safetensors",
                            )
                            logging.info(
                                f"Saved best LoRA adapter (detect_grad) with F1: {round(best_validation_score, 4)}"
                            )
                        else:
                            save_file(
                                model.state_dict(),
                                f"moondream_best_step_{current_step}_detect_grad.safetensors",
                            )
                            logging.info(
                                f"Saved best full model (detect_grad) with F1: {round(best_validation_score, 4)}"
                            )

    wandb.finish()

    # Final checkpoint
    if USE_LORA:
        save_file(
            get_lora_state_dict(model),
            "moondream_lora_finetune_detect_grad.safetensors",
        )
        logging.info(
            "Saved final LoRA adapter to moondream_lora_finetune_detect_grad.safetensors"
        )
    else:
        save_file(
            model.state_dict(),
            "moondream_finetune_detect_grad.safetensors",
        )
        logging.info("Saved final model to moondream_finetune_detect_grad.safetensors")


if __name__ == "__main__":
    main()
