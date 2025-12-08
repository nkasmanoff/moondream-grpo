"""
Utility script to load a fine-tuned Moondream model for inference.

This script demonstrates how to load either:
1. A full fine-tuned model
2. A LoRA adapter + region model on top of the base model

Usage Examples:
---------------
Load full fine-tuned model:
    python load_finetuned_model.py --checkpoint=model_artifacts/moondream_best_step_100.safetensors

Load LoRA + region adapter:
    python load_finetuned_model.py --checkpoint=model_artifacts/moondream_lora_best_step_100.safetensors --use_lora=True
"""

import torch
from safetensors.torch import load_file
import fire

from moondream2.moondream import MoondreamModel, MoondreamConfig
from trainer_helpers import inject_lora_into_model


def load_finetuned_model(
    checkpoint: str = "model_artifacts/moondream_lora_finetune.safetensors",
    base_model_path: str = "moondream2/model.safetensors",
    use_lora: bool = True,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    lora_target_modules: list = None,
    device: str = None,
):
    """
    Load a fine-tuned Moondream model for inference.

    Args:
        checkpoint: Path to the fine-tuned model checkpoint in model_artifacts/
        base_model_path: Path to the base model weights (needed for LoRA)
        use_lora: Whether the checkpoint is a LoRA adapter or full model
        lora_rank: Rank of LoRA matrices (must match training)
        lora_alpha: Scaling factor for LoRA (must match training)
        lora_dropout: Dropout for LoRA layers (must match training)
        lora_target_modules: Which layers have LoRA applied (must match training)
        device: Device to load model on (default: cuda if available, else mps)

    Returns:
        Loaded MoondreamModel ready for inference
    """
    if lora_target_modules is None:
        lora_target_modules = ["qkv", "proj", "fc1", "fc2"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps"

    # Build and load base Moondream model
    print(f"Loading base model from {base_model_path}...")
    model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
    state_dict = load_file(base_model_path)
    model.load_state_dict(state_dict)
    model.to(device)

    # Ensure all buffers are on the right device
    for _, buffer in model.named_buffers():
        buffer.data = buffer.data.to(device)

    if use_lora:
        print(f"Applying LoRA structure (rank={lora_rank}, alpha={lora_alpha})...")
        inject_lora_into_model(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )

        # Load fine-tuned LoRA + region weights
        print(f"Loading fine-tuned LoRA + region weights from {checkpoint}...")
        finetuned_state_dict = load_file(checkpoint)
        model.load_state_dict(finetuned_state_dict, strict=False)
        print("Successfully loaded LoRA adapter and region model weights!")
    else:
        # Load full fine-tuned model
        print(f"Loading full fine-tuned model from {checkpoint}...")
        finetuned_state_dict = load_file(checkpoint)
        model.load_state_dict(finetuned_state_dict)
        print("Successfully loaded full fine-tuned model!")

    model.eval()
    print(f"Model ready for inference on {device}!")
    return model


def main(
    checkpoint: str = "model_artifacts/moondream_lora_finetune.safetensors",
    base_model_path: str = "moondream2/model.safetensors",
    use_lora: bool = True,
):
    """
    Main function to demonstrate loading a fine-tuned model.

    Args:
        checkpoint: Path to the fine-tuned model checkpoint
        base_model_path: Path to the base model weights
        use_lora: Whether the checkpoint is a LoRA adapter
    """
    model = load_finetuned_model(
        checkpoint=checkpoint,
        base_model_path=base_model_path,
        use_lora=use_lora,
    )

    print("\n" + "=" * 60)
    print("Model loaded successfully!")
    print("You can now use the model for inference.")
    print("\nExample usage:")
    print("  detections = model.detect(image, 'basketball player')")
    print("=" * 60)

    return model


if __name__ == "__main__":
    fire.Fire(main)
