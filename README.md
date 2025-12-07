# moondream-grpo

A place to finetune

To get started, install the dependencies:

```bash
pip install -r requirements.txt
```

Then, run the GRPO training script:

```bash
python grpo_trainer.py
```

This will apply RL finetuning to the model on the RefCOCO dataset. You'll need to first download the base model from HuggingFace.

```bash
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/model.safetensors
```

And once you move that model to the appropriate path, you can run the training script.

Several other hyperparameters are available at the top of the `grpo_trainer.py` file.

## Loading LoRA adapters trained with `sft_trainer_detect_grad.py`

The teacher-forced SFT trainer in `sft_trainer_detect_grad.py` can fine-tune Moondream-2 using LoRA adapters implemented in `trainer_helpers.py` (via the `LoRALinear` wrapper), **not** the built-in `moondream2/lora.py` / `variant_state_dict` mechanism.

When you run `sft_trainer_detect_grad.py` with `USE_LORA = True`, it will save LoRA-only checkpoints such as:

-   **Best checkpoint**: `moondream_lora_best_step_{STEP}_detect_grad.safetensors`
-   **Final checkpoint**: `moondream_lora_finetune_detect_grad.safetensors`

To load these adapters for inference:

1. **Load the base Moondream-2 model and weights** (same as training):

```python
import torch
from safetensors.torch import load_file

from moondream2.moondream import MoondreamModel, MoondreamConfig
from trainer_helpers import inject_lora_into_model, LoRALinear

device = "cuda" if torch.cuda.is_available() else "mps"

model = MoondreamModel(config=MoondreamConfig(), setup_caches=True)
base_state = load_file("moondream2/model.safetensors")
model.load_state_dict(base_state)
model.to(device)
```

2. **Inject LoRA layers with the same config used during training**:

```python
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["qkv", "proj", "fc1", "fc2"]

inject_lora_into_model(
    model,
    rank=LORA_RANK,
    alpha=LORA_ALPHA,
    dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
)
```

3. **Load the LoRA adapter weights** saved by the trainer:

```python
from safetensors.torch import load_file

# For example, load the final detect_grad LoRA checkpoint
lora_state = load_file("moondream_lora_finetune_detect_grad.safetensors")

# This will populate only the LoRA parameters (e.g. *.lora_A, *.lora_B)
missing, unexpected = model.load_state_dict(lora_state, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
```

At this point, `model` is the base Moondream-2 model plus your trained LoRA adapters, and you can call the usual detection API (e.g. via `moondream2.moondream_functions.detect(model, image, object_str, settings, temperature=0.0)` or `model.detect(...)`) for inference. Because these adapters live inside `LoRALinear` wrappers from `trainer_helpers.py`, they **cannot** be loaded via `moondream2.lora.variant_state_dict` or Hugging Face variant IDs; they must be loaded directly into the Python model as shown above.

# References

This work is based on the following repositories and tutorials:

-   [Moondream](https://huggingface.co/vikhyatk/moondream2)
-   [Moondream Region Finetuning](https://github.com/vikhyat/moondream/blob/main/moondream/finetune/finetune_region.py)
-   [GRPO Code](https://www.youtube.com/watch?v=yGkJj_4bjpE)

Next steps:

rename repo to reflect sft and grpo

clean up and simplify the teacher forcing trainer

run full training run on multiple categories

share with discord
