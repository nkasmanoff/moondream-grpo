# Moondream Fine-tuning

A repository for fine-tuning Moondream vision-language models using supervised and reinforcement learning approaches.

These trainers are focused on improving the model's ability to detect and localize objects in images, but in the future we can add the other tasks Moondream can do.

### Warning:

This code works best for Moondream 2, and the teacher forced trainer (`sft_trainer_detect_grad.py`).

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Base Model(s)

Download the Moondream 2 base model from Hugging Face:

```bash
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/model.safetensors
mv model.safetensors moondream2/model.safetensors
```

For [Moondream 3](https://huggingface.co/moondream/moondream3-preview), do the same, but place the model at `models/model_md3.safetensors`.

### 3. Prepare Dataset

Any COCO style dataset will work. In this case I wanted to use something relatively difficult for the existing versions of Moondream so we could actually see some improvement. In this case I used the basketball player detection dataset made by RoboFlow. You can download that dataset [here](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true).

Place that (or any other COCO style dataset) in the `datasets/{dataset_name}/` directory.

## Trainers

### 1. SFT Trainer (`sft_trainer.py`)

Supervised fine-tuning with teacher-forced supervision on the region head using synthetic sequences and optional LoRA adapters.

**Run with default settings:**

```bash
python sft_trainer.py
```

**Run with custom parameters:**

```bash
python sft_trainer.py --lr=1e-3 --epochs=20 --use_lora=True --lora_rank=32
```

### 2. SFT Detect Grad Trainer (`sft_trainer_detect_grad.py`)

Teacher-forced region fine-tuning that follows the generative detection path exactly as used during inference for more aligned training.

**Run with default settings:**

```bash
python sft_trainer_detect_grad.py
```

**Run with custom parameters:**

```bash
python sft_trainer_detect_grad.py --lr=1e-5 --epochs=5 --use_lora=True --grad_accum_steps=16
```

### 3. GRPO Trainer (`grpo_trainer.py`)

Group Relative Policy Optimization (GRPO) trainer that uses reinforcement learning to fine-tune the region head by collecting rollouts and computing rewards based on detection quality.

**Run with default settings:**

```bash
python grpo_trainer.py
```

**Run with custom parameters:**

```bash
python grpo_trainer.py --learning_rate=5e-5 --batch_size=5 --num_rollouts=5 --num_epochs=3
```

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

## References

This work is based on the following repositories and tutorials:

-   [Moondream](https://huggingface.co/vikhyatk/moondream2)
-   [Moondream Region Finetuning](https://github.com/vikhyat/moondream/blob/main/moondream/finetune/finetune_region.py)
-   [GRPO Code](https://www.youtube.com/watch?v=yGkJj_4bjpE)
