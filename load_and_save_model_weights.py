"""
Load Moondream2 model from HuggingFace and save weights to safetensors format.

This script downloads the moondream2 model from HuggingFace and saves the model
weights to a local safetensors file that can be used for training and inference.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from safetensors.torch import save_file

# Load the model from HuggingFace
hf_model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-06-21",
    trust_remote_code=True,
    device_map={"": "mps"},  # ...or 'mps', on Apple Silicon
)

# Setup model caches
hf_model.model._setup_caches()

# Save model weights to safetensors format
save_file(
    hf_model.model.state_dict(),
    "model.safetensors",
)

print("Model weights saved to model.safetensors")
# this safetensors is the one which will work.
