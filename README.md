# moondream-grpo

A place to finetune

To get started, install the dependencies:

```bash
pip install -r requirements.txt
```

Then, run the training script:

```bash
python grpo_trainer.py
```

This will apply RL finetuning to the model on the RefCOCO dataset. You'll need to first download the base model from HuggingFace.

```bash
wget https://huggingface.co/vikhyatk/moondream2/resolve/main/model.safetensors
```

And once you move that model to the appropriate path, you can run the training script.

Several other hyperparameters are available at the top of the `grpo_trainer.py` file.

# References

This work is based on the following repositories and tutorials:

-   [Moondream](https://huggingface.co/vikhyatk/moondream2)
-   [Moondream Region Finetuning](https://github.com/vikhyat/moondream/blob/main/moondream/finetune/finetune_region.py)
-   [GRPO Code](https://www.youtube.com/watch?v=yGkJj_4bjpE)
