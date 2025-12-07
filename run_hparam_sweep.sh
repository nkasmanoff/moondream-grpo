#!/bin/bash

# Hyperparameter sweep script for sft_trainer.py
# Run configurations sequentially to test different hyperparameter combinations

# Set to exit on error
set -e

echo "Starting hyperparameter sweep..."
echo "=================================="

# Configuration 1: Default settings
echo ""
echo "Config 1: Default settings"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 2: Lower learning rate
echo ""
echo "Config 2: Lower learning rate (1e-5)"
python sft_trainer.py \
    --lr=1e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 3: Higher learning rate
echo ""
echo "Config 3: Higher learning rate (1e-4)"
python sft_trainer.py \
    --lr=1e-4 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 4: Very low learning rate
echo ""
echo "Config 4: Very low learning rate (5e-6)"
python sft_trainer.py \
    --lr=5e-6 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 5: Lower LoRA rank
echo ""
echo "Config 5: Lower LoRA rank (16)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=16 \
    --lora_alpha=32 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 6: Higher LoRA rank
echo ""
echo "Config 6: Higher LoRA rank (64)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 7: Very high LoRA rank
echo ""
echo "Config 7: Very high LoRA rank (128)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=128 \
    --lora_alpha=256 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 8: More epochs
echo ""
echo "Config 8: More epochs (10)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=10 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 9: Fewer epochs, higher LR
echo ""
echo "Config 9: Fewer epochs (3), higher LR (1e-4)"
python sft_trainer.py \
    --lr=1e-4 \
    --epochs=3 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 10: Lower gradient accumulation
echo ""
echo "Config 10: Lower gradient accumulation (32)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=32 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 11: Higher gradient accumulation
echo ""
echo "Config 11: Higher gradient accumulation (128)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=128 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 12: No LoRA dropout
echo ""
echo "Config 12: No LoRA dropout (0.0)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.0 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 13: Higher LoRA dropout
echo ""
echo "Config 13: Higher LoRA dropout (0.2)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.2 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 14: Lower LR + Higher rank
echo ""
echo "Config 14: Lower LR (1e-5) + Higher rank (64)"
python sft_trainer.py \
    --lr=1e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 15: Higher LR + Lower rank
echo ""
echo "Config 15: Higher LR (1e-4) + Lower rank (16)"
python sft_trainer.py \
    --lr=1e-4 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=16 \
    --lora_alpha=32 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 16: More frequent evaluation
echo ""
echo "Config 16: More frequent evaluation (eval_interval=2)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=2 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 17: Less frequent evaluation
echo ""
echo "Config 17: Less frequent evaluation (eval_interval=10)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=10 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 18: More validation samples
echo ""
echo "Config 18: More validation samples (500)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=500 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 19: Overfit mode (small batch)
echo ""
echo "Config 19: Overfit mode (batch_size=4)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=50 \
    --grad_accum_steps=4 \
    --validation_samples=250 \
    --eval_interval=4 \
    --overfit_batch_size=4 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 20: Full fine-tuning (no LoRA) - Very conservative
echo ""
echo "Config 20: Full fine-tuning (no LoRA) - Very conservative (lr=1e-6)"
python sft_trainer.py \
    --lr=1e-6 \
    --epochs=3 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 20a: Full fine-tuning - Conservative
echo ""
echo "Config 20a: Full fine-tuning (no LoRA) - Conservative (lr=5e-6)"
python sft_trainer.py \
    --lr=5e-6 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 20b: Full fine-tuning - Moderate
echo ""
echo "Config 20b: Full fine-tuning (no LoRA) - Moderate (lr=1e-5)"
python sft_trainer.py \
    --lr=1e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 20c: Full fine-tuning - Moderate with more epochs
echo ""
echo "Config 20c: Full fine-tuning (no LoRA) - Moderate LR, more epochs (lr=1e-5, epochs=10)"
python sft_trainer.py \
    --lr=1e-5 \
    --epochs=10 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 20d: Full fine-tuning - Lower grad accum
echo ""
echo "Config 20d: Full fine-tuning (no LoRA) - Lower grad accum (lr=1e-5, grad_accum=32)"
python sft_trainer.py \
    --lr=1e-5 \
    --epochs=5 \
    --grad_accum_steps=32 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 20e: Full fine-tuning - Higher grad accum
echo ""
echo "Config 20e: Full fine-tuning (no LoRA) - Higher grad accum (lr=5e-6, grad_accum=128)"
python sft_trainer.py \
    --lr=5e-6 \
    --epochs=5 \
    --grad_accum_steps=128 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 20f: Full fine-tuning - Extended training
echo ""
echo "Config 20f: Full fine-tuning (no LoRA) - Extended training (lr=5e-6, epochs=15)"
python sft_trainer.py \
    --lr=5e-6 \
    --epochs=15 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 20g: Full fine-tuning - Higher LR (risky)
echo ""
echo "Config 20g: Full fine-tuning (no LoRA) - Higher LR (lr=2e-5)"
python sft_trainer.py \
    --lr=2e-5 \
    --epochs=3 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 20h: Full fine-tuning - Overfit mode
echo ""
echo "Config 20h: Full fine-tuning (no LoRA) - Overfit mode (batch_size=4)"
python sft_trainer.py \
    --lr=1e-5 \
    --epochs=50 \
    --grad_accum_steps=4 \
    --validation_samples=250 \
    --eval_interval=4 \
    --overfit_batch_size=4 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 21: Balanced - medium LR, medium rank
echo ""
echo "Config 21: Balanced (lr=2e-5, rank=48)"
python sft_trainer.py \
    --lr=2e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=48 \
    --lora_alpha=96 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 22: Conservative - low LR, high rank
echo ""
echo "Config 22: Conservative (lr=5e-6, rank=64)"
python sft_trainer.py \
    --lr=5e-6 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 23: Aggressive - high LR, low rank
echo ""
echo "Config 23: Aggressive (lr=2e-4, rank=16)"
python sft_trainer.py \
    --lr=2e-4 \
    --epochs=3 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=16 \
    --lora_alpha=32 \
    --lora_dropout=0.15 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 24: Extended training
echo ""
echo "Config 24: Extended training (epochs=15)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=15 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 25: High capacity LoRA
echo ""
echo "Config 25: High capacity LoRA (rank=96)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=64 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=96 \
    --lora_alpha=192 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 26: Very high gradient accumulation (LoRA)
echo ""
echo "Config 26: Very high gradient accumulation (grad_accum=256)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=256 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 27: Very high gradient accumulation with lower LR (LoRA)
echo ""
echo "Config 27: Very high gradient accumulation, lower LR (grad_accum=256, lr=1e-5)"
python sft_trainer.py \
    --lr=1e-5 \
    --epochs=5 \
    --grad_accum_steps=256 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 28: Extremely high gradient accumulation (LoRA)
echo ""
echo "Config 28: Extremely high gradient accumulation (grad_accum=512)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=512 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 29: Extremely high gradient accumulation with lower LR (LoRA)
echo ""
echo "Config 29: Extremely high gradient accumulation, lower LR (grad_accum=512, lr=1e-5)"
python sft_trainer.py \
    --lr=1e-5 \
    --epochs=5 \
    --grad_accum_steps=512 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 30: Ultra high gradient accumulation (LoRA)
echo ""
echo "Config 30: Ultra high gradient accumulation (grad_accum=1024)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=1024 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=32 \
    --lora_alpha=64 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 31: Very high gradient accumulation (Full fine-tuning)
echo ""
echo "Config 31: Full fine-tuning - Very high gradient accumulation (grad_accum=256)"
python sft_trainer.py \
    --lr=5e-6 \
    --epochs=5 \
    --grad_accum_steps=256 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 32: Very high gradient accumulation with lower LR (Full fine-tuning)
echo ""
echo "Config 32: Full fine-tuning - Very high gradient accumulation, lower LR (grad_accum=256, lr=1e-6)"
python sft_trainer.py \
    --lr=1e-6 \
    --epochs=5 \
    --grad_accum_steps=256 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 33: Extremely high gradient accumulation (Full fine-tuning)
echo ""
echo "Config 33: Full fine-tuning - Extremely high gradient accumulation (grad_accum=512)"
python sft_trainer.py \
    --lr=5e-6 \
    --epochs=5 \
    --grad_accum_steps=512 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 34: Ultra high gradient accumulation (Full fine-tuning)
echo ""
echo "Config 34: Full fine-tuning - Ultra high gradient accumulation (grad_accum=1024)"
python sft_trainer.py \
    --lr=5e-6 \
    --epochs=5 \
    --grad_accum_steps=1024 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=False \
    --wandb_project=moondream-basketball-ft-sweep

# Configuration 35: High grad accum with high rank LoRA
echo ""
echo "Config 35: High gradient accumulation with high rank LoRA (grad_accum=256, rank=64)"
python sft_trainer.py \
    --lr=5e-5 \
    --epochs=5 \
    --grad_accum_steps=256 \
    --validation_samples=250 \
    --eval_interval=5 \
    --use_lora=True \
    --lora_rank=64 \
    --lora_alpha=128 \
    --lora_dropout=0.1 \
    --wandb_project=moondream-basketball-ft-sweep

echo ""
echo "=================================="
echo "Hyperparameter sweep completed!"
