#!/bin/bash

# Create necessary directories
mkdir -p ./trained_models
mkdir -p ./logs
mkdir -p ./evaluation

# Set training parameters with balanced models in mind
EPOCHS=150
PATIENCE=30
MIN_PRICE_CHANGE=0.005
DIRECTION_WEIGHT=1.0
RANDOM_SEED=42

echo "=========================================================="
echo "Hanabi-1 - Financial Market Transformer - ROBUST MODEL (12-Hour)"
echo "Using BatchNorm and Xavier initialization for balanced training"
echo "Using balanced validation scoring with penalties for extreme predictions"
echo "Using hourly data and fear/greed index with min_price_change=$MIN_PRICE_CHANGE"
echo "=========================================================="

# Train 12-hour window model
echo "Training 12-hour window model..."
python train_model.py \
    --window_size 12 \
    --horizon 1 \
    --hidden_dim 384 \
    --transformer_layers 8 \
    --num_heads 8 \
    --dropout 0.25 \
    --learning_rate 0.00008 \
    --weight_decay 0.001 \
    --direction_weight $DIRECTION_WEIGHT \
    --focal_gamma 1.5 \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --min_price_change $MIN_PRICE_CHANGE \
    --direction_threshold 0.5 \
    --seed $RANDOM_SEED \
    --model_suffix "_robust" \
    --save_path ./trained_models 2>&1 | tee ./logs/train_w12_robust.log

# Evaluate 12-hour window model
echo "Evaluating 12-hour window model..."
python evaluate_ensemble.py \
    --model_path ./trained_models/financial_model_w12_h1_robust.pt \
    --window_size 12 \
    --output_dir ./evaluation 2>&1 | tee ./logs/eval_w12_robust.log

echo "=========================================================="
echo "12-hour model training and evaluation complete!"
echo "Results available in:"
echo "- ./trained_models (model files)"
echo "- ./evaluation (evaluation results)"
echo "- ./logs (log files)"
echo "==========================================================" 