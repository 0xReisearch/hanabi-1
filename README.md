# Hanabi-1: Financial Market Transformer

A robust transformer-based model for predicting financial market movements using price and sentiment data.

## Overview

Hanabi-1 implements a specialized transformer architecture that predicts:

1. **Direction**: Whether the price will move up or down (binary classification)
2. **Volatility**: Expected price variation in the future hour
3. **Price Change**: Magnitude of price movement in percentage
4. **Spread Estimate**: Estimated spread based on volume patterns

The model uses two configurable approaches:
- 4-hour window → 1-hour prediction
- 12-hour window → 1-hour prediction

## Features

- **Advanced Transformer Architecture**: Multi-head self-attention with temporal feature differentiation
- **Robust Training**: BatchNorm, Xavier initialization, and LeakyReLU for stable learning
- **Balanced Prediction**: Specialized design to avoid extreme prediction biases
- **Dynamic Evaluation**: Automatic threshold calibration with confidence metrics
- **Ensemble Capability**: Support for model ensembles to improve prediction robustness

## Data Structure

### Required Data Files

1. **Hourly Price Data**:
   - Located in `hourly_data.csv` 
   - Contains 90 days of hourly data with:
     - Timestamp, Open, High, Low, Close, Price, Volume, MarketCap

2. **Fear and Greed Index Data**:
   - Located in `fear_greed_data/fear_greed_index_enhanced.csv`
   - Contains daily fear and greed index data with:
     - date, fng_value, fng_classification, fng_7d_ma, fng_30d_ma
     - fng_momentum metrics and normalized values

See `example_data/` directory for sample data files with the correct format and structure. The `example_data/README.md` contains detailed specifications on the required data format.

## Project Structure

- **Core Files**:
  - `data_preprocessor.py`: Data loading, preprocessing, and sequence creation
  - `transformer_model.py`: Transformer architecture with specialized heads
  - `train_model.py`: Training pipeline with configurable parameters
  - `predict.py`: Prediction generation and performance evaluation
  - `evaluate_ensemble.py`: Comprehensive evaluation and ensemble support

- **Training Scripts**:
  - `robust_model.sh`: Main training script for both 4h and 12h models
  - `balance_test.sh`: Test script for balance penalty mechanism
  - `seed_test.sh`: Test script for seed-based training

- **Generated Directories**:
  - `trained_models/`: Saved model checkpoints
  - `logs/`: Training and prediction logs
  - `evaluation/`: Evaluation metrics and visualizations
  - `predictions/`: Stored prediction outputs

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

For full Hanabi-1 training with robust architecture:

```bash
./robust_model.sh
```

For custom training configuration:

```bash
python train_model.py \
    --window_size 4 \
    --horizon 1 \
    --hidden_dim 384 \
    --transformer_layers 8 \
    --num_heads 8 \
    --dropout 0.25 \
    --learning_rate 0.00008 \
    --direction_threshold 0.5 \
    --min_price_change 0.005
```

### Prediction

Generate predictions with a trained model:

```bash
python predict.py \
    --model_path trained_models/financial_model_w4_h1_robust.pt \
    --window_size 4 \
    --calibrate_threshold \
    --visualize
```

### Evaluation

Evaluate model performance with rich metrics:

```bash
python evaluate_ensemble.py \
    --model_path trained_models/financial_model_w4_h1_robust.pt \
    --window_size 4
```

## Hanabi-1 Architecture Details

### Core Components

- **Input Processing**: 
  - Feature normalization and sequence creation
  - Positional encoding to preserve temporal information

- **Transformer Backbone**:
  - 8-layer transformer encoder (default)
  - 8 attention heads with 384-dimensional hidden states
  - Temporal feature extraction (last, average, attention-weighted)

- **Prediction Pathways**:
  - Direction (classification): BatchNorm + LeakyReLU for robust learning
  - Volatility (regression): Specialized for pricing variation
  - Price Change (regression): Calibrated for movement estimation
  - Spread (regression): Volume-based spread approximation

### Training Innovations

- **Balanced Learning**:
  - Validation scoring with penalties for extreme predictions
  - Xavier initialization for better weight distribution
  - Focal Loss with gamma=1.5 for handling subtle signals

- **Optimization**:
  - OneCycleLR scheduler with warmup and annealing
  - AdamW optimizer with 8e-5 learning rate
  - Gradient clipping to prevent exploding gradients

## Key Parameters

```
HIDDEN_DIM=384         # Size of hidden dimension
TRANSFORMER_LAYERS=8   # Number of transformer layers
NUM_HEADS=8            # Number of attention heads
DROPOUT=0.25           # Dropout rate for regularization
LEARNING_RATE=0.00008  # Learning rate for optimizer
WEIGHT_DECAY=0.001     # Weight decay for regularization
DIRECTION_WEIGHT=1.0   # Weight for direction loss
FOCAL_GAMMA=1.5        # Focal loss gamma parameter
MIN_PRICE_CHANGE=0.005 # Threshold for hourly price movements
```

## Performance and Visualization

The evaluation system provides:

- **Direction Metrics**: Accuracy, F1, precision, recall by confidence level
- **Regression Metrics**: MAE for volatility, price change, and spread
- **Visual Analysis**: 
  - Time series plots with correct/incorrect predictions
  - Confusion matrices for classification performance
  - Confidence-accuracy curves for reliability assessment
