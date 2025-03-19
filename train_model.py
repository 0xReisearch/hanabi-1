import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import os
import time
from datetime import datetime

from data_preprocessor import FinancialDataPreprocessor
from transformer_model import FinancialTransformerModel, FocalLoss

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("financial_model")

def train_model(
    hourly_data_path,
    fear_greed_data_path,
    window_size=4,
    horizon=1,
    batch_size=64,
    hidden_dim=384,
    transformer_layers=8,
    num_heads=8,
    dropout=0.25,
    learning_rate=0.00005,
    weight_decay=0.002,
    direction_weight=1.0,
    focal_gamma=2.0,
    epochs=400,
    patience=40,
    min_price_change=0.005,
    direction_threshold=0.5,
    save_path='models',
    seed=None,
    model_suffix=""
):
    """
    Train the financial transformer model.
    
    Args:
        hourly_data_path (str): Path to the hourly financial data CSV.
        fear_greed_data_path (str): Path to the fear and greed index enhanced CSV.
        window_size (int): Number of hours of historical data to use.
        horizon (int): Number of hours ahead to predict.
        batch_size (int): Batch size for training.
        hidden_dim (int): Hidden dimension size for the model.
        transformer_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay for optimizer.
        direction_weight (float): Weight for direction loss.
        focal_gamma (float): Gamma parameter for focal loss.
        epochs (int): Maximum number of epochs to train.
        patience (int): Early stopping patience.
        min_price_change (float): Minimum price change for direction prediction.
        direction_threshold (float): Threshold for direction prediction.
        save_path (str): Path to save the model.
    """
    # Create save path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Set random seed if specified for reproducibility
    if seed is not None:
        logger.info(f"Setting random seed to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Enable cuDNN benchmark mode for faster training if not using fixed seed
    if device.type == 'cuda' and seed is None:
        torch.backends.cudnn.benchmark = True
    
    # Initialize data preprocessor
    logger.info("Loading and preprocessing data...")
    preprocessor = FinancialDataPreprocessor(hourly_data_path, fear_greed_data_path)
    
    # Get data loaders
    train_loader, val_loader = preprocessor.get_dataloaders(
        window_size=window_size,
        horizon=horizon,
        batch_size=batch_size
    )
    
    # Print shape of first batch from train_loader
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        logger.info(f"Input batch shape: {inputs.shape}")
        break
    
    # Initialize model
    logger.info("Initializing model...")
    input_dim = inputs.shape[2]  # Get actual input dimension from data
    logger.info(f"Input dimension (from data): {input_dim}")
    
    model = FinancialTransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        transformer_layers=transformer_layers,
        dropout=dropout,
        direction_threshold=direction_threshold
    )
    
    # Move model to device
    model = model.to(device)
    
    # Initialize loss functions
    direction_loss_fn = FocalLoss(gamma=focal_gamma)
    regression_loss_fn = nn.MSELoss()
    
    # Loss weights for different tasks
    loss_weights = {
        'direction': direction_weight,
        'volatility': 0.5,  # Reduced weight since it's normalized
        'price_change': 1.0,  # Full weight since it's normalized
        'spread': 0.3  # Reduced weight since it's normalized
    }
    
    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warmup for 10% of training
        anneal_strategy='cos'
    )
    
    # Training loop
    logger.info("Starting training loop...")
    total_time = 0
    best_val_score = float('-inf')
    best_epoch = 0
    patience_counter = 0
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_f1_scores = []
    val_accuracies = []
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            direction_targets = targets['direction'].to(device)
            volatility_targets = targets['volatility'].to(device)
            price_change_targets = targets['price_change'].to(device)
            spread_targets = targets['spread'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate losses with proper weights
            direction_loss = loss_weights['direction'] * direction_loss_fn(outputs['direction_logits'], direction_targets)
            volatility_loss = loss_weights['volatility'] * regression_loss_fn(outputs['volatility'], volatility_targets)
            price_change_loss = loss_weights['price_change'] * regression_loss_fn(outputs['price_change'], price_change_targets)
            spread_loss = loss_weights['spread'] * regression_loss_fn(outputs['spread'], spread_targets)
            
            # Total loss
            total_loss = direction_loss + volatility_loss + price_change_loss + spread_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {total_loss.item():.6f}")
                logger.info(f"Direction: {direction_loss.item():.6f} | Volatility: {volatility_loss.item():.6f} | Price: {price_change_loss.item():.6f} | Spread: {spread_loss.item():.6f}")
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        val_loss, direction_metrics = validate_model(
            model, val_loader, direction_loss_fn, regression_loss_fn, 
            device, direction_weight, min_price_change, direction_threshold
        )
        val_losses.append(val_loss)
        val_f1_scores.append(direction_metrics['f1'])
        val_accuracies.append(direction_metrics['accuracy'])
        
        # Calculate custom validation score for early stopping
        # Balance between F1 score, accuracy, and precision-recall balance
        if max(direction_metrics['precision'], direction_metrics['recall']) > 0:
            pr_balance = min(direction_metrics['precision'], direction_metrics['recall']) / max(direction_metrics['precision'], direction_metrics['recall'])
        else:
            pr_balance = 0  # Avoid division by zero
            
        # Add a strong penalty for extreme predictions (all one class)
        balance_penalty = 0
        if direction_metrics['precision'] == 0 or direction_metrics['recall'] == 0:
            # Heavy penalty for predicting all one class
            balance_penalty = 0.5
        elif direction_metrics['precision'] < 0.2 or direction_metrics['recall'] < 0.2:
            # Moderate penalty for extreme imbalance
            balance_penalty = 0.3
            
        # Final score prioritizes balanced predictions with good F1 and accuracy
        val_score = (
            direction_metrics['f1'] + 
            0.5 * direction_metrics['accuracy'] + 
            0.5 * pr_balance - 
            0.1 * val_loss -
            balance_penalty
        )
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | "
                   f"Direction F1: {direction_metrics['f1']:.4f} | Accuracy: {direction_metrics['accuracy']:.4f} | "
                   f"Precision: {direction_metrics['precision']:.4f} | Recall: {direction_metrics['recall']:.4f} | "
                   f"PR Balance: {pr_balance:.4f} | Val Score: {val_score:.4f}")
        
        # Check if model improved
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model with optional suffix
            model_filename = f"financial_model_w{window_size}_h{horizon}{model_suffix}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_score': val_score,
                'hyperparams': {
                    'hidden_dim': hidden_dim,
                    'transformer_layers': transformer_layers,
                    'num_heads': num_heads,
                    'dropout': dropout,
                    'direction_threshold': direction_threshold,
                    'input_dim': input_dim,
                    'seed': seed
                },
                'metrics': direction_metrics
            }, os.path.join(save_path, model_filename))
            
            logger.info(f"Model improved, saved checkpoint to {model_filename}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
            
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time/60:.2f} minutes")
    logger.info(f"Best model at epoch {best_epoch} with validation score {best_val_score:.4f}")
    
    # Plot training metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(val_f1_scores, label='F1 Score')
    plt.title('F1 Score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    
    plt.subplot(2, 2, 3)
    plt.plot(val_accuracies, label='Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(2, 2, 4)
    plt.plot([val_f1_scores[i] + 0.5 * val_accuracies[i] - 0.1 * val_losses[i] for i in range(len(val_f1_scores))], label='Val Score')
    plt.title('Validation Score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'training_metrics_w{window_size}_h{horizon}{model_suffix}.png'))


def validate_model(
    model, val_loader, direction_loss_fn, regression_loss_fn, 
    device, direction_weight, min_price_change, direction_threshold
):
    """
    Validate the model.
    
    Args:
        model: The model to validate.
        val_loader: Validation data loader.
        direction_loss_fn: Direction loss function.
        regression_loss_fn: Regression loss function.
        device: Device to use for validation.
        direction_weight: Weight for direction loss.
        min_price_change: Minimum price change for direction evaluation.
        direction_threshold: Threshold for direction prediction.
        
    Returns:
        tuple: (validation_loss, direction_metrics_dict)
    """
    model.eval()
    total_loss = 0
    all_direction_preds = []
    all_direction_targets = []
    
    # Loss weights for different tasks
    loss_weights = {
        'direction': direction_weight,
        'volatility': 0.5,
        'price_change': 1.0,
        'spread': 0.3
    }
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            direction_targets = targets['direction'].to(device)
            volatility_targets = targets['volatility'].to(device)
            price_change_targets = targets['price_change'].to(device)
            spread_targets = targets['spread'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate losses with proper weights
            direction_loss = loss_weights['direction'] * direction_loss_fn(outputs['direction_logits'], direction_targets)
            volatility_loss = loss_weights['volatility'] * regression_loss_fn(outputs['volatility'], volatility_targets)
            price_change_loss = loss_weights['price_change'] * regression_loss_fn(outputs['price_change'], price_change_targets)
            spread_loss = loss_weights['spread'] * regression_loss_fn(outputs['spread'], spread_targets)
            
            # Total loss
            total_loss += (
                direction_loss.item() + 
                volatility_loss.item() + 
                price_change_loss.item() + 
                spread_loss.item()
            )
            
            # Get predictions
            direction_probs = outputs['direction_prob'].cpu().numpy()
            direction_preds = (direction_probs >= direction_threshold).astype(np.float32)
            
            # Store predictions and targets for metrics calculation
            all_direction_preds.extend(direction_preds)
            all_direction_targets.extend(direction_targets.cpu().numpy())
    
    # Calculate average loss
    total_loss /= len(val_loader)
    
    # Convert to numpy arrays
    all_direction_preds = np.array(all_direction_preds)
    all_direction_targets = np.array(all_direction_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_direction_targets, all_direction_preds)
    precision = precision_score(all_direction_targets, all_direction_preds, zero_division=0)
    recall = recall_score(all_direction_targets, all_direction_preds, zero_division=0)
    f1 = f1_score(all_direction_targets, all_direction_preds, zero_division=0)
    
    # Calculate prediction distribution
    if len(all_direction_preds) > 0:
        positive_pct = np.mean(all_direction_preds) * 100
    else:
        positive_pct = 0.0
    
    direction_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'positive_pct': positive_pct
    }
    
    return total_loss, direction_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Financial Transformer Model")
    
    parser.add_argument("--hourly_data", type=str, default="/root/hlmmnn/hourly_data.csv", 
                        help="Path to hourly financial data CSV")
    parser.add_argument("--fear_greed_data", type=str, default="/root/hlmmnn/fear_greed_data/fear_greed_index_enhanced.csv", 
                        help="Path to fear and greed index enhanced CSV")
    parser.add_argument("--window_size", type=int, default=4, 
                        help="Number of hours of historical data to use")
    parser.add_argument("--horizon", type=int, default=1, 
                        help="Number of hours ahead to predict")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--hidden_dim", type=int, default=384, 
                        help="Hidden dimension size for the model")
    parser.add_argument("--transformer_layers", type=int, default=8, 
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, 
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.25, 
                        help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=0.00005, 
                        help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.002, 
                        help="Weight decay for optimizer")
    parser.add_argument("--direction_weight", type=float, default=1.0, 
                        help="Weight for direction loss")
    parser.add_argument("--focal_gamma", type=float, default=1.5, 
                        help="Gamma parameter for focal loss")
    parser.add_argument("--epochs", type=int, default=150, 
                        help="Maximum number of epochs to train")
    parser.add_argument("--patience", type=int, default=25, 
                        help="Early stopping patience")
    parser.add_argument("--min_price_change", type=float, default=0.005, 
                        help="Minimum price change for direction prediction")
    parser.add_argument("--direction_threshold", type=float, default=0.5, 
                        help="Threshold for direction prediction")
    parser.add_argument("--save_path", type=str, default="/root/hlmmnn/new_model/trained_models", 
                        help="Path to save the model")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for reproducibility")
    parser.add_argument("--model_suffix", type=str, default="", 
                        help="Suffix to add to model filename")
    
    args = parser.parse_args()
    
    logger.info("Starting training with the following parameters:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Train the model with the specified parameters
    train_model(
        hourly_data_path=args.hourly_data,
        fear_greed_data_path=args.fear_greed_data,
        window_size=args.window_size,
        horizon=args.horizon,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        direction_weight=args.direction_weight,
        focal_gamma=args.focal_gamma,
        epochs=args.epochs,
        patience=args.patience,
        min_price_change=args.min_price_change,
        direction_threshold=args.direction_threshold,
        save_path=args.save_path,
        seed=args.seed,
        model_suffix=args.model_suffix
    )