import argparse
import torch
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from datetime import datetime
import glob

from data_preprocessor import FinancialDataPreprocessor
from transformer_model import FinancialTransformerModel

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("financial_model_evaluation")

def load_model(model_path, device):
    """Load a trained model from a checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get hyperparameters
    hyperparams = checkpoint['hyperparams']
    logger.info(f"Model hyperparameters: {hyperparams}")
    
    # Initialize model with same hyperparameters
    model = FinancialTransformerModel(
        input_dim=hyperparams['input_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        transformer_layers=hyperparams.get('transformer_layers', 8),
        num_heads=hyperparams.get('num_heads', 8),
        dropout=hyperparams.get('dropout', 0.25),
        direction_threshold=hyperparams.get('direction_threshold', 0.5)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, checkpoint

def create_evaluation_dataset(hourly_data_path, fear_greed_data_path, window_size=4, horizon=1):
    """Create a dataset for model evaluation."""
    # Initialize preprocessor
    preprocessor = FinancialDataPreprocessor(hourly_data_path, fear_greed_data_path)
    
    # Get entire dataset to evaluate throughout the full period
    data_dict = preprocessor.prepare_data(window_size, horizon, train_ratio=1.0)
    
    # Create tensor dataset for evaluation
    X = torch.tensor(data_dict['X_train'], dtype=torch.float32)
    
    # Split y into its components (for comparison)
    y = data_dict['y_train']
    y_direction = torch.tensor(y[:, 0], dtype=torch.float32).unsqueeze(1)
    y_volatility = torch.tensor(y[:, 1], dtype=torch.float32).unsqueeze(1)
    y_price_change = torch.tensor(y[:, 2], dtype=torch.float32).unsqueeze(1)
    y_spread = torch.tensor(y[:, 3], dtype=torch.float32).unsqueeze(1)
    
    # Get timestamps
    merged_data = preprocessor._merge_hourly_and_fear_greed()
    timestamps = list(merged_data.index)[window_size:window_size+len(X)]
    
    if len(timestamps) != len(X):
        logger.warning(f"Timestamp length {len(timestamps)} doesn't match data length {len(X)}. Truncating...")
        timestamps = timestamps[:len(X)]
    
    # Package everything into a dataset
    return {
        'X': X,
        'y_direction': y_direction,
        'y_volatility': y_volatility,
        'y_price_change': y_price_change,
        'y_spread': y_spread,
        'timestamps': timestamps,
        'preprocessor': preprocessor
    }

def evaluate_single_model(model, dataset, device, batch_size=64, direction_threshold=0.5):
    """Evaluate a single model using the provided dataset."""
    model.eval()
    
    # Unpack dataset
    X = dataset['X']
    y_direction = dataset['y_direction']
    y_volatility = dataset['y_volatility']
    y_price_change = dataset['y_price_change']
    y_spread = dataset['y_spread']
    timestamps = dataset['timestamps']
    
    all_preds = {
        'timestamp': [],
        'direction_prob': [],
        'direction_pred': [],
        'direction_actual': [],
        'volatility_pred': [],
        'volatility_actual': [],
        'price_change_pred': [],
        'price_change_actual': [],
        'spread_pred': [],
        'spread_actual': [],
        'confidence': []
    }
    
    # Make predictions in batches
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            batch_outputs = model(batch_X)
            
            # Extract predictions
            direction_probs = batch_outputs['direction_prob'].cpu().numpy()
            direction_preds = (direction_probs >= direction_threshold).astype(np.float32)
            volatility_preds = batch_outputs['volatility'].cpu().numpy()
            price_change_preds = batch_outputs['price_change'].cpu().numpy()
            spread_preds = batch_outputs['spread'].cpu().numpy()
            
            # Extract actual values
            batch_indices = range(i, min(i + batch_size, len(X)))
            batch_timestamps = [timestamps[j] for j in batch_indices]
            batch_y_direction = y_direction[batch_indices].cpu().numpy()
            batch_y_volatility = y_volatility[batch_indices].cpu().numpy()
            batch_y_price_change = y_price_change[batch_indices].cpu().numpy()
            batch_y_spread = y_spread[batch_indices].cpu().numpy()
            
            # Calculate confidence scores
            confidence_scores = np.abs(direction_probs - 0.5) * 2
            
            # Store predictions and actual values
            all_preds['timestamp'].extend(batch_timestamps)
            all_preds['direction_prob'].extend(direction_probs.flatten())
            all_preds['direction_pred'].extend(direction_preds.flatten())
            all_preds['direction_actual'].extend(batch_y_direction.flatten())
            all_preds['volatility_pred'].extend(volatility_preds.flatten())
            all_preds['volatility_actual'].extend(batch_y_volatility.flatten())
            all_preds['price_change_pred'].extend(price_change_preds.flatten())
            all_preds['price_change_actual'].extend(batch_y_price_change.flatten())
            all_preds['spread_pred'].extend(spread_preds.flatten())
            all_preds['spread_actual'].extend(batch_y_spread.flatten())
            all_preds['confidence'].extend(confidence_scores.flatten())
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(all_preds)
    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    results_df.set_index('timestamp', inplace=True)
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def evaluate_ensemble(models, dataset, device, batch_size=64, direction_threshold=0.5):
    """Evaluate an ensemble of models using the provided dataset."""
    # Unpack dataset
    X = dataset['X']
    y_direction = dataset['y_direction']
    y_volatility = dataset['y_volatility']
    y_price_change = dataset['y_price_change']
    y_spread = dataset['y_spread']
    timestamps = dataset['timestamps']
    
    all_preds = {
        'timestamp': [],
        'direction_prob': [],
        'direction_pred': [],
        'direction_actual': [],
        'volatility_pred': [],
        'volatility_actual': [],
        'price_change_pred': [],
        'price_change_actual': [],
        'spread_pred': [],
        'spread_actual': [],
        'confidence': []
    }
    
    # Make predictions in batches
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            
            # Initialize batch predictions
            batch_direction_probs = []
            batch_volatility_preds = []
            batch_price_change_preds = []
            batch_spread_preds = []
            
            # Get predictions from each model
            for model in models:
                model.eval()
                batch_outputs = model(batch_X)
                
                batch_direction_probs.append(batch_outputs['direction_prob'].cpu().numpy())
                batch_volatility_preds.append(batch_outputs['volatility'].cpu().numpy())
                batch_price_change_preds.append(batch_outputs['price_change'].cpu().numpy())
                batch_spread_preds.append(batch_outputs['spread'].cpu().numpy())
            
            # Average predictions across models
            direction_probs = np.mean(batch_direction_probs, axis=0)
            direction_preds = (direction_probs >= direction_threshold).astype(np.float32)
            volatility_preds = np.mean(batch_volatility_preds, axis=0)
            price_change_preds = np.mean(batch_price_change_preds, axis=0)
            spread_preds = np.mean(batch_spread_preds, axis=0)
            
            # Extract actual values
            batch_indices = range(i, min(i + batch_size, len(X)))
            batch_timestamps = [timestamps[j] for j in batch_indices]
            batch_y_direction = y_direction[batch_indices].cpu().numpy()
            batch_y_volatility = y_volatility[batch_indices].cpu().numpy()
            batch_y_price_change = y_price_change[batch_indices].cpu().numpy()
            batch_y_spread = y_spread[batch_indices].cpu().numpy()
            
            # Calculate ensemble confidence scores
            # Confidence is higher when models agree (lower standard deviation)
            direction_std = np.std(batch_direction_probs, axis=0)
            base_confidence = np.abs(direction_probs - 0.5) * 2
            # Adjust confidence based on model agreement
            confidence_scores = base_confidence * (1 - direction_std)
            
            # Store predictions and actual values
            all_preds['timestamp'].extend(batch_timestamps)
            all_preds['direction_prob'].extend(direction_probs.flatten())
            all_preds['direction_pred'].extend(direction_preds.flatten())
            all_preds['direction_actual'].extend(batch_y_direction.flatten())
            all_preds['volatility_pred'].extend(volatility_preds.flatten())
            all_preds['volatility_actual'].extend(batch_y_volatility.flatten())
            all_preds['price_change_pred'].extend(price_change_preds.flatten())
            all_preds['price_change_actual'].extend(batch_y_price_change.flatten())
            all_preds['spread_pred'].extend(spread_preds.flatten())
            all_preds['spread_actual'].extend(batch_y_spread.flatten())
            all_preds['confidence'].extend(confidence_scores.flatten())
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(all_preds)
    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    results_df.set_index('timestamp', inplace=True)
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    return results_df, metrics

def calculate_metrics(results_df):
    """Calculate performance metrics from prediction results."""
    # Direction prediction metrics
    y_pred = results_df['direction_pred'].values
    y_true = results_df['direction_actual'].values
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Regression metrics for volatility, price change, and spread
    volatility_mae = np.mean(np.abs(results_df['volatility_pred'] - results_df['volatility_actual']))
    price_change_mae = np.mean(np.abs(results_df['price_change_pred'] - results_df['price_change_actual']))
    spread_mae = np.mean(np.abs(results_df['spread_pred'] - results_df['spread_actual']))
    
    # Calculate metrics by confidence level
    confidence_bins = [0, 0.25, 0.5, 0.75, 1.0]
    results_df['confidence_bin'] = pd.cut(results_df['confidence'], bins=confidence_bins, labels=['Very Low', 'Low', 'Medium', 'High'])
    
    conf_metrics = {}
    for bin_name in ['Very Low', 'Low', 'Medium', 'High']:
        bin_df = results_df[results_df['confidence_bin'] == bin_name]
        if len(bin_df) > 0:
            bin_accuracy = accuracy_score(bin_df['direction_actual'], bin_df['direction_pred'])
            bin_count = len(bin_df)
            bin_percent = len(bin_df) / len(results_df) * 100
            conf_metrics[bin_name] = {
                'accuracy': bin_accuracy,
                'count': bin_count,
                'percent': bin_percent
            }
    
    # Compile metrics
    metrics = {
        'direction': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'positive_rate': float(np.mean(y_pred))
        },
        'regression': {
            'volatility_mae': volatility_mae,
            'price_change_mae': price_change_mae,
            'spread_mae': spread_mae
        },
        'confidence': conf_metrics
    }
    
    return metrics

def visualize_results(results_df, metrics, output_path, title_prefix=''):
    """Generate visualizations for model evaluation results."""
    os.makedirs(output_path, exist_ok=True)
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot direction prediction probabilities over time
    plt.figure(figsize=(15, 8))
    plt.plot(results_df.index, results_df['direction_prob'], label='Direction Probability')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    
    # Color points by correct/incorrect prediction
    correct_mask = results_df['direction_pred'] == results_df['direction_actual']
    plt.scatter(
        results_df[correct_mask].index, 
        results_df[correct_mask]['direction_prob'],
        color='green', alpha=0.5, label='Correct Prediction'
    )
    plt.scatter(
        results_df[~correct_mask].index, 
        results_df[~correct_mask]['direction_prob'],
        color='red', alpha=0.5, label='Incorrect Prediction'
    )
    
    plt.title(f'{title_prefix}Direction Prediction Probabilities')
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_path, f'direction_probs_{timestamp}.png'))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(results_df['direction_actual'], results_df['direction_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{title_prefix}Confusion Matrix (Accuracy: {metrics["direction"]["accuracy"]:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_path, f'confusion_matrix_{timestamp}.png'))
    
    # Plot confidence vs. accuracy
    plt.figure(figsize=(10, 6))
    conf_acc = [(k, v['accuracy'], v['count']) for k, v in metrics['confidence'].items()]
    conf_bins, accuracies, counts = zip(*sorted(conf_acc, key=lambda x: ['Very Low', 'Low', 'Medium', 'High'].index(x[0])))
    
    bars = plt.bar(conf_bins, accuracies, color='skyblue')
    plt.title(f'{title_prefix}Prediction Accuracy by Confidence Level')
    plt.xlabel('Confidence Level')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.02, 
            f'n={count}', 
            ha='center'
        )
    
    plt.savefig(os.path.join(output_path, f'confidence_accuracy_{timestamp}.png'))
    
    # Plot volatility and price change predictions vs actual
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(results_df.index, results_df['volatility_actual'], label='Actual', color='blue')
    plt.plot(results_df.index, results_df['volatility_pred'], label='Predicted', color='red', alpha=0.7)
    plt.title(f'{title_prefix}Volatility (MAE: {metrics["regression"]["volatility_mae"]:.4f})')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(results_df.index, results_df['price_change_actual'], label='Actual', color='blue')
    plt.plot(results_df.index, results_df['price_change_pred'], label='Predicted', color='red', alpha=0.7)
    plt.title(f'{title_prefix}Price Change (MAE: {metrics["regression"]["price_change_mae"]:.4f})')
    plt.xlabel('Date')
    plt.ylabel('Price Change')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'regression_metrics_{timestamp}.png'))
    
    logger.info(f"Visualizations saved to {output_path}")

def save_metrics(metrics, output_path, filename):
    """Save metrics to a JSON file, converting numpy types to Python types."""
    os.makedirs(output_path, exist_ok=True)
    
    # Convert numpy types to standard Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_serializable(obj.tolist())
        else:
            return obj
    
    # Convert metrics to JSON-serializable types
    serializable_metrics = convert_to_serializable(metrics)
    
    # Save to file
    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {os.path.join(output_path, filename)}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Financial Transformer Models")
    
    parser.add_argument("--model_path", type=str, help="Path to a trained model")
    parser.add_argument("--model_dir", type=str, help="Directory containing multiple trained models (for ensemble)")
    parser.add_argument("--hourly_data", type=str, default="/root/hlmmnn/hourly_data.csv", 
                        help="Path to hourly financial data CSV")
    parser.add_argument("--fear_greed_data", type=str, default="/root/hlmmnn/fear_greed_data/fear_greed_index_enhanced.csv", 
                        help="Path to fear and greed index enhanced CSV")
    parser.add_argument("--window_size", type=int, default=4, 
                        help="Number of hours of historical data to use")
    parser.add_argument("--horizon", type=int, default=1, 
                        help="Number of hours ahead to predict")
    parser.add_argument("--output_dir", type=str, default="/root/hlmmnn/new_model/evaluation", 
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], 
                        help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Check if at least one of model_path or model_dir is provided
    if not args.model_path and not args.model_dir:
        parser.error("At least one of --model_path or --model_dir must be provided")
    
    # Use CPU if CUDA is not available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, using CPU instead")
        args.device = "cpu"
    
    device = torch.device(args.device)
    
    # Create evaluation dataset
    logger.info(f"Creating evaluation dataset with window_size={args.window_size}, horizon={args.horizon}")
    dataset = create_evaluation_dataset(
        args.hourly_data,
        args.fear_greed_data,
        args.window_size,
        args.horizon
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evaluate single model if provided
    if args.model_path:
        logger.info(f"Evaluating single model: {args.model_path}")
        model, checkpoint = load_model(args.model_path, device)
        
        # Get model name from path
        model_name = os.path.basename(args.model_path).replace('.pt', '')
        
        results_df, metrics = evaluate_single_model(
            model, dataset, device, 
            direction_threshold=model.direction_threshold
        )
        
        # Save results
        results_df.to_csv(os.path.join(args.output_dir, f"{model_name}_results_{timestamp}.csv"))
        save_metrics(metrics, args.output_dir, f"{model_name}_metrics_{timestamp}.json")
        
        # Visualize results
        visualize_results(
            results_df, metrics, args.output_dir,
            title_prefix=f"{model_name} - "
        )
        
        # Print metrics summary
        logger.info(f"=== {model_name} Evaluation Results ===")
        logger.info(f"Direction Accuracy: {metrics['direction']['accuracy']:.4f}")
        logger.info(f"Direction F1 Score: {metrics['direction']['f1']:.4f}")
        logger.info(f"Direction Precision: {metrics['direction']['precision']:.4f}")
        logger.info(f"Direction Recall: {metrics['direction']['recall']:.4f}")
        logger.info(f"Prediction distribution: {metrics['direction']['positive_rate']*100:.1f}% positive")
        logger.info(f"Volatility MAE: {metrics['regression']['volatility_mae']:.6f}")
        logger.info(f"Price Change MAE: {metrics['regression']['price_change_mae']:.6f}")
        logger.info(f"Spread MAE: {metrics['regression']['spread_mae']:.6f}")
    
    # Evaluate ensemble if model directory is provided
    if args.model_dir:
        logger.info(f"Evaluating ensemble of models from: {args.model_dir}")
        
        # Find model files for the specified window size
        model_pattern = os.path.join(args.model_dir, f"financial_model_w{args.window_size}_*.pt")
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            logger.error(f"No model files found matching pattern: {model_pattern}")
            return
        
        logger.info(f"Found {len(model_files)} models for ensemble")
        
        # Load all models
        models = []
        for model_path in model_files:
            logger.info(f"Loading model: {model_path}")
            model, _ = load_model(model_path, device)
            models.append(model)
        
        # Evaluate ensemble
        results_df, metrics = evaluate_ensemble(
            models, dataset, device, 
            direction_threshold=0.5  # Use standard threshold for ensemble
        )
        
        # Save results
        output_prefix = f"ensemble_w{args.window_size}_h{args.horizon}"
        results_df.to_csv(os.path.join(args.output_dir, f"{output_prefix}_results_{timestamp}.csv"))
        save_metrics(metrics, args.output_dir, f"{output_prefix}_metrics_{timestamp}.json")
        
        # Visualize results
        visualize_results(
            results_df, metrics, args.output_dir,
            title_prefix=f"Ensemble ({len(models)} models) - "
        )
        
        # Print metrics summary
        logger.info(f"=== Ensemble Evaluation Results ({len(models)} models) ===")
        logger.info(f"Direction Accuracy: {metrics['direction']['accuracy']:.4f}")
        logger.info(f"Direction F1 Score: {metrics['direction']['f1']:.4f}")
        logger.info(f"Direction Precision: {metrics['direction']['precision']:.4f}")
        logger.info(f"Direction Recall: {metrics['direction']['recall']:.4f}")
        logger.info(f"Prediction distribution: {metrics['direction']['positive_rate']*100:.1f}% positive")
        logger.info(f"Volatility MAE: {metrics['regression']['volatility_mae']:.6f}")
        logger.info(f"Price Change MAE: {metrics['regression']['price_change_mae']:.6f}")
        logger.info(f"Spread MAE: {metrics['regression']['spread_mae']:.6f}")
    
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()