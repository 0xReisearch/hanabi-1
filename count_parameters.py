import torch
from transformer_model import FinancialTransformerModel

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_details(model):
    """Print detailed parameter information for each component of the model."""
    total_params = 0
    details = {}
    
    # Input embedding
    input_params = sum(p.numel() for p in model.input_embedding.parameters() if p.requires_grad)
    details["Input Embedding"] = input_params
    total_params += input_params
    
    # Positional encoding (typically not trainable)
    pos_params = sum(p.numel() for p in model.positional_encoding.parameters() if p.requires_grad)
    if pos_params > 0:
        details["Positional Encoding"] = pos_params
        total_params += pos_params
    
    # Transformer encoder layers
    encoder_params = sum(p.numel() for p in model.transformer_encoder.parameters() if p.requires_grad)
    details["Transformer Encoder"] = encoder_params
    total_params += encoder_params
    
    # Feature normalization
    feat_norm_params = sum(p.numel() for p in model.feature_norm.parameters() if p.requires_grad)
    details["Feature Normalization"] = feat_norm_params
    total_params += feat_norm_params
    
    # Direction pathway
    direction_params = sum(p.numel() for p in model.direction_pathway.parameters() if p.requires_grad)
    details["Direction Pathway"] = direction_params
    total_params += direction_params
    
    # Volatility pathway
    volatility_params = sum(p.numel() for p in model.volatility_pathway.parameters() if p.requires_grad)
    details["Volatility Pathway"] = volatility_params
    total_params += volatility_params
    
    # Price change pathway
    price_change_params = sum(p.numel() for p in model.price_change_pathway.parameters() if p.requires_grad)
    details["Price Change Pathway"] = price_change_params
    total_params += price_change_params
    
    # Spread pathway
    spread_params = sum(p.numel() for p in model.spread_pathway.parameters() if p.requires_grad)
    details["Spread Pathway"] = spread_params
    total_params += spread_params
    
    # Feature attention
    attention_params = sum(p.numel() for p in model.feature_attention.parameters() if p.requires_grad)
    details["Feature Attention"] = attention_params
    total_params += attention_params
    
    # Print details
    print("=== Parameter Details ===")
    for component, params in details.items():
        print(f"{component}: {params:,} parameters ({params/total_params*100:.2f}%)")
    
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    
    return total_params

if __name__ == "__main__":
    # Initialize model with default parameters
    model = FinancialTransformerModel(
        input_dim=24,         # 24 features based on our data
        hidden_dim=384,       # Default hidden dimension
        num_heads=8,          # Default number of attention heads
        transformer_layers=8, # Default number of transformer layers
        dropout=0.25          # Default dropout rate
    )
    
    # Count and print parameter details
    print_parameter_details(model)
    
    # Print model architecture summary
    print("\n=== Model Architecture ===")
    print(f"Input dimension: 24 features")
    print(f"Hidden dimension: 384")
    print(f"Transformer layers: 8")
    print(f"Attention heads: 8")
    print(f"Attention dimension per head: {384 // 8}")
    
    # Also try calculating with different configurations for comparison
    print("\n=== Comparing Different Configurations ===")
    configs = [
        {"hidden_dim": 192, "transformer_layers": 4, "num_heads": 4},
        {"hidden_dim": 384, "transformer_layers": 8, "num_heads": 8},
        {"hidden_dim": 512, "transformer_layers": 12, "num_heads": 8}
    ]
    
    for config in configs:
        model = FinancialTransformerModel(
            input_dim=24,
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            transformer_layers=config["transformer_layers"]
        )
        params = count_parameters(model)
        print(f"Configuration: hidden_dim={config['hidden_dim']}, "
              f"transformer_layers={config['transformer_layers']}, "
              f"num_heads={config['num_heads']}")
        print(f"Total parameters: {params:,}")
        print("")