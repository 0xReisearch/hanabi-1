import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FinancialTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=384,
        num_heads=8,
        transformer_layers=8,
        dropout=0.25,
        direction_threshold=0.45
    ):
        super(FinancialTransformerModel, self).__init__()
        
        self.direction_threshold = direction_threshold
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        print(f"Model initialized with input_dim={input_dim}")
        
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_layers
        )
        
        self.feature_norm = nn.LayerNorm(hidden_dim)
        
        self.direction_pathway = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.volatility_pathway = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.price_change_pathway = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.spread_pathway = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        for m in self.direction_pathway.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.05, 0.05)
    
    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.feature_norm(x)
        
        last_hidden = x[:, -1]
        avg_hidden = torch.mean(x, dim=1)
        
        attn_weights = self.feature_attention(x).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)
        attn_hidden = torch.sum(x * attn_weights, dim=1)
        
        combined_features = torch.cat([last_hidden, avg_hidden, attn_hidden], dim=1)
        
        volatility_pred = self.volatility_pathway(combined_features)
        price_change_pred = self.price_change_pathway(combined_features)
        spread_pred = self.spread_pathway(combined_features)
        
        batch_size = combined_features.size(0)
        features = combined_features
        
        x = self.direction_pathway[0](features)
        x = self.direction_pathway[1](x)
        x = self.direction_pathway[2](x)
        x = self.direction_pathway[3](x)
        
        x = self.direction_pathway[4](x)
        x = self.direction_pathway[5](x)
        x = self.direction_pathway[6](x)
        x = self.direction_pathway[7](x)
        
        x = self.direction_pathway[8](x)
        x = self.direction_pathway[9](x)
        x = self.direction_pathway[10](x)
        x = self.direction_pathway[11](x)
        
        direction_logits = self.direction_pathway[12](x)
        direction_prob = torch.sigmoid(direction_logits)
        
        return {
            'direction_logits': direction_logits,
            'direction_prob': direction_prob,
            'volatility': volatility_pred,
            'price_change': price_change_pred,
            'spread': spread_pred
        }
    
    def predict_direction(self, direction_prob):
        return (direction_prob >= self.direction_threshold).float()
    
    def get_prediction_confidence(self, direction_prob):
        return torch.abs(direction_prob - self.direction_threshold) * 2


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss