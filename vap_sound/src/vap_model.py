## Model definition

import torch
import torch.nn as nn

# Model
class VAPModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, lstm_hidden_dim=256, num_heads=8, num_layers=4, output_dim=1):
        super(VAPModel, self).__init__()
        
        # Transformer layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        
        # LSTM Layer 
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        
        # Fully connected layer to produce final output
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.transformer(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return torch.sigmoid(x)