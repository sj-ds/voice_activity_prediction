import pickle
import torch
import torch.nn as nn

# Model
class VAPModel(nn.Module):
    def __init__(self, input_dim=40, lstm_hidden_dim=256, num_heads=8,
                 transformer_layers=4, lstm_layers=2, output_dim=1):
        super(VAPModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=transformer_layers
        )
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return torch.sigmoid(x)

    @staticmethod
    def load_model_pickle(path="vap_model_6.pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        model.eval()
        return model