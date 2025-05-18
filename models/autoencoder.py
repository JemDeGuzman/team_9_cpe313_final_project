import torch
import torch.nn as nn
import numpy as np

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_autoencoder_model(model_path, input_dim):
    model = SimpleAutoencoder(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_reconstruction_errors(model, data_np):
    with torch.no_grad():
        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        reconstructed = model(data_tensor)
        errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1).numpy()
    return errors

def detect_anomalies(model, data_np, threshold=None):
    errors = get_reconstruction_errors(model, data_np)

    if threshold is None:
        threshold = np.percentile(errors, 95)

    anomaly_labels = (errors > threshold).astype(int)
    return errors, anomaly_labels
