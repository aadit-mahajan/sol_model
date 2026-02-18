import smi_ted
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

TED_MODEL_NAME = "bisectgroup/materials-smi-ted-fork"
_tokenizer = AutoTokenizer.from_pretrained(TED_MODEL_NAME)
_model = AutoModel.from_pretrained(TED_MODEL_NAME)
_model.smi_ted.tokenizer = _tokenizer
_model.smi_ted.set_padding_idx_from_tokenizer()
_model.eval()

def encode(smiles_list, batch_size=128):
    embeddings = []

    if smiles_list is None or len(smiles_list) == 0:
        return np.array([])
    
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            out = _model.encode(batch)
            if isinstance(out, torch.Tensor):
                if out.dim() == 3:
                    cls_vec = out[:, 0, :]
                elif out.dim() == 2:
                    cls_vec = out if out.size(0) == len(batch) else out[0, :].unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected TED shape: {out.shape}")

                embeddings.append(cls_vec.cpu())
            else:
                raise ValueError("TED returned non-tensor output.")

    return torch.cat(embeddings, dim=0).numpy()
    
def decode(embeddings):
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    embeddings = embeddings.unsqueeze(0)

    _model.smi_ted.tokenizer = _tokenizer
    _model.smi_ted.set_padding_idx_from_tokenizer()
    with torch.no_grad():
        decoded_smiles = _model.decode(embeddings)
    return decoded_smiles

class SolubilityModel(nn.Module):
    def __init__(self, enc_out_dim, hidden_dim=128):
        super().__init__()
        self.cation_encoder = nn.Linear(enc_out_dim, hidden_dim)
        self.anion_encoder = nn.Linear(enc_out_dim, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Split into cation and anion
        cation_emb = self.cation_encoder(x[:, :x.shape[1]//2])
        anion_emb = self.anion_encoder(x[:, x.shape[1]//2:])
        interaction = cation_emb * anion_emb # Element-wise interaction
        addition = cation_emb + anion_emb # Element-wise addition

        # Combine
        combined = torch.cat([cation_emb.squeeze(1), anion_emb.squeeze(1), interaction.squeeze(1), addition.squeeze(1)], dim=1)
        
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu().numpy()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item() * X_batch.size(0)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_total_loss += loss.item() * X_batch.size(0)

        train_loss = train_total_loss / len(train_loader.dataset)
        val_loss = val_total_loss / len(val_loader.dataset)

        epoch_losses = {
            "train_loss": train_loss,
            "val_loss": val_loss
        }
        losses.append(epoch_losses)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return losses, model

def create_zero_shot_split(data, test_ratio=0.1, val_ratio= 0.1,seed=None):
    np.random.seed(seed)
    unique_cations = data['Cation'].unique()
    unique_anions = data['Anion'].unique()
    
    test_cations = np.random.choice(unique_cations, size=int(len(unique_cations)*test_ratio), replace = False)
    test_anions = np.random.choice(unique_anions, size=int(len(unique_anions)*test_ratio), replace=False)

    # remove from dataset
    unique_cations = unique_cations[~np.isin(unique_cations, test_cations)]
    unique_anions = unique_anions[~np.isin(unique_anions, test_anions)]

    val_cations = np.random.choice(unique_cations, size=int(len(unique_cations)*val_ratio), replace=False)
    val_anions = np.random.choice(unique_anions, size=int(len(unique_anions)*val_ratio), replace=False)

    # remove again
    unique_cations = unique_cations[~np.isin(unique_cations, val_cations)]
    unique_anions = unique_anions[~np.isin(unique_anions, val_anions)]
    train_cations = unique_cations
    train_anions = unique_anions

    train_data = data[data['Cation'].isin(train_cations) & data['Anion'].isin(train_anions)]
    test_data = data[data['Cation'].isin(test_cations) & data['Anion'].isin(test_anions)]
    val_data = data[data['Cation'].isin(val_cations) & data['Anion'].isin(val_anions)]

    train_c_features = encode(train_data['Cation_SMILES'].tolist())
    train_a_features = encode(train_data['Anion_SMILES'].tolist())
    test_c_features = encode(test_data['Cation_SMILES'].tolist())
    test_a_features = encode(test_data['Anion_SMILES'].tolist())
    val_c_features = encode(val_data['Cation_SMILES'].tolist())
    val_a_features = encode(val_data['Anion_SMILES'].tolist())

    X_train = np.hstack((train_c_features, train_a_features))
    y_train = train_data['Capacity'].apply(lambda x: np.log10(np.clip(x, 1e-8, None))).values

    X_test = np.hstack((test_c_features, test_a_features))
    y_test = test_data['Capacity'].apply(lambda x: np.log10(np.clip(x, 1e-8, None))).values

    X_val = np.hstack((val_c_features, val_a_features))
    y_val = val_data['Capacity'].apply(lambda x: np.log10(np.clip(x, 1e-8, None))).values
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, val_loader

# Ensemble predictions
def ensemble_predict(models, X):
    preds = [model.predict(X) for model in models]
    return np.mean(preds, axis=0)

def main():
    data = pd.read_csv("sol_data_organic.csv")
    data.head()

    # checkpoint file path
    output_dir = "checkpoints"
    losses_dir = "losses"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(losses_dir, exist_ok=True)
    input_dim = 768  # encoder output dim 
    num_epochs = 10
    device = 'cuda:0'
    # Train multiple models with different splits
    
    models = []
    for seed in range(5):
        print('-'*50)
        train, test, val = create_zero_shot_split(data, seed=seed)
        print(f"Train size: {len(train.dataset)}, Val size: {len(val.dataset)}, Test size: {len(test.dataset)}")
        print("training model number ", seed)
        model = SolubilityModel(input_dim, hidden_dim=32)
        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        losses, model = train_model(model, train, val, criterion, optimizer, num_epochs, device)
        models.append(model)

        plt.figure(figsize=(8, 6))
        plt.plot([loss['train_loss'] for loss in losses], label='Train Loss')
        plt.plot([loss['val_loss'] for loss in losses], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss (Seed {seed})')
        plt.legend()
        plt.savefig(os.path.join(losses_dir, f"loss_curve_seed_{seed}.png"))
        plt.close()

    _, test, _ = create_zero_shot_split(data, seed=42)
    # prediction on test set
    X_test = torch.tensor(test.dataset.tensors[0].numpy(), dtype=torch.float32).to(device)
    y_test = test.dataset.tensors[1].to(device)
    predictions = ensemble_predict(models, X_test)

    r2 = r2_score(y_test.cpu().numpy(), predictions)
    mse = mean_squared_error(y_test.cpu().numpy(), predictions)
    print(f"Test R2: {r2:.4f}, Test MSE: {mse:.4f}")

    y_test = y_test.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Log Capacity')
    plt.ylabel('Predicted Log Capacity')
    plt.title('Neural Network Model Predictions (Test Set)')
    plt.legend()
    plt.savefig('nn_predictions.png')
    plt.close()

    for idx, model in enumerate(models):
        model_path = os.path.join(output_dir, f"solubility_model_seed_{idx}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model checkpoint to {model_path}")

if __name__ == "__main__":
    main()