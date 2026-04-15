import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from transformers import AutoTokenizer, AutoModel
import selfies as sf
import torch
import torch.nn.functional as F
from typing import Union
from tqdm import tqdm

_tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
_model = AutoModel.from_pretrained("ibm/materials.selfies-ted")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = _model.to(_device)

def smiles_to_selfies(smiles_list: list[str]) -> list[str]:
    """Convert a list of SMILES to tokenizer-ready SELFIES strings."""
    result = []
    for smi in smiles_list:
        sel = sf.encoder(smi)
        sel = sel.replace("][", "] [")  # space-separate tokens
        result.append(sel)
    return result

@torch.no_grad()
def encode(
    smiles: Union[str, list[str]],
    batch_size: int = 64,
    max_length: int = 128,
) -> torch.Tensor:
    """
    Encode SMILES string(s) into mean-pooled embeddings.

    Args:
        smiles:     A single SMILES string or a list of them.
        batch_size: Number of molecules to process per forward pass.
        max_length: Max token length (truncated/padded to this).

    Returns:
        embeddings: Tensor of shape (N, hidden_dim) on CPU.
    """
    if isinstance(smiles, str):
        smiles = [smiles]

    selfies_list = smiles_to_selfies(smiles)
    all_embeddings = []

    _model.eval()
    for i in tqdm(range(0, len(selfies_list), batch_size)):
        batch = selfies_list[i : i + batch_size]

        tokens = _tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = tokens["input_ids"].to(_device)
        attention_mask = tokens["attention_mask"].to(_device)

        outputs = _model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, D)

        # Mean pooling over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        sum_embeddings = torch.sum(hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask  # (B, D)

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)  # (N, D)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, num_experts)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class SolubilityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=5):
        super(SolubilityModel, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gating_weights = self.gating_network(x)  # (B, num_experts)
        expert_outputs = torch.stack([expert(x).squeeze(-1) for expert in self.experts], dim=1)  # (B, num_experts)
        output = torch.sum(gating_weights * expert_outputs, dim=1)  # (B,)
        return output

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item() * X_batch.size(0)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
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

def create_train_test_split(data, test_ratio = 0.1, seed = None):
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(42)

    unique_cations = data['Cation'].unique()
    unique_anions = data['Anion'].unique()
    
    num_test_cations = max(1, int(len(unique_cations) * test_ratio))
    num_test_anions = max(1, int(len(unique_anions) * test_ratio))

    test_cations = np.random.choice(unique_cations, size=num_test_cations, replace=False)
    test_anions = np.random.choice(unique_anions, size=num_test_anions, replace=False)

    train_data = data[~data['Cation'].isin(test_cations) & ~data['Anion'].isin(test_anions)]
    test_data = data[data['Cation'].isin(test_cations) & data['Anion'].isin(test_anions)]

    if len(train_data) == 0:
        raise ValueError("Train split is empty. Lower test_ratio or check data diversity.")
    if len(test_data) == 0:
        raise ValueError(
            "Test split is empty under strict OOD split. Lower test_ratio or adjust split strategy."
        )

    train_c_features = encode(train_data['Cation_SMILES'].tolist())
    train_a_features = encode(train_data['Anion_SMILES'].tolist())
    test_c_features = encode(test_data['Cation_SMILES'].tolist())
    test_a_features = encode(test_data['Anion_SMILES'].tolist())

    X_train = torch.cat((train_c_features, train_a_features), dim=1).numpy()
    y_train = train_data['Capacity'].apply(lambda x: np.log10(np.clip(x, 1e-8, None))).values

    X_test = torch.cat((test_c_features, test_a_features), dim=1).numpy()
    y_test = test_data['Capacity'].apply(lambda x: np.log10(np.clip(x, 1e-8, None))).values

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def split_train_val(dataset, val_ratio=0.1, seed=None):
    """Split a TensorDataset into train/val DataLoaders."""
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    return train_loader, val_loader

# Ensemble predictions
@torch.no_grad()
def ensemble_predict(models, X, device):
    X = X.to(device)
    preds = []
    for model in models:
        model = model.to(device)
        model.eval()
        preds.append(model(X).detach().cpu().numpy())
    return np.mean(np.stack(preds, axis=0), axis=0)

def main():
    data = pd.read_csv("sol_data_organic.csv")

    # checkpoint file path
    output_dir = "checkpoints"
    losses_dir = "losses"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(losses_dir, exist_ok=True)
    num_epochs = 15
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Train multiple models with different splits
    train_loader, test_loader = create_train_test_split(data, test_ratio=0.1, seed=42)

    input_dim = train_loader.dataset.tensors[0].shape[1]
    print(f"Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")
    print(f"Inferred input_dim: {input_dim}")
    
    train_data = train_loader.dataset

    models = []
    seed_metrics = []

    for seed in range(5):
        print('-'*50)
        train, val = split_train_val(train_data, val_ratio=0.2, seed=seed)
        print(f"Train size: {len(train.dataset)}, Val size: {len(val.dataset)}")
        print("training model number ", seed)
        model = SolubilityModel(input_dim, hidden_dim=512, num_experts=8)
        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        losses, model = train_model(model, train, val, criterion, optimizer, num_epochs, device)
        models.append(model)

        best_val_loss = min(loss["val_loss"] for loss in losses)
        best_val_epoch = int(np.argmin([loss["val_loss"] for loss in losses])) + 1
        final_train_loss = losses[-1]["train_loss"]
        final_val_loss = losses[-1]["val_loss"]
        seed_metrics.append(
            {
                "seed": seed,
                "best_val_epoch": best_val_epoch,
                "best_val_loss": best_val_loss,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
            }
        )

        plt.figure(figsize=(8, 6))
        plt.plot([loss['train_loss'] for loss in losses], label='Train Loss')
        plt.plot([loss['val_loss'] for loss in losses], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss (Seed {seed})')
        plt.legend()
        plt.savefig(os.path.join(losses_dir, f"loss_curve_seed_{seed}_selfies.png"))
        plt.close()

    # prediction on test set
    X_test = torch.tensor(test_loader.dataset.tensors[0].numpy(), dtype=torch.float32)
    y_test = test_loader.dataset.tensors[1].to(device)
    predictions = ensemble_predict(models, X_test, device)

    r2 = r2_score(y_test.cpu().numpy(), predictions)
    mse = mean_squared_error(y_test.cpu().numpy(), predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.cpu().numpy(), predictions)
    print(f"Test R2: {r2:.4f}, Test MSE: {mse:.4f}, Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")

    y_test = y_test.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Log Capacity')
    plt.ylabel('Predicted Log Capacity')
    plt.title('Neural Network Model Predictions (Test Set)')
    plt.legend()
    plt.savefig('nn_predictions_selfies.png')
    plt.close()

    for idx, model in enumerate(models):
        model_path = os.path.join(output_dir, f"solubility_model_seed_{idx}_selfies.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model checkpoint to {model_path}")

    metrics_path = os.path.join(output_dir, "selfies_moe_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("SELFIES MoE Solubility Training/Evaluation Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"Num epochs: {num_epochs}\n")
        f.write(f"Num models (ensemble size): {len(models)}\n")
        f.write(f"Input dim: {input_dim}\n")
        f.write(f"Train size: {len(train_loader.dataset)}\n")
        f.write(f"Test size: {len(test_loader.dataset)}\n\n")

        f.write("Per-seed train/val summary\n")
        f.write("-" * 60 + "\n")
        for m in seed_metrics:
            f.write(
                f"Seed {m['seed']}: "
                f"best_val_epoch={m['best_val_epoch']}, "
                f"best_val_loss={m['best_val_loss']:.6f}, "
                f"final_train_loss={m['final_train_loss']:.6f}, "
                f"final_val_loss={m['final_val_loss']:.6f}\n"
            )

        f.write("\nAggregate train/val across seeds\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"Mean best val loss: {np.mean([m['best_val_loss'] for m in seed_metrics]):.6f} "
            f"+- {np.std([m['best_val_loss'] for m in seed_metrics]):.6f}\n"
        )
        f.write(
            f"Mean final train loss: {np.mean([m['final_train_loss'] for m in seed_metrics]):.6f} "
            f"+- {np.std([m['final_train_loss'] for m in seed_metrics]):.6f}\n"
        )
        f.write(
            f"Mean final val loss: {np.mean([m['final_val_loss'] for m in seed_metrics]):.6f} "
            f"+- {np.std([m['final_val_loss'] for m in seed_metrics]):.6f}\n"
        )

        f.write("\nTest metrics (ensemble prediction on strict OOD test set)\n")
        f.write("-" * 60 + "\n")
        f.write(f"R2: {r2:.6f}\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")

    print(f"Saved metrics report to {metrics_path}")

if __name__ == "__main__":
    main()