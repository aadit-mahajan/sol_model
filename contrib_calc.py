import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datamol as dm
import os
import io
from rdkit import Chem
from PIL import Image
import rdkit
import smi_ted
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
from rdkit.Chem import BRICS
from tqdm import tqdm

rdkit.RDLogger.DisableLog('rdApp.*')

# torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
        
def generate_samples():
    org_data = pd.read_csv('./sol_data_organic.csv')
    org_data['Capacity'] = org_data['Capacity'].apply(lambda x: np.log10(np.clip(x, 1e-8, None)))
    sorted_data = org_data.sort_values(by='Capacity', ascending=False)

    groupby_cation = sorted_data.groupby('Cation')
    groupby_anion = sorted_data.groupby('Anion')

    visited_cations = set()
    visited_anions = set()

    cation_samples = []
    anion_samples = []
    n_samples = 3
    for name, group in groupby_cation:
        if name not in visited_cations:
            for i in range(min(n_samples, len(group))):
                cation_samples.append(group.iloc[i])
            visited_cations.add(name)
    for name, group in groupby_anion:
        if name not in visited_anions:
            for i in range(min(n_samples, len(group))):
                anion_samples.append(group.iloc[i])
            visited_anions.add(name)

    cation_samples_df = pd.DataFrame(cation_samples)
    anion_samples_df = pd.DataFrame(anion_samples)

    combined_samples_df = pd.concat([cation_samples_df, anion_samples_df]).drop_duplicates(subset=['Cation', 'Anion']).reset_index(drop=True)

    return combined_samples_df

def load_ensemble_model(checkpoint_path='./checkpoints'):
    enc_out_dim = 768
    hidden_dim = 32
    models = []
    for f in os.listdir(checkpoint_path):
        if f.startswith("solubility_model_seed_") and f.endswith(".pt"):
            model = SolubilityModel(enc_out_dim, hidden_dim)
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, f), map_location=torch.device('cpu')))
            model.eval()
            models.append(model)
            print(f"Loaded model checkpoint from {f}")
    if not models:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoint_path}")
    return models

def ensemble_predict(models, X):
    predictions = np.array([model.predict(X) for model in models])
    return predictions.mean(axis=0)

def get_fragments(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # BRICS bonds to break
    brics_bonds = BRICS.FindBRICSBonds(mol)

    # build adjacency without BRICS bonds
    broken_bonds = set()
    for (a, b), _ in brics_bonds:
        broken_bonds.add(tuple(sorted((a, b))))

    n_atoms = mol.GetNumAtoms()

    # graph traversal on ORIGINAL atom indices
    visited = set()
    fragments = []

    for start in range(n_atoms):
        if start in visited:
            continue

        stack = [start]
        comp = []

        while stack:
            atom_idx = stack.pop()
            if atom_idx in visited:
                continue

            visited.add(atom_idx)
            comp.append(atom_idx)

            atom = mol.GetAtomWithIdx(atom_idx)

            for nbr in atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                bond = tuple(sorted((atom_idx, nbr_idx)))

                if bond in broken_bonds:
                    continue

                if nbr_idx not in visited:
                    stack.append(nbr_idx)

        fragments.append(tuple(comp))

    return mol, fragments

def mask_fragment(mol, atom_indices):
    rw = Chem.RWMol(mol)
    n_atoms = rw.GetNumAtoms()

    for idx in atom_indices:
        if idx >= n_atoms:
            continue   # safety guard

        atom = rw.GetAtomWithIdx(idx)

        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)


    masked = rw.GetMol()
    Chem.SanitizeMol(masked)

    return masked

def fragment_contributions(smiles, partner_smiles, models, is_cation=True):

    mol, frags = get_fragments(smiles)

    ref_input = np.hstack((encode([smiles]),
                           encode([partner_smiles])))
    ref_pred = ensemble_predict(models, torch.tensor(ref_input.reshape(1, -1), dtype=torch.float32))[0]

    contribs = []

    for frag_atoms in frags:

        masked = mask_fragment(mol, frag_atoms)
        masked_smiles = Chem.MolToSmiles(masked)
        charge = sum([atom.GetFormalCharge() for atom in masked.GetAtoms()])
            
        
        if is_cation:
            # check if the charge is preserved in the masked version, if not skip this fragment
            if charge < 0:
                continue

            inp = np.hstack((encode([masked_smiles]),
                             encode([partner_smiles])))
        else:
            # do same here
            if charge > 0:
                continue
            inp = np.hstack((encode([partner_smiles]),
                             encode([masked_smiles])))

        new_pred = ensemble_predict(models, torch.tensor(inp.reshape(1, -1), dtype=torch.float32))[0]

        contrib = ref_pred - new_pred   # intuitive sign

        contribs.append({
            "atoms": frag_atoms,
            "score": float(contrib)
        })

    return contribs

def fragment_to_atom_scores(mol, frag_contribs):

    atom_scores = np.zeros(mol.GetNumAtoms())
    counts = np.zeros(mol.GetNumAtoms())

    for frag in frag_contribs:
        score = frag["score"]

        for idx in frag["atoms"]:
            atom_scores[idx] += score
            counts[idx] += 1

    counts[counts == 0] = 1
    atom_scores /= counts

    return atom_scores

def visualize_fragment_contributions(smiles, contribs):
    """
    Visualize fragment contributions using RDKit's SimilarityMaps.
    """
    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()

    # ----- atom score accumulation -----
    atom_contribs = np.zeros(n_atoms, dtype=float)
    atom_counts = np.zeros(n_atoms, dtype=float)

    for frag in contribs:
        atoms = frag["atoms"]
        score = frag["score"]

        for idx in atoms:
            if idx >= n_atoms:
                continue  # safety
            atom_contribs[idx] += score
            atom_counts[idx] += 1

    # Average overlapping fragment contributions
    atom_counts[atom_counts == 0] = 1
    atom_contribs = atom_contribs / atom_counts

    # ----- normalization for visualization -----
    max_abs = np.max(np.abs(atom_contribs))
    if max_abs == 0:
        max_abs = 1.0

    norm_contribs = (atom_contribs / max_abs).tolist()

    # ----- similarity map -----
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)

    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        norm_contribs,
        colorMap='coolwarm',
        contourLines=10,
        alpha=0.5,
        draw2d=drawer
    )

    drawer.FinishDrawing()

    png = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png))

    return atom_contribs.tolist(), img

def main():
    os.makedirs('./fragment_contribs_smi', exist_ok=True)

    combined_samples_df = generate_samples()
    models = load_ensemble_model()

    frag_contrib_path = './fragment_contribs_smi'
    for index, entry in tqdm(combined_samples_df.iterrows(), total=len(combined_samples_df), desc="Calculating fragment contributions"):
        smiles_cation = entry['Cation_SMILES']
        smiles_anion = entry['Anion_SMILES']
        cation_label = f"Cation: {entry['Cation']}"
        capacity = entry['Capacity']

        try:
            contribs_cation = fragment_contributions(smiles_cation, smiles_anion, models, is_cation=True)
            contribs_anion = fragment_contributions(smiles_anion, smiles_cation, models, is_cation=False)
            contribs_cation, img_cation = visualize_fragment_contributions(smiles_cation, contribs_cation)
            contribs_anion, img_anion = visualize_fragment_contributions(smiles_anion, contribs_anion)
        except:
            continue

        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        ax[0].imshow(img_cation)
        ax[0].axis('off')
        ax[0].set_title(f"{cation_label}")
        ax[1].imshow(img_anion)
        ax[1].axis('off')
        ax[1].set_title(f"Anion: {entry['Anion']}")
        fig.suptitle('Fragment Contribution Heatmaps', fontsize=16)
        fig.text(0.5, 0.04, f'True log(Capacity): {np.log10(capacity):.2f}', ha='center', fontsize=12)
        plt.savefig(os.path.join(frag_contrib_path, f'{entry["Cation"]}_{entry["Anion"]}_fragment_contrib.png'))
        plt.close()

if __name__ == "__main__":
    main()
    print("run complete.")

    