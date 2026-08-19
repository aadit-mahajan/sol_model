# sol_model

Predicting the solubility capacity of ionic liquids from the SMILES strings of their cation and anion.

The repo compares three modelling routes on the same dataset and the same held out split, then adds a fragment level interpretability pass so you can see which parts of a molecule the model is leaning on.

> TODO for the author: define what `Capacity` actually measures, including units and the source of the measurements. The rest of this README describes it only as the regression target.

## What is in here

Three modelling tracks, run on the same data.

| Track | Featurisation | Head | Where |
| --- | --- | --- | --- |
| Baseline | ECFP4 count fingerprints, 2048 bits per ion | XGBoost, Random Forest, LightGBM | `emergency_models.ipynb` |
| SMI-TED | `bisectgroup/materials-smi-ted-fork` CLS embedding, 768 dim per ion | MLP ensemble, 5 seeds | `train.py`, `smi_ted_model.ipynb` |
| SELFIES-TED | `ibm/materials.selfies-ted` mean pooled embedding, 1024 dim per ion | MLP ensemble, and a mixture of experts | `selfies_train.py`, `selfies_train_moe.py`, `selfies_ted_model.ipynb` |

Plus one interpretability tool.

| Tool | What it does | Where |
| --- | --- | --- |
| Fragment contributions | Breaks each ion into BRICS fragments, masks one fragment at a time, and measures how much the ensemble prediction moves | `contrib_calc.py` |

## Data

Two CSV files, same schema: `Cation`, `Anion`, `Capacity`, `Cation_SMILES`, `Anion_SMILES`.

| File | Rows | Unique cations | Unique anions | Notes |
| --- | --- | --- | --- | --- |
| `sol_data_cleaned.csv` | 13,607 | 162 | 84 | Full set, includes inorganic anions such as chloride, tetrafluoroborate and hexafluorophosphate |
| `sol_data_organic.csv` | 9,882 | 162 | 61 | Organic anions only. This is the file the training scripts use |

`Capacity` spans roughly 0.014 to 1073. Every script models `log10(Capacity)`, not the raw value.

> TODO for the author: record where this data came from and how it was cleaned.

## The split, and why it matters

All the training scripts use a strict generalisation split rather than a random one.

```
test  = rows where the cation is unseen AND the anion is unseen
train = rows where the cation is seen  AND the anion is seen
```

So the test set contains only ion pairs where neither half has appeared in training. Nothing leaks. This is a much harder test than a random 80/20 split on rows, because a random split lets the model memorise individual ions.

One consequence to be aware of. Rows where only one of the two ions is held out fall into neither bucket and are dropped. At `test_ratio = 0.1` that removes roughly 1,750 of the 9,882 rows, and leaves a small test set. See Known issues.

## Results

The only run with metrics committed to the repo is the SELFIES mixture of experts, in `checkpoints/selfies_moe_metrics.txt`.

| Metric | Value |
| --- | --- |
| Test R2 | 0.851 |
| Test RMSE | 0.318 |
| Test MAE | 0.280 |
| Train rows | 8,030 |
| Test rows | 96 |
| Ensemble size | 5 |
| Epochs | 15 |

Metrics are on `log10(Capacity)`. An RMSE of 0.318 in log space is roughly a factor of two in the original units.

The 96 row test set is small, so treat the R2 as indicative rather than settled.

Prediction scatter plots for the other two neural tracks are saved as `nn_predictions.png` (SMI-TED) and `nn_predictions_selfies.png` (SELFIES-TED). Per seed loss curves are in `losses/`.

> TODO for the author: add the baseline and SMI-TED numbers to a single comparison table so the three tracks can be read side by side.

## Model architecture

Both MLP tracks use the same idea. Encode the cation and the anion separately, then combine them in a way that lets the network see the pairing rather than just the concatenation.

```
cation embedding -> linear -> c
anion  embedding -> linear -> a

features = [c, a, c * a, c + a]   # elementwise product and sum carry the interaction

features -> 3 layer MLP with ReLU and dropout -> log10(Capacity)
```

The mixture of experts variant in `selfies_train_moe.py` replaces the single head with 9 expert MLPs and a softmax gating network over the concatenated embedding. That is the configuration behind the results above.

Five models are trained on different train and validation shuffles, and predictions are averaged.

## Fragment contributions

`contrib_calc.py` answers the question of which chemical groups drive a high or low prediction.

1. Split the ion into fragments at BRICS bonds.
2. For each fragment, neutralise it by turning its atoms into plain carbon.
3. Re-encode and re-predict.
4. The contribution is the drop from the original prediction.

Fragments that would flip the formal charge of the ion are skipped, so a cation stays a cation.

Scores are pushed back to atoms and rendered as an RDKit similarity map. Output goes to `fragment_contribs_smi/`, one PNG per ion pair. About 400 pairs are covered.

## Repo layout

```
train.py                    SMI-TED embeddings, MLP ensemble, entry point
selfies_train.py            SELFIES-TED embeddings, MLP ensemble
selfies_train_moe.py        SELFIES-TED embeddings, mixture of experts, best result so far
contrib_calc.py             BRICS fragment masking and heatmaps

emergency_models.ipynb      ECFP baselines, XGBoost and Random Forest and LightGBM
smi_ted_model.ipynb         SMI-TED exploration, includes a cross attention variant
selfies_ted_model.ipynb     SELFIES-TED exploration, includes an embedding cache

sol_data_cleaned.csv        full dataset
sol_data_organic.csv        organic anions only, used by the scripts

checkpoints/                trained weights and the MoE metrics report
losses/                     per seed training curves
fragment_contribs_smi/      fragment contribution heatmaps
nn_predictions*.png         test set scatter plots
smi_ted_xgb_model.joblib    saved XGBoost model on SMI-TED features
```

## Setup

No dependency file is committed yet. The imports across the repo need roughly this:

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn tqdm joblib
pip install rdkit datamol selfies
pip install xgboost lightgbm optuna
```

`selfies-ted` downloads from the Hugging Face Hub on first use and needs nothing extra.

`smi-ted` is different. `train.py`, `contrib_calc.py` and `smi_ted_model.ipynb` all do `import smi_ted`, which comes from IBM's materials foundation model repository and is not on PyPI. Install it from source before running those files.

A GPU is assumed. `train.py` and `selfies_train.py` hardcode `cuda:0`, and `selfies_train_moe.py` hardcodes `cuda:1`. Change these if your machine differs.

> TODO for the author: add a `requirements.txt` with pinned versions, and the exact install command for `smi_ted`.

## Running it

Train the best current model.

```bash
python selfies_train_moe.py
```

This writes five checkpoints to `checkpoints/`, loss curves to `losses/`, a metrics report to `checkpoints/selfies_moe_metrics.txt`, and a scatter plot to `nn_predictions_selfies.png`.

Train the SMI-TED version.

```bash
python train.py
```

Generate fragment heatmaps. This needs SMI-TED checkpoints from `train.py` to be present.

```bash
python contrib_calc.py
```

## Licence

TODO. No licence file is present, so the default is all rights reserved. Add one if this is meant to be reused.
