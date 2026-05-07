# PolyPredict — Polymer Property Prediction

A machine-learning system for predicting key polymer properties from SMILES strings, combining three complementary model architectures with a FastAPI backend and interactive web frontend.

## Predicted Properties

| Property | Unit | Description |
|---|---|---|
| **Tg** | °C | Glass transition temperature |
| **FFV** | fraction | Fractional free volume |
| **Tc** | normalized | Crystallisation temperature |
| **Density** | g/cm³ | Polymer density |
| **Rg** | Å | Radius of gyration |

## Models

| Model | Architecture | Avg R² |
|---|---|---|
| M1 GNN | Graph Neural Network | ~0.76 |
| M2 Transformer | 6-layer Transformer + RDKit fusion | ~0.82 |
| M3 PINN | Physics-Informed Neural Network | ~0.97 |

## Project Structure

```
POLYMERS/
├── index.html                          # Frontend — polymer property prediction UI
├── backend/
│   ├── backend.py                      # FastAPI backend (uvicorn)
│   └── polymer_structures.json         # Polymer structure reference data
├── ALL_outputs/
│   ├── model1_gnn.pth                  # Trained GNN weights
│   ├── model2_transformer.pth          # Trained Transformer weights
│   ├── model3_pinn.pth                 # Trained PINN weights
│   ├── pinn_*_spec*.pth                # Specialist PINN models per property
│   ├── ffv_xgb.json                    # XGBoost FFV model
│   ├── feats_test_v2.npy               # Precomputed test features
│   ├── submission*.csv                 # Competition submission files
│   └── ...
└── notebooka800e7507d (1).ipynb        # Training & analysis notebook
```

## Setup & Usage

### Install dependencies

```bash
pip install fastapi uvicorn torch rdkit numpy scikit-learn pandas
```

### Run the backend

```bash
cd backend
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

Then open `index.html` in your browser (or serve it from a static server).

### Input

Enter any valid polymer SMILES string into the UI. The backend computes RDKit descriptors and runs all three models, returning ensemble predictions with confidence ranges.

## Notes

- `feats_train_v2.npy`, `ALL_outputs.zip`, and `ALL_outputs/model3_all_outputs.zip` are excluded from this repo due to GitHub's 100 MB file size limit.
- The notebook contains full training code, hyperparameter details, and evaluation results.
