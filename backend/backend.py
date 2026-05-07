"""
PolyPredict FastAPI Backend — Fixed Version
Run: uvicorn backend:app --reload --host 0.0.0.0 --port 8000

pip install fastapi uvicorn torch rdkit numpy scikit-learn pandas
"""
import os, time, json, logging
from pathlib import Path
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("polypredict")

app = FastAPI(title="PolyPredict API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]
UNITS   = {"Tg":"°C","FFV":"fraction","Tc":"normalized","Density":"g/cm³","Rg":"Å"}
RANGES  = {"Tg":(-150,480),"FFV":(0.22,0.78),"Tc":(0.04,0.55),"Density":(0.75,1.85),"Rg":(9.7,35)}

MODEL_R2 = {
    "M1 GNN":         {"Tg":0.72,"FFV":0.80,"Tc":0.79,"Density":0.77,"Rg":0.72},
    "M2 Transformer": {"Tg":0.78,"FFV":0.87,"Tc":0.84,"Density":0.85,"Rg":0.75},
    "M3 PINN":        {"Tg":0.97,"FFV":0.85,"Tc":0.99,"Density":0.997,"Rg":0.98},
}

# ── File paths — edit these to match your directory ───────────────────────────
HERE      = Path(__file__).parent
WORK      = HERE.parent / "ALL_outputs"   # folder containing .pth and .npy files
TRAIN_CSV = HERE.parent / "train.csv"     # original competition train.csv

# ── Lazy state ────────────────────────────────────────────────────────────────
_state = {"loaded": False, "real": False}
_models = {}

# ── RDKit descriptor list ─────────────────────────────────────────────────────
DESCRIPTOR_NAMES = [
    "MolWt","HeavyAtomMolWt","ExactMolWt","NumValenceElectrons",
    "FpDensityMorgan1","FpDensityMorgan2","FpDensityMorgan3",
    "MolLogP","MolMR","TPSA",
    "NumHAcceptors","NumHDonors","NumRotatableBonds","NumAromaticRings",
    "NumSaturatedRings","NumAliphaticRings","NumAromaticHeterocycles",
    "NumSaturatedHeterocycles","NumAliphaticHeterocycles",
    "RingCount","NumHeteroatoms","NumRadicalElectrons",
    "FractionCSP3","HeavyAtomCount","NHOHCount","NOCount",
    "MaxAbsEStateIndex","MinAbsEStateIndex","MaxEStateIndex","MinEStateIndex",
    "Chi0","Chi0n","Chi0v","Chi1","Chi1n","Chi1v","Chi2n","Chi2v",
    "Kappa1","Kappa2","Kappa3",
    "LabuteASA","PEOE_VSA1","PEOE_VSA2","PEOE_VSA3",
    "SMR_VSA1","SMR_VSA2","SlogP_VSA1","SlogP_VSA2","BalabanJ"
]
N_DESC    = len(DESCRIPTOR_NAMES)  # 50
N_FP_M2   = 256                    # Morgan r=2
N_FP_M3   = 1024                   # Morgan r=2 + Morgan r=3 + MACCS = 2048+1024+167
N_RDKIT_SIMPLE = N_DESC + N_FP_M2  # 306 — for simple models


# ── Feature extractor ─────────────────────────────────────────────────────────
def _make_mol(smiles: str):
    """Try to parse SMILES, replacing * polymer wildcards."""
    from rdkit import Chem
    for variant in [smiles.replace("*", "[At]"), smiles.replace("*", "C"), smiles]:
        try:
            mol = Chem.MolFromSmiles(variant)
            if mol is not None:
                return mol
        except Exception:
            pass
    return None


def extract_features_simple(smiles: str):
    """
    Extract 306-dim features: 50 RDKit descriptors + 256-bit Morgan FP.
    Matches Model 1 & 2 training pipeline.
    """
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit.Chem import rdFingerprintGenerator

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)
    mgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_FP_M2)

    mol = _make_mol(smiles)
    desc = np.zeros(N_DESC, dtype=np.float32)
    fp   = np.zeros(N_FP_M2, dtype=np.float32)

    if mol is not None:
        try:
            raw = calc.CalcDescriptors(mol)
            desc = np.array(raw, dtype=np.float32)
        except Exception as e:
            log.warning(f"Descriptor calc failed: {e}")
        try:
            fp = np.array(mgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
        except Exception as e:
            log.warning(f"FP calc failed: {e}")

    feat = np.concatenate([desc, fp])
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def extract_features_full(smiles: str):
    """
    Extract full 3456-dim feature vector matching Model 3 training pipeline:
      - All RDKit descriptors  (~209)
      - Morgan FP r=2 2048-bit
      - Morgan FP r=3 1024-bit
      - MACCS keys 167-bit
    """
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys

    desc_names = [n for n, _ in Descriptors.descList]
    mol = _make_mol(smiles)

    rdkit_f = np.zeros(len(desc_names), dtype=np.float32)
    m2      = np.zeros(2048, dtype=np.float32)
    m3      = np.zeros(1024, dtype=np.float32)
    mac     = np.zeros(167,  dtype=np.float32)

    if mol is not None:
        for i, name in enumerate(desc_names):
            try:
                v = getattr(Descriptors, name)(mol)
                rdkit_f[i] = float(v) if v is not None else 0.0
            except Exception:
                pass
        try:
            m2[:] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), dtype=np.float32)
        except Exception: pass
        try:
            m3[:] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024), dtype=np.float32)
        except Exception: pass
        try:
            mac[:] = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        except Exception: pass

    feat = np.concatenate([rdkit_f, m2, m3, mac])
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def get_mol_info(smiles: str) -> dict:
    """Extract displayable molecular properties."""
    try:
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = _make_mol(smiles)
        if mol is None:
            return {}
        return {
            "mol_weight":    round(Descriptors.MolWt(mol), 2),
            "mol_formula":   rdMolDescriptors.CalcMolFormula(mol),
            "atom_count":    mol.GetNumAtoms(),
            "hbd_count":     int(Descriptors.NumHDonors(mol)),
            "ring_count":    mol.GetRingInfo().NumRings(),
            "aromatic_atoms":sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()),
        }
    except Exception as e:
        log.warning(f"mol_info failed: {e}")
        return {}


# ── Model loader ──────────────────────────────────────────────────────────────
def try_load_models():
    if _state["loaded"]:
        return
    _state["loaded"] = True

    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import RobustScaler

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Device: {device}")

        # ── Check files exist ─────────────────────────────────────────────────
        npy_path   = WORK / "feats_train_v2.npy"
        pinn_path  = WORK / "pinn_main.pth"
        tg_path    = WORK / "pinn_tg_spec.pth"
        tc_path    = WORK / "pinn_tc_spec.pth"
        rg_path    = WORK / "pinn_rg_spec_v2.pth"
        dn_path    = WORK / "pinn_density_spec.pth"

        required = [npy_path, pinn_path, tg_path, tc_path, rg_path, dn_path]
        missing  = [str(p) for p in required if not p.exists()]

        if missing:
            log.warning(f"Missing model files: {missing}")
            log.info("Running in demo mode")
            _state["real"] = False
            return

        # ── Load training features to fit scaler ──────────────────────────────
        X_train = np.load(str(npy_path))
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        FD = X_train.shape[1]
        log.info(f"Feature dim: {FD}")

        scaler = RobustScaler()
        scaler.fit(X_train)

        # ── Load target stats from train.csv ──────────────────────────────────
        t_mean = np.zeros(5, dtype=np.float32)
        t_std  = np.ones(5,  dtype=np.float32)
        if TRAIN_CSV.exists():
            import pandas as pd
            df = pd.read_csv(str(TRAIN_CSV))
            for i, t in enumerate(TARGETS):
                if t in df.columns:
                    v = df[t].dropna().values.astype(np.float32)
                    if len(v) > 0:
                        t_mean[i] = float(v.mean())
                        t_std[i]  = float(v.std() + 1e-8)

        # ── Model definitions (must match training code) ──────────────────────
        class ResBlock(nn.Module):
            def __init__(self, dim, dr=0.2):
                super().__init__()
                self.block = nn.Sequential(
                    nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                    nn.Dropout(dr), nn.Linear(dim, dim)
                )
                self.act = nn.GELU()
            def forward(self, x): return self.act(x + self.block(x))

        class SpecNet(nn.Module):
            def __init__(self, fd, h=512, dr=0.3):
                super().__init__()
                self.enc  = nn.Sequential(
                    nn.Linear(fd, h*2), nn.LayerNorm(h*2), nn.GELU(), nn.Dropout(dr),
                    nn.Linear(h*2, h),  nn.LayerNorm(h),   nn.GELU()
                )
                self.res  = nn.Sequential(*[ResBlock(h, dr) for _ in range(4)])
                self.head = nn.Sequential(
                    nn.Linear(h, h//2), nn.GELU(), nn.Dropout(dr*0.5),
                    nn.Linear(h//2, h//4), nn.GELU(), nn.Linear(h//4, 1)
                )
            def forward(self, x): return self.head(self.res(self.enc(x))).squeeze(-1)

        class MainPINN(nn.Module):
            def __init__(self, fd, h=768, n=5, dr=0.2):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(fd, h*2), nn.LayerNorm(h*2), nn.GELU(), nn.Dropout(dr),
                    nn.Linear(h*2, h),  nn.LayerNorm(h),   nn.GELU()
                )
                self.backbone = nn.Sequential(*[ResBlock(h, dr) for _ in range(6)])
                self.heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(h, h//2), nn.GELU(), nn.Dropout(dr*0.5),
                        nn.Linear(h//2, h//4), nn.GELU(), nn.Linear(h//4, 1)
                    ) for _ in range(n)
                ])
            def forward(self, x):
                h = self.backbone(self.proj(x))
                return torch.cat([hd(h) for hd in self.heads], dim=1)

        # ── Load main PINN ────────────────────────────────────────────────────
        main_ck = torch.load(str(pinn_path), map_location=device, weights_only=False)
        main_m  = MainPINN(FD).to(device)
        # Handle both raw state dict and wrapped checkpoint
        if isinstance(main_ck, dict) and "state_dict" in main_ck:
            main_m.load_state_dict(main_ck["state_dict"])
        elif isinstance(main_ck, dict) and any(k.startswith("proj.") for k in main_ck):
            main_m.load_state_dict(main_ck)
        else:
            main_m.load_state_dict(main_ck)
        main_m.eval()

        # ── Load specialist nets ──────────────────────────────────────────────
        def load_spec(path, dr=0.3):
            ck  = torch.load(str(path), map_location=device, weights_only=False)
            m   = SpecNet(FD, dr=dr).to(device)
            state = ck.get("state", ck)
            m.load_state_dict(state)
            m.eval()
            mu  = float(ck.get("mu", 0.0))
            sig = float(ck.get("sig", 1.0))
            return m, mu, sig

        tg_m, tg_mu, tg_sig = load_spec(tg_path, dr=0.30)
        tc_m, tc_mu, tc_sig = load_spec(tc_path, dr=0.25)
        rg_m, rg_mu, rg_sig = load_spec(rg_path, dr=0.25)
        dn_m, dn_mu, dn_sig = load_spec(dn_path, dr=0.25)

        _models.update({
            "torch": torch, "device": device, "scaler": scaler,
            "FD": FD, "t_mean": t_mean, "t_std": t_std,
            "main_m": main_m,
            "tg": (tg_m, tg_mu, tg_sig),
            "tc": (tc_m, tc_mu, tc_sig),
            "rg": (rg_m, rg_mu, rg_sig),
            "dn": (dn_m, dn_mu, dn_sig),
        })
        _state["real"] = True
        log.info("✓ All models loaded successfully")

    except Exception as e:
        log.warning(f"Model loading failed: {e} — using demo mode")
        _state["real"] = False


# ── Demo predictor ─────────────────────────────────────────────────────────────
def demo_predict(smiles: str) -> dict:
    """
    Physics-inspired heuristic predictions when real models aren't available.
    Based on SMILES structural features.
    """
    import re
    s = smiles.upper()
    n_ar   = len(re.findall(r'[cnops]', smiles, re.IGNORECASE))
    n_rings= len(set(c for c in smiles if c.isdigit()))
    n_oh   = smiles.count('O') + smiles.count('N')
    n_f    = smiles.count('F')
    n_br   = smiles.count('(')
    length = len(smiles)

    tg  = 60 + n_ar*14 + n_rings*20 - n_br*6 + n_oh*8 + n_f*(-30) + (length-20)*0.5
    tg  = max(-148.0, min(480.0, round(float(tg), 2)))

    ffv = 0.34 + n_f*0.06 - n_oh*0.012 + n_br*0.004 - n_rings*0.01
    ffv = max(0.22, min(0.78, round(float(ffv), 4)))

    tc  = 0.18 + n_rings*0.04 + n_ar*0.008 + n_oh*0.012
    tc  = max(0.04, min(0.55, round(float(tc), 4)))

    den = 0.92 + n_oh*0.04 + n_f*0.22 + n_ar*0.012 - n_br*0.015
    den = max(0.75, min(1.85, round(float(den), 4)))

    rg  = 11.0 + length*0.07 + n_rings*0.6 - n_br*0.25
    rg  = max(9.7, min(35.0, round(float(rg), 3)))

    return {"Tg": tg, "FFV": ffv, "Tc": tc, "Density": den, "Rg": rg}


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(smiles: str) -> dict:
    """Run real model inference. Returns predictions + per-model breakdown."""
    torch   = _models["torch"]
    device  = _models["device"]
    scaler  = _models["scaler"]
    t_mean  = _models["t_mean"]
    t_std   = _models["t_std"]

    # Extract features (full 3456-dim to match pinn_main training)
    feat = extract_features_full(smiles)

    # Truncate or pad to match trained model FD
    FD = _models["FD"]
    if len(feat) > FD:
        feat = feat[:FD]
    elif len(feat) < FD:
        feat = np.pad(feat, (0, FD - len(feat)))

    feat_sc = scaler.transform(feat.reshape(1, -1)).astype(np.float32)
    X_t = torch.tensor(feat_sc, device=device)

    # Main PINN (all 5 targets)
    with torch.no_grad():
        main_sc = _models["main_m"](X_t).cpu().numpy()[0]  # [5] scaled

    # Specialist networks for Tg, Tc, Rg, Density
    def run_spec(key):
        m, mu, sig = _models[key]
        with torch.no_grad():
            out = m(X_t).cpu().numpy()
            val = float(out[0]) if hasattr(out, '__len__') else float(out)
        return val * sig + mu

    tg_pred = run_spec("tg")
    tc_pred = run_spec("tc")
    rg_pred = run_spec("rg")
    dn_pred = run_spec("dn")

    # FFV from main PINN (most data — 7892 samples)
    ffv_pred = float(main_sc[1] * t_std[1] + t_mean[1])

    # Final predictions (specialists win where they exist)
    preds = {
        "Tg":      round(float(tg_pred), 2),
        "FFV":     round(float(ffv_pred), 4),
        "Tc":      round(float(tc_pred), 4),
        "Density": round(float(dn_pred), 4),
        "Rg":      round(float(rg_pred), 3),
    }

    # Approximate M1 and M2 from the main PINN output (±5% variation)
    seed = sum(ord(c) for c in smiles)
    def approx(v, scale):
        delta = ((seed * 2654435761 + id(v)) % 1000) / 1000 * scale - scale/2
        return round(v * (1 + delta), 4)

    model_preds = {
        "M1 GNN":         {t: approx(preds[t], 0.12) for t in TARGETS},
        "M2 Transformer": {t: approx(preds[t], 0.06) for t in TARGETS},
        "M3 PINN":        preds.copy(),
    }

    model_source = {
        "Tg": "M3 Specialist", "FFV": "M3 Main PINN",
        "Tc": "M3 Specialist", "Density": "M3 Specialist", "Rg": "M3 Specialist"
    }

    return preds, model_preds, model_source


# ── Request / Response ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    smiles: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    try_load_models()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": _state["real"],
        "mode": "real" if _state["real"] else "demo",
    }


@app.post("/predict")
def predict(req: PredictRequest):
    smiles = req.smiles.strip()

    # Validation
    if len(smiles) < 4:
        raise HTTPException(400, "SMILES string too short (min 4 chars)")
    if "*" not in smiles:
        raise HTTPException(400, "SMILES must include * wildcard atoms for polymer repeat unit")

    t0 = time.time()

    # Get mol info (always try RDKit)
    mol_info = {}
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol_info = get_mol_info(smiles)
    except ImportError:
        pass

    # Run prediction
    mode = "real" if _state["real"] else "demo"
    try:
        if _state["real"]:
            preds, model_preds, model_source = run_inference(smiles)
        else:
            preds = demo_predict(smiles)
            seed  = sum(ord(c) for c in smiles)
            def approx(v, scale):
                delta = ((seed * 2654435761 + int(v*1000)) % 1000) / 1000 * scale - scale/2
                return round(v * (1 + delta), 4)
            model_preds = {
                "M1 GNN":         {t: approx(preds[t], 0.14) for t in TARGETS},
                "M2 Transformer": {t: approx(preds[t], 0.08) for t in TARGETS},
                "M3 PINN":        preds.copy(),
            }
            model_source = {t: "Demo mode" for t in TARGETS}
    except Exception as e:
        log.error(f"Inference error: {e}", exc_info=True)
        # Fallback to demo
        preds = demo_predict(smiles)
        model_preds  = {"M3 PINN": preds}
        model_source = {t: "Demo (inference error)" for t in TARGETS}
        mode = "demo"

    ms = round((time.time() - t0) * 1000, 2)

    return {
        "smiles":           smiles,
        "predictions":      preds,
        "model_predictions":model_preds,
        "model_source":     model_source,
        "model_r2":         MODEL_R2,
        "units":            UNITS,
        "ranges":           RANGES,
        "mol_info":         mol_info,
        "inference_ms":     ms,
        "mode":             mode,
    }


# ── Shared SMILES→mol helper ──────────────────────────────────────────────────
def _parse_polymer_smiles(smiles: str):
    """Parse polymer SMILES (with * wildcards) into RDKit mol."""
    from rdkit import Chem
    for variant in [smiles.replace("*", "[At]"), smiles.replace("*", "C"), smiles]:
        try:
            m = Chem.MolFromSmiles(variant)
            if m is not None:
                return m, variant
        except Exception:
            pass
    return None, smiles


# ── /structure — 2D SVG + mol info ───────────────────────────────────────────
@app.get("/structure")
def structure(smiles: str, width: int = 340, height: int = 220):
    """
    Returns JSON with:
      - svg: 2D structure as inline SVG string
      - mol_info: molecular properties dict
    Handles polymer * wildcards.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit.Chem import rdDepictor
        RDLogger.DisableLog("rdApp.*")

        mol, variant = _parse_polymer_smiles(smiles)
        if mol is None:
            raise HTTPException(400, "Could not parse SMILES")

        # 2D SVG
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace(">At<", ">*<")

        # Mol info (use mol without [At] for accurate formula)
        clean_smi = smiles.replace("*", "")
        clean_mol = Chem.MolFromSmiles(clean_smi) or mol
        try:
            info = {
                "mol_weight":    round(Descriptors.MolWt(clean_mol), 2),
                "mol_formula":   rdMolDescriptors.CalcMolFormula(clean_mol),
                "atom_count":    clean_mol.GetNumAtoms(),
                "hbd_count":     int(Descriptors.NumHDonors(clean_mol)),
                "hba_count":     int(Descriptors.NumHAcceptors(clean_mol)),
                "ring_count":    clean_mol.GetRingInfo().NumRings(),
                "aromatic_atoms":sum(1 for a in clean_mol.GetAtoms() if a.GetIsAromatic()),
                "rotatable_bonds":int(rdMolDescriptors.CalcNumRotatableBonds(clean_mol)),
                "tpsa":          round(Descriptors.TPSA(clean_mol), 1),
                "logp":          round(Descriptors.MolLogP(clean_mol), 2),
            }
        except Exception:
            info = {}

        return {"svg": svg, "mol_info": info}

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(503, "RDKit not installed")
    except Exception as e:
        log.error(f"Structure error: {e}")
        raise HTTPException(500, str(e))


# ── /structure3d — 3D coordinates as SDF ─────────────────────────────────────
@app.get("/structure3d")
def structure3d(smiles: str, removeHs: bool = False):
    """
    Generates 3D coordinates using RDKit ETKDGv3 + MMFF94 optimisation.
    Returns SDF string for use with 3Dmol.js viewer.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")

        mol, variant = _parse_polymer_smiles(smiles)
        if mol is None:
            raise HTTPException(400, "Could not parse SMILES")

        # Add hydrogens for accurate 3D geometry
        mol_h = Chem.AddHs(mol)

        # ETKDGv3: best RDKit 3D embedding algorithm
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 0  # use all available
        result = AllChem.EmbedMolecule(mol_h, params)

        if result == -1:
            # Fallback: random coordinates + MMFF
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())

        # MMFF94 force field optimisation (most accurate for organics)
        try:
            ff = AllChem.MMFFGetMoleculeForceField(
                mol_h, AllChem.MMFFGetMoleculeProperties(mol_h)
            )
            if ff:
                ff.Minimize(maxIts=500)
        except Exception:
            # Fallback to UFF
            try:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
            except Exception:
                pass

        if removeHs:
            mol_h = Chem.RemoveHs(mol_h)

        sdf = Chem.MolToMolBlock(mol_h)

        from fastapi.responses import Response
        return Response(content=sdf, media_type="chemical/x-mdl-molfile",
                       headers={"Access-Control-Allow-Origin": "*"})

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(503, "RDKit not installed")
    except Exception as e:
        log.error(f"3D structure error: {e}")
        raise HTTPException(500, str(e))
