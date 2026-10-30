"""Configuration centrale du projet (chemins + hyperparamètres)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Dossier qui contient graphs/ routings/ slices/ et les *.tar.gz (après dézippage)
DATA_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "results" / "models"
for d in (PROC_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_WORKERS = 2            # processus pour l'extraction (mettre 1 si problème sous Windows)

# Cible : ln(délai moyen) d'un flux  (le délai est en secondes, très asymétrique)
TARGET = "y_ln_delay"

# Protocole de séparation (au niveau ÉCHANTILLON = simulation, jamais au niveau flux)
TEST_FRAC = 0.2
VAL_FRAC = 0.15
SHIFT_QUANTILE = 0.65    # split "shift" : train/val = réseaux petits, test = les plus grands

# Événements (voir 04_events.py)
EVENT_LOSS_THR = 0.90    # quasi-effondrement : > 90 % des paquets du flux perdus
EVENT_Z_THR = 3.0        # délai anormal pour son type de slice (z-score robuste MAD) > 3

# GNN
GNN = dict(hidden=32, layers=3, dropout=0.2, lr=3e-3, weight_decay=1e-3,
           epochs=120, patience=25, batch_size=8)
