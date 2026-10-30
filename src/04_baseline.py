"""ETAPES 3 & 7 (L3) - Baselines et modèle ML classique.

Problème ML : prédire ln(délai moyen) de chaque flux (QoS d'un slice) à partir de ses
paramètres d'entrée (trafic, slice, chemin, charge, configuration). C'est un problème de
gestion de slices : un opérateur doit savoir à l'avance si l'allocation actuelle respecte le
délai exigé (delta) et quel slice sera dégradé.

Protocole : split par SIMULATION (jamais par flux). Deux protocoles :
  random : test de même distribution ; shift : test = les réseaux les plus grands (changement
           brusque de taille/trafic), train/val = réseaux plus petits.
Modèles : mean_train (référence triviale), ridge (baseline simple, features sans info de
congestion), hgb (ML non linéaire, avec features de charge agrégées sur le chemin).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from config import *
from features import load_all, FEATURE_GROUPS, ALL_FEATURES
from utils import split_samples
from evaluation import evaluate

d = load_all(); S, F = d["samples"], d["flows"]
BASE = FEATURE_GROUPS["traffic"] + FEATURE_GROUPS["slice"] + FEATURE_GROUPS["path_struct"]
res, preds = [], []


def fit_predict(kind, cols, tr, va, te):
    if kind == "mean_train":
        m = tr[TARGET].mean(); return {k: np.full(len(x), m) for k, x in [("val", va), ("test", te)]}
    if kind == "ridge":
        best = None
        for a in [0.01, 0.1, 1, 10, 100]:
            mdl = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(tr[cols], tr[TARGET])
            e = np.mean((mdl.predict(va[cols]) - va[TARGET]) ** 2)
            if best is None or e < best[0]: best = (e, mdl, a)
        return {"val": best[1].predict(va[cols]), "test": best[1].predict(te[cols])}
    if kind == "hgb":
        best = None
        for lr, depth in [(0.05, None), (0.05, 6), (0.1, 4)]:
            mdl = HistGradientBoostingRegressor(learning_rate=lr, max_depth=depth, max_iter=400, early_stopping=False,
                                                l2_regularization=1.0, random_state=SEED).fit(tr[cols], tr[TARGET])
            e = np.mean((mdl.predict(va[cols]) - va[TARGET]) ** 2)
            if best is None or e < best[0]: best = (e, mdl)
        return {"val": best[1].predict(va[cols]), "test": best[1].predict(te[cols])}


for mode in ["random", "shift"]:
    sp = split_samples(S, mode, TEST_FRAC, VAL_FRAC, SHIFT_QUANTILE, SEED)
    tr, va, te = [F[F.sample_id.isin(sp[k])] for k in ("train", "val", "test")]
    print(mode, "| train/val/test simulations:", len(sp["train"]), len(sp["val"]), len(sp["test"]),
          "| flux:", len(tr), len(va), len(te), "| taux événements test: %.3f" % te.event.mean())
    for name, kind, cols in [("mean_train", "mean_train", []), ("ridge_base", "ridge", BASE), ("hgb_all", "hgb", ALL_FEATURES)]:
        p = fit_predict(kind, cols, tr, va, te)
        t2 = te.assign(pred=p["test"]); res += evaluate(t2, name, mode)
        preds.append(t2[["sample_id", "flow_id", "y_ln_delay", "pred", "event", "slice_type"]].assign(model=name, split=mode))
        # courbe d'ablation par groupe de features (hgb uniquement)
        if name == "hgb_all":
            for g in FEATURE_GROUPS:
                cols_ab = [c for c in ALL_FEATURES if c not in FEATURE_GROUPS[g]]
                pa = fit_predict("hgb", cols_ab, tr, va, te)
                res += evaluate(te.assign(pred=pa["test"]), "hgb_sans_" + g, mode)
R = pd.DataFrame(res); R.to_csv(TAB_DIR / "L3_resultats_baseline_ml.csv", index=False)
pd.concat(preds).to_pickle(PROC_DIR / "preds_baseline.pkl")
pd.set_option("display.width", 200)
print(R[R.subset != "non_event"].pivot_table(index=["split", "model"], columns="subset", values=["MAE", "MSE"]).round(3))
