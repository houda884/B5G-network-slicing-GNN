"""ETAPES 7 & 8 (L6) - Comparaison globale / événementielle, ablation, analyse d'erreurs, figures."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from features import load_all
from utils import metrics

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
B = pd.read_csv(TAB_DIR / "L3_resultats_baseline_ml.csv"); B["seed"] = 0
G = pd.read_csv(TAB_DIR / "L5_resultats_gnn.csv")
R = pd.concat([B, G], ignore_index=True)
# moyenne +- écart-type sur les seeds
agg = R.groupby(["split", "model", "subset"])[["MAE", "MSE"]].agg(["mean", "std"]).round(3)
agg.to_csv(TAB_DIR / "L6_tableau_complet.csv")

main_models = ["mean_train", "ridge_base", "hgb_all", "gnn_best"]
def pick(df):
    df = df.copy(); df["model"] = df.model.str.replace(r"gnn_best\(.*\)", "gnn_best", regex=True); return df
Rp = pick(R)
T = Rp[Rp.model.isin(main_models) & Rp.subset.isin(["global", "event"])].groupby(["split", "model", "subset"])[["MAE", "MSE"]].mean()
T = T.unstack("subset"); T.columns = ["%s_%s" % (a, b) for a, b in T.columns]
T = T.reindex([(s, m) for s in ["random", "shift"] for m in main_models])
T.round(3).to_csv(TAB_DIR / "L6_comparaison_principale.csv"); print("\n=== COMPARAISON PRINCIPALE (test) ===\n", T.round(3))

# robustesse : dégradation random -> shift et gain relatif au baseline
rob = []
for m in main_models:
    for sub in ["global", "event"]:
        a = Rp[(Rp.model == m) & (Rp.split == "random") & (Rp.subset == sub)].MSE.mean()
        b = Rp[(Rp.model == m) & (Rp.split == "shift") & (Rp.subset == sub)].MSE.mean()
        rob.append(dict(model=m, subset=sub, MSE_random=a, MSE_shift=b, ratio_shift_sur_random=b / a))
rob = pd.DataFrame(rob).round(3); rob.to_csv(TAB_DIR / "L6_robustesse_shift.csv", index=False); print("\n=== ROBUSTESSE (MSE) ===\n", rob)

# ablations
ab = Rp[Rp.model.str.startswith(("hgb_sans", "abl_", "gnn_best", "hgb_all")) & (Rp.subset == "global")].groupby(["split", "model"])[["MAE", "MSE"]].mean().round(3)
ab.to_csv(TAB_DIR / "L6_ablations.csv"); print("\n=== ABLATIONS (global) ===\n", ab)

# comparaison des architectures GNN
arch = G[G.model.isin(["gnn_sage", "gnn_gcn", "gnn_gat"]) & (G.subset == "global")][["split", "model", "MAE", "MSE", "val_mse", "params", "epochs"]].round(3)
arch.to_csv(TAB_DIR / "L6_architectures_gnn.csv", index=False); print("\n=== ARCHITECTURES GNN ===\n", arch)

# --------- figures
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for i, sub in enumerate(["global", "event"]):
    for j, met in enumerate(["MAE", "MSE"]):
        a = ax[i, j]; w = 0.2
        for k, m in enumerate(main_models):
            v = [Rp[(Rp.model == m) & (Rp.split == s) & (Rp.subset == sub)][met].mean() for s in ["random", "shift"]]
            a.bar(np.arange(2) + k * w, v, w, label=m)
        a.set_xticks(np.arange(2) + 1.5 * w); a.set_xticklabels(["random", "shift"]); a.set_title("%s - %s (test)" % (met, sub))
ax[0, 0].legend(); plt.tight_layout(); plt.savefig(FIG_DIR / "L6_comparaison.png", dpi=130); plt.close()

d = load_all(); F = d["flows"]
P = pd.concat([pd.read_pickle(PROC_DIR / "preds_baseline.pkl"), pd.read_pickle(PROC_DIR / "preds_gnn.pkl")])
P["model"] = P.model.str.replace(r"gnn_best\(.*\)", "gnn_best", regex=True)
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for i, s in enumerate(["random", "shift"]):
    for j, m in enumerate(["ridge_base", "hgb_all", "gnn_best"]):
        p = P[(P.split == s) & (P.model == m)]
        if "seed" in p: p = p[p.seed.fillna(0) == 0]
        ev = p.event == 1
        ax[i, j].scatter(p.y_ln_delay[~ev], p.pred[~ev], s=2, alpha=.25); ax[i, j].scatter(p.y_ln_delay[ev], p.pred[ev], s=4, c="r", alpha=.5)
        lim = [-12, 8]; ax[i, j].plot(lim, lim, "k--", lw=1); ax[i, j].set_xlim(lim); ax[i, j].set_ylim(lim)
        ax[i, j].set_title("%s | %s (rouge = événements)" % (s, m)); ax[i, j].set_xlabel("ln(délai) réel"); ax[i, j].set_ylabel("prédit")
plt.tight_layout(); plt.savefig(FIG_DIR / "L6_predit_vs_reel.png", dpi=130); plt.close()

# erreur par type de slice et par taille de réseau (split shift)
rows = []
for s in ["random", "shift"]:
    for m in ["ridge_base", "hgb_all", "gnn_best"]:
        p = P[(P.split == s) & (P.model == m)]
        if "seed" in p: p = p[p.seed.fillna(0) == 0]
        p = p.merge(F[["sample_id", "flow_id", "N"]], on=["sample_id", "flow_id"])
        for ty, g in p.groupby("slice_type"):
            rows.append(dict(split=s, model=m, groupe="slice=" + ty, **metrics(g.y_ln_delay, g.pred)))
        for q, g in p.groupby(pd.qcut(p.N, 3, labels=["N petit", "N moyen", "N grand"])):
            rows.append(dict(split=s, model=m, groupe=q, **metrics(g.y_ln_delay, g.pred)))
E = pd.DataFrame(rows).round(3); E.to_csv(TAB_DIR / "L6_erreurs_par_groupe.csv", index=False); print("\n=== ERREURS PAR GROUPE ===\n", E[E.split == "shift"])

curves = pd.read_csv(TAB_DIR / "L5_courbes_apprentissage.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for a, s in zip(ax, ["random", "shift"]):
    c = curves[(curves.split == s) & curves.model.str.startswith("gnn_best") & (curves.seed == 0)]
    a.plot(c.epoch, c.train_mse, label="train MSE (standardisé x var)"); a.plot(c.epoch, c.val_mse, label="validation MSE"); a.set_yscale("log"); a.set_title("Apprentissage GNN - " + s); a.legend(); a.set_xlabel("epoch")
plt.tight_layout(); plt.savefig(FIG_DIR / "L5_courbes_apprentissage.png", dpi=130); plt.close()

# importance des features (permutation, HGB) sur le test random
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from features import ALL_FEATURES
from utils import split_samples
S = d["samples"]; sp = split_samples(S, "random", TEST_FRAC, VAL_FRAC, SHIFT_QUANTILE, SEED)
tr, te = F[F.sample_id.isin(sp["train"])], F[F.sample_id.isin(sp["test"])]
m = HistGradientBoostingRegressor(learning_rate=.05, max_iter=400, l2_regularization=1.0, early_stopping=False, random_state=SEED).fit(tr[ALL_FEATURES], tr[TARGET])
pi = permutation_importance(m, te[ALL_FEATURES], te[TARGET], n_repeats=5, random_state=SEED, scoring="neg_mean_squared_error")
imp = pd.Series(pi.importances_mean, ALL_FEATURES).sort_values(); imp.to_csv(TAB_DIR / "L6_importance_features_hgb.csv", header=["delta_MSE"])
imp.tail(15).plot.barh(figsize=(7, 6), title="Importance par permutation (HGB, test random) : hausse de MSE"); plt.tight_layout(); plt.savefig(FIG_DIR / "L6_importance_features.png", dpi=130); plt.close()
print("\nFigures et tables écrites dans results/")
