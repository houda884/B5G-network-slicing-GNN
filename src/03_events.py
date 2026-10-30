"""ETAPE 4 (L4) - Événements : définition, taux, visualisations.

Mesure de changement : écart robuste (z-score MAD) de ln(délai) d'un flux par rapport à la
médiane des flux du MÊME type de slice ("à quel point ce flux est anormalement dégradé ?").
Règle : événement si z > EVENT_Z_THR (3, ~ 3 écarts-types robustes)  OU  perte > 90 %.
Justification : les simulations sont indépendantes (pas de série temporelle), le "changement
brusque" est donc un basculement de régime d'un flux (fonctionnement normal -> congestion
sévère / effondrement), pas une variation d'un pas de temps au suivant.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import FIG_DIR, TAB_DIR, EVENT_Z_THR, EVENT_LOSS_THR
from features import load_all

d = load_all(); S, F, L = d["samples"], d["flows"], d["links"]
print("taux d'événements global : %.3f" % F.event.mean())
t = F.groupby("slice_type").agg(n=("event", "size"), taux_event=("event", "mean"),
                                 sla_violation=("sla_violation", "mean"), loss_med=("loss_ratio", "median"))
print(t); t.to_csv(TAB_DIR / "L4_taux_evenements_par_slice.csv")
bins = pd.qcut(F.N, 4); tb = F.groupby(bins, observed=True).event.agg(["size", "mean"]); print(tb)
tb.to_csv(TAB_DIR / "L4_taux_evenements_par_taille.csv")
ev_s = F.groupby("sample_id").event.mean()
print("échantillons avec >=1 événement : %d / %d" % ((ev_s > 0).sum(), len(ev_s)))

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for ty, c in zip(["eMBB", "URLLC", "mMTC"], ["C0", "C1", "C2"]):
    ax[0, 0].hist(F[F.slice_type == ty].event_z.clip(-6, 12), 80, alpha=.5, label=ty, color=c)
ax[0, 0].axvline(EVENT_Z_THR, color="k", ls="--"); ax[0, 0].set_title("z-score robuste de ln(délai) par type de slice"); ax[0, 0].legend()
ev = F.event == 1
ax[0, 1].scatter(F.loss_ratio[~ev], F.y_ln_delay[~ev], s=2, alpha=.3, label="normal")
ax[0, 1].scatter(F.loss_ratio[ev], F.y_ln_delay[ev], s=3, alpha=.5, c="r", label="événement")
ax[0, 1].set_xlabel("taux de perte"); ax[0, 1].set_ylabel("ln(délai)"); ax[0, 1].legend(); ax[0, 1].set_title("Flux normaux vs événements")
ax[1, 0].bar(["eMBB", "URLLC", "mMTC"], [F[F.slice_type == k].event.mean() for k in ["eMBB", "URLLC", "mMTC"]]); ax[1, 0].set_title("Taux d'événements par slice")
ax[1, 1].scatter(S.N, ev_s.values, s=12); ax[1, 1].set_xlabel("N"); ax[1, 1].set_ylabel("part de flux en événement"); ax[1, 1].set_title("Événements vs taille du réseau")
plt.tight_layout(); plt.savefig(FIG_DIR / "L4_evenements.png", dpi=130); plt.close()

# vérification visuelle : 1 simulation à fort taux d'événements, relation charge -> délai
sid = ev_s.idxmax(); g = F[F.sample_id == sid]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].scatter(g.max_rho[g.event == 0], g.y_ln_delay[g.event == 0], s=10, label="normal")
ax[0].scatter(g.max_rho[g.event == 1], g.y_ln_delay[g.event == 1], s=14, c="r", label="événement")
ax[0].set_xlabel("max rho sur le chemin"); ax[0].set_ylabel("ln(délai)"); ax[0].legend(); ax[0].set_title("Simulation %d (N=%d)" % (sid, g.N.iloc[0]))
l = L[L.sample_id == sid]
ax[1].scatter(l.rho, l.util_sim, s=8); ax[1].set_xlabel("charge offerte / capacité"); ax[1].set_ylabel("utilisation simulée")
ax[1].set_title("Liens saturés -> flux en événement")
plt.tight_layout(); plt.savefig(FIG_DIR / "L4_verification_visuelle.png", dpi=130); plt.close()
print("figures L4 écrites")
