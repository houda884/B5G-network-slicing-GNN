"""ETAPE 2 (L1) - Exploration : tableau des variables + figures + qualité des données."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import FIG_DIR, TAB_DIR
from features import load_all, ALL_FEATURES

d = load_all(); S, F, L = d["samples"], d["flows"], d["links"]

# ---------- 1. Tableau "variable -> signification -> type -> rôle -> justification"
rows = [
 ("sample_id","identifiant d'une simulation (1 topologie + 1 jeu de slices + 1 trafic)","id","regroupement","unité de split train/val/test (évite la fuite : les flux d'une simulation partagent les mêmes liens)"),
 ("N / n_links / n_flows","nb de noeuds / liens dirigés / flux actifs","entier","contexte","taille du réseau : sert à créer un changement de distribution (split shift)"),
 ("slice_type","eMBB / URLLC / mMTC","catégorielle","DEMANDE (type de service)","exigences QoS différentes (débit / latence / massif)"),
 ("delta","exigence de délai du slice (SLA)","continu [0,1]","QoS cible / SLA","permet de définir une violation de SLA"),
 ("bw (EqLambda)","débit offert du flux (bps) : 0.2 / 3.5 / 18 Mbps","continu discret","DEMANDE","charge injectée dans le réseau"),
 ("pkt_size, pkts_lambda","taille moyenne des paquets (bits) et débit de paquets","continu","DEMANDE","détermine temps de service et congestion"),
 ("hops","nombre de liens du chemin routé","entier","RESSOURCE (chemin)","plus de sauts = plus de files d'attente traversées"),
 ("capacity","capacité du lien (bps) : 50/100/350/400 Mbps","continu","RESSOURCE","bande passante disponible"),
 ("is_wfq, queue_size","politique d'ordonnancement du noeud émetteur et taille des files","catégorielle / entier","RESSOURCE (config)","partage de capacité entre slices (WFQ) ou FIFO"),
 ("n_flows_link, load, rho","nb de flux / charge offerte / charge offerte ÷ capacité (d'après le ROUTAGE)","continu","DEMANDE vs RESSOURCE","indicateur direct de congestion, calculé à partir des entrées seulement"),
 ("planned_reservation / planned_traffic","réservation et trafic PLANIFIÉS par slice sur le lien (chemins de slices, pas ceux simulés)","continu","RESSOURCE allouée (partiellement incohérente avec le routage simulé, corr. 0.4-0.7)","feature de configuration, à utiliser avec prudence"),
 ("avg_delay (cible)","délai moyen du flux (s), de 4e-6 à 1825","continu très asymétrique","QoS (SORTIE)","cible : on prédit ln(délai)"),
 ("loss_ratio","pertes / paquets générés","continu [0,1]","QoS (SORTIE)","utilisée pour les événements et l'analyse, pas comme feature"),
 ("jitter, p90","gigue, percentile 90 du délai","continu","QoS (SORTIE)","métriques alternatives"),
 ("util_sim, loss_sim","utilisation / pertes simulées des liens","continu","QoS réseau (SORTIE)","analyse et visualisation d'événements ; jamais en entrée (fuite)"),
]
tab = pd.DataFrame(rows, columns=["variable","signification","type","rôle potentiel","justification"])
tab.to_csv(TAB_DIR / "L1_tableau_variables.csv", index=False, encoding="utf-8-sig")
# ---------- 2. Qualité des données
# On nettoie d'abord les colonnes contenant des listes pour éviter l'erreur "unhashable type: 'list'"
F_safe = F.drop(columns=["links"]).copy()
for col in F_safe.columns:
    if F_safe[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
        F_safe[col] = F_safe[col].astype(str)

q = dict(
    n_samples=len(S), 
    n_flows=len(F), 
    n_links=len(L),
    flows_missing=int(F[["avg_delay","loss_ratio","bw"]].isna().sum().sum()),
    links_missing_util=int(L["util_sim"].isna().sum()),
    flows_duplicated=int(F_safe.duplicated().sum()),  # <-- Utilisation de F_safe ici
    delay_nonpositive=int((F.avg_delay <= 0).sum()),
    loss_out_of_range=int(((F.loss_ratio < 0) | (F.loss_ratio > 1)).sum()),
    intensity_unique=sorted(S.intensity.unique().tolist()),
    util_gt1=int((L.util_sim > 1).sum())
)
pd.Series(q).to_csv(TAB_DIR / "L1_qualite_donnees.csv", header=["valeur"])
print(pd.Series(q)); print(S.describe().T[["min","50%","max"]])

# ---------- 3. Figures
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0,0].hist(np.log10(F.avg_delay), 60); ax[0,0].set_title("log10(délai moyen) des flux"); ax[0,0].set_xlabel("log10 s")
for t in ["eMBB","URLLC","mMTC"]:
    ax[0,1].hist(F[F.slice_type==t].y_ln_delay, 50, alpha=.5, label=t, density=True)
ax[0,1].set_title("ln(délai) par type de slice"); ax[0,1].legend()
ax[0,2].hist(F.loss_ratio, 50); ax[0,2].set_title("Taux de perte par flux")
ax[1,0].hist(S.N, 30); ax[1,0].set_title("Taille des réseaux (nb de noeuds)")
ax[1,1].hist(L.util_sim.dropna(), 60, log=True); ax[1,1].set_title("Utilisation simulée des liens")
ax[1,2].scatter(F.max_rho, F.y_ln_delay, s=2, alpha=.3, c=F.is_embb*0+F.is_urllc+2*F.is_mmtc, cmap="viridis")
ax[1,2].set_xscale("symlog"); ax[1,2].set_title("ln(délai) vs charge max sur le chemin (rho)"); ax[1,2].set_xlabel("max rho")
plt.tight_layout(); plt.savefig(FIG_DIR / "L1_distributions.png", dpi=130); plt.close()

cols = ALL_FEATURES + ["y_ln_delay", "loss_ratio"]
C = F[cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(12, 10)); im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90); ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
plt.colorbar(im); ax.set_title("Corrélation de Spearman"); plt.tight_layout(); plt.savefig(FIG_DIR / "L1_correlations.png", dpi=130); plt.close()
C[["y_ln_delay","loss_ratio"]].to_csv(TAB_DIR / "L1_correlations_cibles.csv")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(S.N, S.global_delay, s=10); ax[0].set_xlabel("N"); ax[0].set_ylabel("délai global moyen"); ax[0].set_yscale("log")
ax[1].scatter(S.N, S.global_losses / S.global_packets, s=10); ax[1].set_xlabel("N"); ax[1].set_ylabel("taux de perte global")
ax[2].scatter(S.n_flows, S.global_delay, s=10); ax[2].set_xlabel("n_flows"); ax[2].set_yscale("log")
plt.tight_layout(); plt.savefig(FIG_DIR / "L1_niveau_simulation.png", dpi=130); plt.close()
print("EDA terminée -> results/figures, results/tables")