"""Construction des features (UNIQUEMENT à partir des entrées de la simulation :
trafic, slices, topologie, routage, configuration des liens) et des événements.
Les sorties du simulateur (délai, pertes, utilisation des liens) ne sont jamais des features."""
import numpy as np
import pandas as pd
from config import PROC_DIR, EVENT_LOSS_THR, EVENT_Z_THR

TYPES = ["eMBB", "URLLC", "mMTC"]

FEATURE_GROUPS = {
    "traffic":     ["log_bw", "log_pkt_size", "log_pkts_lambda"],
    "slice":       ["is_embb", "is_urllc", "is_mmtc", "delta"],
    "path_struct": ["hops", "log_min_cap", "log_sum_tx", "n_wfq_hops"],
    "path_load":   ["max_rho", "mean_rho", "sum_rho", "max_nflows",
                    "max_rho_embb", "max_rho_urllc", "max_rho_mmtc"],
    "queue_wfq":   ["max_rho_q", "mean_rho_q", "sum_rho_q", "min_share", "log_min_guar_ratio"],
    "planned":     ["mean_plan_res", "max_plan_res"],
    "context":     ["log_N", "log_n_flows"],
}
ALL_FEATURES = [c for g in FEATURE_GROUPS.values() for c in g]


def load_raw():
    s = pd.read_csv(PROC_DIR / "samples.csv")
    f = pd.read_csv(PROC_DIR / "flows.csv")
    l = pd.read_csv(PROC_DIR / "links.csv")
    for c, typ in (("links", int), ("hop_res", int), ("hop_q", int), ("hop_share", float), ("hop_rho_q", float)):
        f[c] = f[c].apply(lambda x: [typ(float(v)) for v in str(x).split()])
    return s, f, l


def add_link_load(f, l):
    """Charge offerte par lien d'après le ROUTAGE réel : total + par type de slice."""
    rows = []
    for sid, fg in f.groupby("sample_id"):
        n_links = int((l["sample_id"] == sid).sum())
        load = np.zeros((n_links, 3)); nfl = np.zeros(n_links)
        for lk, bw, ty in zip(fg["links"], fg["bw"], fg["slice_type"]):
            k = TYPES.index(ty)
            for x in lk:
                load[x, k] += bw; nfl[x] += 1
        rows.append(pd.DataFrame(dict(sample_id=sid, link_id=np.arange(n_links), n_flows_link=nfl,
                                      load_embb=load[:, 0], load_urllc=load[:, 1], load_mmtc=load[:, 2])))
    l = l.merge(pd.concat(rows), on=["sample_id", "link_id"], how="left").fillna(
        dict(n_flows_link=0, load_embb=0, load_urllc=0, load_mmtc=0))
    cap = l["capacity"]
    l["load"] = l[["load_embb", "load_urllc", "load_mmtc"]].sum(axis=1)
    l["rho"] = l["load"] / cap
    for t in ("embb", "urllc", "mmtc"):
        l["rho_" + t] = l["load_" + t] / cap
    l["plan_res"] = l["planned_reservation"] / cap
    l["plan_traffic"] = l["planned_traffic"] / cap
    return l


def build_flow_features(s, f, l):
    f = f.copy()
    cap_by = l.set_index(["sample_id", "link_id"])["capacity"]
    f["max_rho_q"] = f.hop_rho_q.apply(max); f["mean_rho_q"] = f.hop_rho_q.apply(np.mean)
    f["sum_rho_q"] = f.hop_rho_q.apply(sum); f["min_share"] = f.hop_share.apply(min)
    # débit minimal garanti sur le chemin (part WFQ x capacité) rapporté à la demande du flux
    gu = [min(sh * cap_by[(sid, lk)] for sh, lk in zip(shs, lks)) / bw
          for sid, shs, lks, bw in zip(f.sample_id, f.hop_share, f.links, f.bw)]
    f["log_min_guar_ratio"] = np.log10(gu)
    f["is_embb"] = (f.slice_type == "eMBB").astype(float)
    f["is_urllc"] = (f.slice_type == "URLLC").astype(float)
    f["is_mmtc"] = (f.slice_type == "mMTC").astype(float)
    f["log_bw"] = np.log10(f.bw); f["log_pkt_size"] = np.log10(f.pkt_size)
    f["log_pkts_lambda"] = np.log10(f.pkts_lambda)
    L = l.set_index(["sample_id", "link_id"])
    agg = {k: [] for k in ["log_min_cap", "log_sum_tx", "n_wfq_hops", "max_rho", "mean_rho", "sum_rho",
                           "max_nflows", "max_rho_embb", "max_rho_urllc", "max_rho_mmtc",
                           "mean_plan_res", "max_plan_res"]}
    byS = {sid: g.set_index("link_id") for sid, g in l.groupby("sample_id")}
    for sid, lk, ps in zip(f.sample_id, f.links, f.pkt_size):
        g = byS[sid].loc[lk]
        agg["log_min_cap"].append(np.log10(g.capacity.min()))
        agg["log_sum_tx"].append(np.log10((ps / g.capacity).sum()))
        agg["n_wfq_hops"].append(g.is_wfq.sum())
        agg["max_rho"].append(g.rho.max()); agg["mean_rho"].append(g.rho.mean()); agg["sum_rho"].append(g.rho.sum())
        agg["max_nflows"].append(g.n_flows_link.max())
        for t in ("embb", "urllc", "mmtc"):
            agg["max_rho_" + t].append(g["rho_" + t].max())
        agg["mean_plan_res"].append(g.plan_res.mean()); agg["max_plan_res"].append(g.plan_res.max())
    for k, v in agg.items():
        f[k] = v
    sm = s.set_index("sample_id")
    f["log_N"] = np.log10(f.sample_id.map(sm["N"])); f["log_n_flows"] = np.log10(f.sample_id.map(sm["n_flows"]))
    f["N"] = f.sample_id.map(sm["N"])
    f["y_ln_delay"] = np.log(f["avg_delay"])           # CIBLE (régression)
    return f


def add_events(f):
    """Événement = dégradation brutale de QoS d'un flux (voir 03_events.py pour la justification).
       (a) délai anormalement élevé POUR SON TYPE DE SLICE : z-score robuste (médiane/MAD de
           ln(délai) du même type) > EVENT_Z_THR      OU
       (b) quasi-effondrement : perte > EVENT_LOSS_THR.
    Le masque ne sert qu'à découper l'évaluation ; il n'est jamais une entrée des modèles."""
    f = f.copy()
    z = np.zeros(len(f))
    for t in TYPES:
        m = (f.slice_type == t).values
        if m.sum() == 0:
            continue
        med = np.median(f.loc[m, "y_ln_delay"]); mad = 1.4826 * np.median(np.abs(f.loc[m, "y_ln_delay"] - med))
        z[m] = (f.loc[m, "y_ln_delay"] - med) / max(mad, 1e-9)
    f["event_z"] = z
    f["event"] = ((z > EVENT_Z_THR) | (f["loss_ratio"] > EVENT_LOSS_THR)).astype(int)
    f["sla_violation"] = (f["avg_delay"] > f["delta"]).astype(int)
    return f


def load_all():
    """Charge tout, construit features + événements. Résultat mis en cache."""
    cache = PROC_DIR / "flows_features.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    s, f, l = load_raw()
    l = add_link_load(f, l)
    f = add_events(build_flow_features(s, f, l))
    out = dict(samples=s, flows=f, links=l)
    pd.to_pickle(out, cache)
    return out
