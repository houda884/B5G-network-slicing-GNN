"""
ETAPE 1 - Extraction : lit le dataset avec datanetAPI et produit des tables propres.

Sorties (data/processed/) :
  samples.csv  : 1 ligne / simulation (taille du réseau, intensité, métriques globales...)
  flows.csv    : 1 ligne / flux (src->dst) : features d'entrée + cibles QoS
  links.csv    : 1 ligne / lien dirigé : capacité, charge offerte, utilisation simulée...
  graphs.pkl   : structures numpy pour le GNN (liens, flux, incidence flux-lien)

NB importante (vérifiée sur les données) : les chemins simulés sont ceux de la
matrice de routage (corrélation avec l'utilisation réelle des liens ~0.93-0.98) ;
le champ 'path' des slices et les attributs 'traffic'/'reservation' des liens
correspondent à un chemin PLANIFIÉ différent (corr. ~0.4-0.7). On utilise donc
routing_matrix pour construire les features.
"""
import sys, pickle, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import DATA_DIR, PROC_DIR, N_WORKERS
from datanetAPI import DatanetAPI

SLICE_TYPES = ["eMBB", "URLLC", "mMTC"]


def _first_int(v):
    return float(str(v).split(",")[0].split(";")[0])


def _reservation_sum(r):
    try:
        return sum(float(x) for x in str(r).split(",") if x != "")
    except ValueError:
        return 0.0


def _node_queue_maps(G):
    """Pour chaque noeud : dict ToS -> index de file, et poids WFQ par port (liste de listes, en fraction)."""
    maps, weights = {}, {}
    for n, d in G.nodes(data=True):
        groups = str(d.get("tosToQoSqueue", "")).split(";")
        maps[n] = {int(x): q for q, g in enumerate(groups) for x in g.split(",") if x != ""}
        w = str(d.get("schedulingWeights", "-"))
        if d.get("schedulingPolicy") == "WFQ" and w not in ("-", ""):
            weights[n] = [[float(x) / 100.0 for x in g.split(",")] for g in w.split(";")]
        else:
            weights[n] = None
    return maps, weights


def process_sample(s, sid, fname):
    G = s.topology_object
    N = G.number_of_nodes()
    # ---- liens
    links, lidx = [], {}
    for a in G:
        for b in G[a]:
            d = G[a][b][0]
            na = G.nodes[a]
            ps = s.port_stats[a].get(b, {})
            lidx[(a, b)] = len(links)
            links.append(dict(
                sample_id=sid, link_id=len(links), src=a, dst=b,
                capacity=float(d["bandwidth"]), link_speed_raw=float(d.get("LinkSpeedRaw", np.nan)),
                is_wfq=int(na.get("schedulingPolicy") == "WFQ"),
                queue_size=_first_int(na.get("queueSizes", 0)),
                n_qos_levels=int(na.get("levelsQoS", 1)),
                planned_reservation=_reservation_sum(d.get("reservation", "")),
                planned_traffic=_reservation_sum(d.get("traffic", "")),
                util_sim=ps.get("utilization", np.nan),      # SORTIE simulateur (pas une feature)
                loss_sim=ps.get("losses", np.nan)))          # SORTIE simulateur (pas une feature)
    # ---- flux
    flows = []
    for si, sl in enumerate(s.slices):
        for f in sl.flows:
            i, j = f.origin_node, f.destination
            tf = s.traffic_matrix[i, j]["Flows"]
            if not tf:
                continue
            tf = tf[0]
            perf = s.performance_matrix[i, j]["AggInfo"]
            path = s.routing_matrix[i, j]
            lk = [lidx[(a, b)] for a, b in zip(path[:-1], path[1:])]
            gen = tf["PktsGen"]
            flows.append(dict(
                sample_id=sid, flow_id=len(flows), src=i, dst=j, slice_idx=si,
                slice_type=sl.type, delta=sl.delta,
                bw=tf["TimeDistParams"]["EqLambda"], pkts_lambda=tf["TimeDistParams"]["AvgPktsLambda"],
                pkt_size=tf["SizeDistParams"]["AvgPktSize"], tos=tf["ToS"], hops=len(lk),
                links=lk,
                # cibles / sorties QoS
                avg_delay=perf["AvgDelay"], ln_delay=perf["AvgLnDelay"], jitter=perf["Jitter"],
                p90=perf["p90"], pkts_drop=perf["PktsDrop"], pkts_gen=gen,
                loss_ratio=float(np.clip(perf["PktsDrop"] / gen, 0, 1)) if gen > 0 else np.nan))
    # ---- file (queue) traversée à chaque saut + part de capacité garantie (WFQ)
    maps, weights = _node_queue_maps(G)
    res_idx, res_info = {}, []          # (link_id, queue) -> id de "ressource" ; infos
    for fl in flows:
        qs, shares, rids = [], [], []
        for a, b, lk in zip(s.routing_matrix[fl["src"], fl["dst"]][:-1], s.routing_matrix[fl["src"], fl["dst"]][1:], fl["links"]):
            q = maps[a].get(int(fl["tos"]), 0)
            w = weights[a]
            port = G[a][b][0]["port"]
            share = 1.0 if w is None else (w[port][q] if port < len(w) and q < len(w[port]) else 1.0 / len(w[0]))
            key = (lk, q)
            if key not in res_idx:
                res_idx[key] = len(res_info)
                res_info.append(dict(link_id=lk, queue=q, share=max(share, 1e-5), capacity=links[lk]["capacity"],
                                     is_wfq=links[lk]["is_wfq"], load=0.0))
            res_info[res_idx[key]]["load"] += fl["bw"]
            rids.append(res_idx[key]); qs.append(q); shares.append(max(share, 1e-5))
        fl["hop_res"] = rids; fl["hop_q"] = qs; fl["hop_share"] = shares
    for fl in flows:
        fl["hop_rho_q"] = [res_info[r]["load"] / (res_info[r]["share"] * res_info[r]["capacity"]) for r in fl["hop_res"]]
    smp = dict(sample_id=sid, file=fname, N=N, n_links=len(links), n_flows=len(flows),
               n_slices=len(s.slices), intensity=s.maxAvgLambda, sim_time=s.simulation_time,
               global_packets=s.global_packets, global_losses=s.global_losses,
               global_delay=s.global_delay, mean_capacity=np.mean([l["capacity"] for l in links]),
               frac_wfq=float(np.mean([l["is_wfq"] for l in links])),
               **{"n_" + t: sum(sl.type == t for sl in s.slices) for t in SLICE_TYPES})
    return smp, flows, links, res_info


def process_file(args):
    root, fname, start_id = args
    api = DatanetAPI(str(DATA_DIR))
    api.set_files_to_process([(root, fname)])
    out = []
    for k, s in enumerate(api):
        out.append(process_sample(s, start_id + k, fname))
    return out


def main():
    api = DatanetAPI(str(DATA_DIR))
    files = api.get_available_files()
    if not files:
        sys.exit("Aucun .tar.gz trouvé dans %s (voir README : dézipper le dataset dans data/raw)" % DATA_DIR)
    print(len(files), "fichiers tar.gz trouvés")
    jobs = [(r, f, 100000 * k) for k, (r, f) in enumerate(files)]   # ids uniques par fichier
    t0 = time.time()
    if N_WORKERS > 1:
        with Pool(N_WORKERS) as p:
            res = p.map(process_file, jobs)
    else:
        res = [process_file(j) for j in jobs]
    S, F, L, R = [], [], [], []
    for r in res:
        for smp, fl, lk, ri in r:
            S.append(smp); F.extend(fl); L.extend(lk)
            for x in ri: x["sample_id"] = smp["sample_id"]
            R.extend(ri)
    # ids d'échantillons compacts 0..n-1
    remap = {old: new for new, old in enumerate(sorted(d["sample_id"] for d in S))}
    for d in S + F + L + R:
        d["sample_id"] = remap[d["sample_id"]]
    samples = pd.DataFrame(S).sort_values("sample_id").reset_index(drop=True)
    flows = pd.DataFrame(F)
    links = pd.DataFrame(L)
    # graphs.pkl : liste ordonnée par sample_id
    resources = pd.DataFrame(R)
    graphs = {}
    for sid, fg in flows.groupby("sample_id"):
        graphs[sid] = dict(flow_links=fg.sort_values("flow_id")["links"].tolist(),
                           flow_res=fg.sort_values("flow_id")["hop_res"].tolist())
    flows_csv = flows.copy()
    for c in ("links", "hop_res", "hop_q", "hop_share", "hop_rho_q"):
        flows_csv[c] = flows_csv[c].apply(lambda x: " ".join(map(str, x)))
    resources.to_csv(PROC_DIR / "resources.csv", index=False)
    samples.to_csv(PROC_DIR / "samples.csv", index=False)
    flows_csv.to_csv(PROC_DIR / "flows.csv", index=False)
    links.to_csv(PROC_DIR / "links.csv", index=False)
    with open(PROC_DIR / "graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)
    print("samples", samples.shape, "flows", flows.shape, "links", links.shape,
          "en %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
