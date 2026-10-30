"""ETAPES 5 & 6 (L5) - Représentation graphe + GNN.

GRAPHE (hétérogène, encodé dans un graphe homogène avec 2 encodeurs de features) :
  noeuds "flux"      : un flux (slice flow) src->dst. Features : débit, taille de paquet, type de
                       slice (one-hot), delta (SLA), nb de sauts.
  noeuds "ressource" : une file d'un lien (lien, file WFQ). Features : capacité, part WFQ garantie,
                       capacité garantie, WFQ/FIFO, log(degré) [= nb de flux qui la traversent].
  arêtes (non dirigées) : flux <-> ressource traversée par le flux (d'après le routage).
Pourquoi un graphe : le délai d'un flux dépend de la CONTENTION avec les autres flux qui partagent
ses files ; ce partage est une relation (graphe), pas une propriété du flux seul. Un vecteur de
features ne peut le coder que par des agrégats choisis à la main (cf. HGB).
Par défaut le GNN NE reçoit PAS les charges agrégées (rho) : il doit les apprendre par agrégation.
Sortie : ln(délai) de chaque noeud flux. Perte : MSE sur les noeuds flux.
"""
import argparse, copy, time, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GCNConv, GATv2Conv
from config import *
from features import load_all
from utils import split_samples
from evaluation import evaluate

torch.set_num_threads(2)
FLOW_COLS = ["log_bw", "log_pkt_size", "log_pkts_lambda", "is_embb", "is_urllc", "is_mmtc", "delta", "hops"]
SLICE_COLS = ["is_embb", "is_urllc", "is_mmtc", "delta"]


def build_graphs(d, variant):
    """Une Data par simulation. variant : dict(drop_slice, drop_res_cfg, with_load).
    Trois types de noeuds, dans un tenseur unique x = [feat flux | feat lien | feat file] :
      type 0 = flux ; type 1 = lien ; type 2 = file (lien, queue WFQ)
    arêtes (non dirigées) : flux-file et flux-lien (ressources traversées) ; file-lien (appartenance)."""
    F, R, L = d["flows"], d["resources"], d["links"]
    fcols = [c for c in FLOW_COLS if not (variant.get("drop_slice") and c in SLICE_COLS)]
    graphs = {}
    Rg = {sid: g for sid, g in R.groupby("sample_id")}
    Lg = {sid: g.set_index("link_id") for sid, g in L.groupby("sample_id")}
    for sid, fg in F.groupby("sample_id"):
        fg = fg.sort_values("flow_id"); rg = Rg[sid].reset_index(drop=True); lg = Lg[sid]
        nf, nq = len(fg), len(rg)
        used = sorted(rg.link_id.unique()); lmap = {l: i for i, l in enumerate(used)}; nl = len(used)
        offL, offQ = nf, nf + nl                       # début des noeuds lien / file
        degq = np.zeros(nq); degl = np.zeros(nl); src, dst = [], []
        for i, (hr, hl) in enumerate(zip(fg.hop_res, fg.links)):
            for r in hr:
                src.append(i); dst.append(offQ + r); degq[r] += 1
            for lk in hl:
                src.append(i); dst.append(offL + lmap[lk]); degl[lmap[lk]] += 1
        for r in range(nq):
            src.append(offQ + r); dst.append(offL + lmap[rg.link_id[r]])
        cap_l = lg.loc[used, "capacity"].values
        cap_q = rg.capacity.values; share = rg.share.values
        lcols = [np.log10(cap_l), np.log10(np.maximum(degl, 1))]
        qcols = [np.log10(np.maximum(degq, 1))]
        if not variant.get("drop_res_cfg"):            # configuration WFQ : part garantie, FIFO/WFQ
            lcols.append(lg.loc[used, "is_wfq"].values.astype(float))
            qcols += [np.log10(share), np.log10(share * cap_q)]
        if variant.get("with_load"):                   # charges agrégées données à la main (contrôle)
            lcols += [np.log10(lg.loc[used, "rho"].values + 1e-3)]
            qcols += [np.log10(rg.load.values / (share * cap_q) + 1e-3)]
        xf, xl, xq = fg[fcols].values, np.stack(lcols, 1), np.stack(qcols, 1)
        df_, dl_, dq_ = xf.shape[1], xl.shape[1], xq.shape[1]
        n = nf + nl + nq
        x = np.zeros((n, df_ + dl_ + dq_), dtype=np.float32)
        x[:nf, :df_] = xf; x[offL:offQ, df_:df_ + dl_] = xl; x[offQ:, df_ + dl_:] = xq
        ntype = np.concatenate([np.zeros(nf), np.ones(nl), 2 * np.ones(nq)]).astype(np.int64)
        e = torch.tensor([src, dst], dtype=torch.long)
        y = np.zeros(n, dtype=np.float32); y[:nf] = fg[TARGET].values
        fid = -np.ones(n, dtype=np.int64); fid[:nf] = fg.flow_id.values
        graphs[sid] = Data(x=torch.tensor(x), edge_index=torch.cat([e, e.flip(0)], 1), y=torch.tensor(y),
                           ntype=torch.tensor(ntype), is_flow=torch.tensor(ntype == 0), fid=torch.tensor(fid),
                           sid_node=torch.full((n,), int(sid), dtype=torch.long), dims=torch.tensor([df_, dl_, dq_]))
    return graphs


class Net(nn.Module):
    def __init__(self, dims, conv, hidden, layers, dropout, heads=4, aggr=("mean", "max")):
        super().__init__()
        self.dims = [int(v) for v in dims]; self.off = np.cumsum([0] + self.dims)
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for k in self.dims])
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            if conv == "sage": c = SAGEConv(hidden, hidden, aggr=list(aggr))   # mean+max : charge moyenne ET goulot d'étranglement
            elif conv == "gcn": c = GCNConv(hidden, hidden)
            elif conv == "gat": c = GATv2Conv(hidden, hidden // heads, heads=heads)
            else: raise ValueError(conv)
            self.convs.append(c); self.norms.append(nn.LayerNorm(hidden))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, b):
        x = 0
        for k in range(3):            # un encodeur par type de noeud
            h = self.enc[k](b.x[:, self.off[k]:self.off[k + 1]])
            x = x + (b.ntype == k).unsqueeze(1) * h
        for c, nrm in zip(self.convs, self.norms):
            x = x + self.drop(torch.relu(nrm(c(x, b.edge_index))))          # connexion résiduelle
        return self.head(x).squeeze(-1)


def run_one(graphs, sp, conv, layers, seed, cfg):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_ids, va_ids, te_ids = sp["train"], sp["val"], sp["test"]
    g_ref = graphs[next(iter(tr_ids))]; dims = g_ref.dims
    off = np.cumsum([0] + [int(v) for v in dims])
    xs = torch.cat([graphs[i].x for i in tr_ids]); nt = torch.cat([graphs[i].ntype for i in tr_ids])
    ytr = torch.cat([graphs[i].y[graphs[i].is_flow] for i in tr_ids]); mu, sd = ytr.mean().item(), ytr.std().item()
    mean, std = torch.zeros(xs.size(1)), torch.ones(xs.size(1))
    for k in range(3):
        blk = xs[nt == k][:, off[k]:off[k + 1]]; mean[off[k]:off[k + 1]] = blk.mean(0); std[off[k]:off[k + 1]] = blk.std(0) + 1e-6
    def norm(g):
        g = copy.copy(g); z = (g.x - mean) / std
        for k in range(3):                                   # zéro hors du bloc propre au type du noeud
            m = (g.ntype != k).unsqueeze(1); z[:, off[k]:off[k + 1]] = z[:, off[k]:off[k + 1]].masked_fill(m, 0.0)
        g.x = z; return g
    G = {i: norm(g) for i, g in graphs.items()}
    mk = lambda ids, sh: DataLoader([G[i] for i in sorted(ids)], batch_size=cfg["batch_size"], shuffle=sh)
    dl = dict(train=mk(tr_ids, True), val=mk(va_ids, False), test=mk(te_ids, False))
    model = Net(dims, conv, cfg["hidden"], layers, cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)

    def predict(loader):
        model.eval(); P, Y, SID, FID = [], [], [], []
        with torch.no_grad():
            for b in loader:
                m = b.is_flow; p = model(b) * sd + mu
                P.append(p[m]); Y.append(b.y[m]); SID.append(b.sid_node[m]); FID.append(b.fid[m])
        return [torch.cat(v).numpy() for v in (P, Y, SID, FID)]

    best, best_state, bad, hist = 1e9, None, 0, []
    for ep in range(cfg["epochs"]):
        model.train(); tl = 0.0; n = 0
        for b in dl["train"]:
            opt.zero_grad(); m = b.is_flow
            loss = ((model(b)[m] - (b.y[m] - mu) / sd) ** 2).mean(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); tl += loss.item() * int(m.sum()); n += int(m.sum())
        p, y, _, _ = predict(dl["val"]); vmse = float(np.mean((p - y) ** 2)); sched.step(vmse)
        hist.append((ep, tl / n * sd ** 2, vmse))
        if vmse < best - 1e-4: best, best_state, bad = vmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    model.load_state_dict(best_state)
    p, y, sid, fid = predict(dl["test"])
    return p, y, sid, fid, hist, sum(x.numel() for x in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="random,shift")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=GNN["epochs"])
    ap.add_argument("--quick", action="store_true", help="test rapide (peu d'epochs, 1 seed)")
    a = ap.parse_args()
    cfg = dict(GNN); cfg["epochs"] = 8 if a.quick else a.epochs
    n_seeds = 1 if a.quick else a.seeds

    d = load_all(); d["resources"] = pd.read_csv(PROC_DIR / "resources.csv")
    F = d["flows"].set_index(["sample_id", "flow_id"], drop=False)
    variants = {"full": {}, "no_slice_info": dict(drop_slice=True), "no_wfq_config": dict(drop_res_cfg=True),
                "plus_load_features": dict(with_load=True)}
    G = {k: build_graphs(d, v) for k, v in variants.items()}
    print("graphes construits ; ex. :", G["full"][0])
    rows, preds, curves = [], [], []
    for mode in a.splits.split(","):
        sp = split_samples(d["samples"], mode, TEST_FRAC, VAL_FRAC, SHIFT_QUANTILE, SEED)
        def go(name, graphs, conv, layers, seeds):
            for s in range(seeds):
                t0 = time.time()
                p, y, sid, fid, hist, npar = run_one(graphs, sp, conv, layers, SEED + s, cfg)
                te = F.loc[list(zip(sid, fid))][["sample_id", "flow_id", "event", "slice_type"]].reset_index(drop=True)
                te["y_ln_delay"] = y; te["pred"] = p
                for r in evaluate(te, name, mode): r.update(seed=s, params=npar, epochs=len(hist), val_mse=min(h[2] for h in hist)); rows.append(r)
                preds.append(te.assign(model=name, split=mode, seed=s))
                curves.extend([dict(split=mode, model=name, seed=s, epoch=h[0], train_mse=h[1], val_mse=h[2]) for h in hist])
                g = [r for r in rows if r["model"] == name and r["split"] == mode and r["subset"] == "global"][-1]
                print("  %-7s %-20s seed%d  test MAE %.3f MSE %.3f | %3d ep %.0fs" % (mode, name, s, g["MAE"], g["MSE"], len(hist), time.time() - t0), flush=True)
        # 1) comparaison des architectures (1 seed) -> choix sur la VALIDATION
        for conv in ["sage", "gcn", "gat"]:
            go("gnn_" + conv, G["full"], conv, cfg["layers"], 1)
        sel = min(("sage", "gcn", "gat"), key=lambda c: [r["val_mse"] for r in rows if r["model"] == "gnn_" + c and r["split"] == mode][-1])
        print("  >> architecture retenue (val) pour %s : %s" % (mode, sel))
        # 2) architecture retenue : plusieurs seeds
        go("gnn_best(%s)" % sel, G["full"], sel, cfg["layers"], n_seeds)
        # 3) ablations / contrôles (1 seed)
        go("abl_mlp_sans_graphe", G["full"], sel, 0, 1)
        go("abl_sans_info_slice", G["no_slice_info"], sel, cfg["layers"], 1)
        go("abl_sans_config_wfq", G["no_wfq_config"], sel, cfg["layers"], 1)
        go("abl_plus_features_charge", G["plus_load_features"], sel, cfg["layers"], 1)
    R = pd.DataFrame(rows); R.to_csv(TAB_DIR / "L5_resultats_gnn.csv", index=False)
    pd.concat(preds).to_pickle(PROC_DIR / "preds_gnn.pkl"); pd.DataFrame(curves).to_csv(TAB_DIR / "L5_courbes_apprentissage.csv", index=False)
    print(R[R.subset != "non_event"].pivot_table(index=["split", "model"], columns="subset", values=["MAE", "MSE"]).round(3))


if __name__ == "__main__":
    main()
