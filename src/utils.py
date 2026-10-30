import numpy as np
import pandas as pd


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(y) == 0:
        return dict(MAE=np.nan, MSE=np.nan, RMSE=np.nan, R2=np.nan, n=0)
    e = p - y
    mse = float(np.mean(e ** 2))
    var = float(np.var(y))
    return dict(MAE=float(np.mean(np.abs(e))), MSE=mse, RMSE=mse ** .5,
                R2=float(1 - mse / var) if var > 0 else np.nan, n=int(len(y)))


def split_samples(samples: pd.DataFrame, mode: str, test_frac, val_frac, shift_q, seed):
    """Retourne dict {train, val, test} -> ensembles d'identifiants d'échantillons.
    mode='random' : tirage aléatoire d'échantillons (même distribution)
    mode='shift'  : train/val = petits réseaux, test = grands réseaux (changement brusque
                    de taille de topologie => distribution différente)."""
    rng = np.random.RandomState(seed)
    ids = samples["sample_id"].values
    if mode == "random":
        perm = rng.permutation(ids)
        n_te = int(len(ids) * test_frac); n_va = int(len(ids) * val_frac)
        return dict(test=set(perm[:n_te]), val=set(perm[n_te:n_te + n_va]),
                    train=set(perm[n_te + n_va:]))
    if mode == "shift":
        thr = samples["N"].quantile(shift_q)
        small = samples[samples["N"] <= thr]["sample_id"].values
        big = samples[samples["N"] > thr]["sample_id"].values
        small = rng.permutation(small)
        n_va = int(len(small) * val_frac / (1 - test_frac))
        return dict(test=set(big), val=set(small[:n_va]), train=set(small[n_va:]))
    raise ValueError(mode)
