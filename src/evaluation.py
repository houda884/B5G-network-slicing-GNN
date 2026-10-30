import numpy as np, pandas as pd
from utils import metrics


def evaluate(df, model, split_mode, pred_col="pred", y_col="y_ln_delay"):
    """df : lignes de TEST avec colonnes y, pred, event. Retourne 3 lignes (global / event / non-event)."""
    out = []
    for name, m in [("global", np.ones(len(df), bool)), ("event", df.event.values == 1), ("non_event", df.event.values == 0)]:
        r = metrics(df.loc[m, y_col], df.loc[m, pred_col])
        out.append(dict(split=split_mode, model=model, subset=name, **r))
    return out
