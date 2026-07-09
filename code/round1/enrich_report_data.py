"""Agrega tablas de deciles e histogramas de score al JSON de reporte de la Ronda 1."""
import json
import numpy as np
import pandas as pd

PROJECT = "/Users/onlive/Library/CloudStorage/GoogleDrive-amatias.morales25@gmail.com/Mi unidad/respaldo-matias-agm/Trabajos Personales/postulacion-bice/DataScientistTest"
REPORT_DATA_PATH = f"{PROJECT}/reports/round1_report_data.json"


def decile_table(y, score):
    df = pd.DataFrame({"y": y, "score": score})
    df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop") + 1
    g = df.groupby("decile").agg(n=("y", "size"), avg_pred=("score", "mean"), default_rate=("y", "mean")).reset_index()
    g = g.sort_values("decile", ascending=False)
    return g.to_dict("records")


def score_hist(scores, bins=30, max_val=None):
    scores = np.asarray(scores)
    hi = max_val if max_val is not None else min(1.0, np.quantile(scores, 0.995))
    hi = max(hi, 0.05)
    counts, edges = np.histogram(scores, bins=bins, range=(0, hi))
    return {"counts": counts.tolist(), "edges": edges.tolist(), "n_above_max": int((scores > hi).sum())}


def score_hist_by_class(y, scores, bins=30, max_val=None):
    y = np.asarray(y)
    scores = np.asarray(scores)
    hi = max_val if max_val is not None else min(1.0, np.quantile(scores, 0.995))
    hi = max(hi, 0.05)
    c0, edges = np.histogram(scores[y == 0], bins=bins, range=(0, hi))
    c1, _ = np.histogram(scores[y == 1], bins=bins, range=(0, hi))
    return {"counts_no_default": c0.tolist(), "counts_default": c1.tolist(), "edges": edges.tolist()}


def main():
    d = json.load(open(REPORT_DATA_PATH))

    d["decile_logit_valid"] = decile_table(d["score_valid_logit"]["y"], d["score_valid_logit"]["score"])
    d["decile_lgbm_valid"] = decile_table(d["score_valid_lgbm"]["y"], d["score_valid_lgbm"]["score"])

    common_max = float(max(
        np.quantile(d["score_valid_logit"]["score"], 0.995),
        np.quantile(d["score_valid_lgbm"]["score"], 0.995),
        np.quantile(d["score_test_logit"], 0.995),
        np.quantile(d["score_test_lgbm"], 0.995),
    ))

    d["hist_valid_logit"] = score_hist_by_class(d["score_valid_logit"]["y"], d["score_valid_logit"]["score"], max_val=common_max)
    d["hist_valid_lgbm"] = score_hist_by_class(d["score_valid_lgbm"]["y"], d["score_valid_lgbm"]["score"], max_val=common_max)
    d["hist_test_logit"] = score_hist(d["score_test_logit"], max_val=common_max)
    d["hist_test_lgbm"] = score_hist(d["score_test_lgbm"], max_val=common_max)

    # ya no se necesitan los arrays crudos completos en el reporte (quedan solo
    # para los graficos ya agregados); se recortan para que el HTML no sea gigante
    del d["score_valid_logit"]
    del d["score_valid_lgbm"]
    del d["score_test_logit"]
    del d["score_test_lgbm"]

    with open(REPORT_DATA_PATH, "w") as f:
        json.dump(d, f, indent=2, allow_nan=True)
    print("OK, enriched", REPORT_DATA_PATH)


if __name__ == "__main__":
    main()
