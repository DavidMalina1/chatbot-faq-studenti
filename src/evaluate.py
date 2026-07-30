"""Evaluarea comparativă a metodelor de căutare.

Pentru fiecare întrebare de test (o parafrază a unei întrebări din FAQ),
verificăm dacă răspunsul corect apare în primele K rezultate returnate.

Metrici raportate:
  - recall@1, recall@3, recall@5: proporția întrebărilor de test pentru
    care răspunsul corect apare în primele 1 / 3 / 5 rezultate;
  - MRR (Mean Reciprocal Rank): media valorilor 1/rang, unde rangul este
    poziția răspunsului corect în lista ordonată.

Rulare:
    python src/evaluate.py                 # keyword + tfidf + lsa
    python src/evaluate.py --sbert         # include și sentence-transformers

Rezultatele sunt salvate în reports/rezultate/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from search import build_engines
from utils import ROOT, load_faq, load_test_questions

OUT_DIR = ROOT / "reports" / "rezultate"
K_VALUES = (1, 3, 5)
LABELS = {
    "keyword": "Cuvinte-cheie (bază)",
    "tfidf": "TF-IDF + cosinus",
    "lsa": "Semantic (LSA)",
    "ensemble": "Combinat (medie ponderată)",
    "sbert": "Semantic (SBERT)",
}


def rank_of_expected(engine, query: str, expected_id: int) -> int:
    """Poziția (1-indexată) a răspunsului corect în lista ordonată."""
    scores = engine.scores(query)
    order = np.argsort(-scores)
    ids = engine.faq["id"].to_numpy()[order]
    return int(np.where(ids == expected_id)[0][0]) + 1


def evaluate_engine(engine, tests: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Calculează metricile globale și rezultatele per întrebare."""
    rows = []
    t0 = time.perf_counter()
    for _, row in tests.iterrows():
        rank = rank_of_expected(engine, row["intrebare_test"], row["id_asteptat"])
        top1 = engine.search(row["intrebare_test"], top_k=1).iloc[0]
        rows.append({
            "intrebare_test": row["intrebare_test"],
            "id_asteptat": row["id_asteptat"],
            "id_prezis_top1": top1["id"],
            "intrebare_prezisa_top1": top1["intrebare"],
            "scor_top1": round(float(top1["scor"]), 3),
            "rang_raspuns_corect": rank,
        })
    elapsed_ms = (time.perf_counter() - t0) * 1000 / len(tests)

    detail = pd.DataFrame(rows)
    metrics = {"metoda": engine.name}
    for k in K_VALUES:
        metrics[f"recall@{k}"] = round((detail["rang_raspuns_corect"] <= k).mean(), 3)
    metrics["MRR"] = round((1 / detail["rang_raspuns_corect"]).mean(), 3)
    metrics["timp_mediu_ms"] = round(elapsed_ms, 2)
    return metrics, detail


def plot_metrics(summary: pd.DataFrame, path: Path) -> None:
    """Grafic cu bare: recall@K pentru fiecare metodă."""
    methods = summary["metoda"].tolist()
    x = np.arange(len(K_VALUES))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#9aa5b1", "#4a7fb5", "#2c5f2d", "#7b4fa6", "#b85042"]
    for i, m in enumerate(methods):
        vals = [summary.loc[summary["metoda"] == m, f"recall@{k}"].iloc[0]
                for k in K_VALUES]
        bars = ax.bar(x + i * width, vals, width,
                      label=LABELS.get(m, m), color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"recall@{k}" for k in K_VALUES])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Proporție întrebări rezolvate")
    ax.set_title("Compararea metodelor de căutare pe setul de test")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rank_distribution(details: dict[str, pd.DataFrame], path: Path) -> None:
    """Distribuția rangului răspunsului corect, pe metode."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.arange(0.5, 11.5, 1)
    colors = ["#9aa5b1", "#4a7fb5", "#2c5f2d", "#7b4fa6", "#b85042"]
    for i, (name, detail) in enumerate(details.items()):
        ranks = detail["rang_raspuns_corect"].clip(upper=10)
        hist, _ = np.histogram(ranks, bins=bins)
        ax.plot(range(1, 11), np.cumsum(hist) / len(ranks), marker="o",
                label=LABELS.get(name, name), color=colors[i % len(colors)])
    ax.set_xlabel("Rang maxim acceptat (K)")
    ax.set_ylabel("Proporție cumulată (recall@K)")
    ax.set_title("Recall cumulat în funcție de K")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbert", action="store_true",
                        help="include și varianta sentence-transformers")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faq = load_faq()
    tests = load_test_questions()
    engines = build_engines(faq, include_sbert=args.sbert)

    summaries, details = [], {}
    for name, engine in engines.items():
        metrics, detail = evaluate_engine(engine, tests)
        summaries.append(metrics)
        details[name] = detail
        detail.to_csv(OUT_DIR / f"detalii_{name}.csv", index=False)
        greseli = detail[detail["rang_raspuns_corect"] > 1]
        print(f"\n=== {LABELS.get(name, name)} ===")
        print({k: v for k, v in metrics.items() if k != "metoda"})
        print(f"Întrebări cu top-1 greșit: {len(greseli)} / {len(detail)}")

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_DIR / "metrici.csv", index=False)
    plot_metrics(summary, OUT_DIR / "recall_comparatie.png")
    plot_rank_distribution(details, OUT_DIR / "recall_cumulat.png")

    print("\nTabel final:")
    print(summary.to_string(index=False))
    print(f"\nRezultatele au fost salvate în {OUT_DIR}")


if __name__ == "__main__":
    main()
