"""Turn results.json into the plots used in the README."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = json.load(open("results.json"))
Path("assets").mkdir(exist_ok=True)
BLUE, RED, GREEN, GRAY, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#ea580c"
RATE_COLORS = ["#1e3a8a", "#2563eb", "#60a5fa", "#ea580c"]


def distributions() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    a = axes[0]
    for run in R["coin_running_means"]:
        a.plot(np.arange(1, len(run) + 1), run, lw=1)
    a.axhline(0.5, color="k", ls="--", lw=1)
    a.set(xscale="log", ylim=(0, 1), xlabel="number of coin flips n", ylabel="share of heads",
          title="Law of large numbers: 5 runs of coin flips")
    a.grid(alpha=0.3)

    a = axes[1]
    rows = R["std_of_mean"]
    n = [r["n"] for r in rows]
    a.loglog(n, [r["theory_std"] for r in rows], "-", color=GRAY, lw=3, label="theory √(p(1−p)/n)")
    a.loglog(n, [r["empirical_std"] for r in rows], "o", color=BLUE, label="simulated (2000 repeats)")
    a.set(xlabel="sample size n", ylabel="std of the sample mean",
          title="Variance of a mean shrinks like 1/n\n(100× more data → 10× less noise)")
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=9)

    a = axes[2]
    for (k, d), col in zip(R["dice_sums"].items(), RATE_COLORS):
        v = np.array(d["values"], float)
        x = np.arange(d["start"], d["start"] + len(v))
        # standardise so all sums fit one axis: (x − mean) / std
        z = (x - d["theory_mean"]) / np.sqrt(d["theory_var"])
        a.plot(z, v / v.sum() * np.sqrt(d["theory_var"]), "-o" if int(k) <= 2 else "-", ms=3, color=col, label=f"sum of {k} dice" if k != "1" else "1 die")
    zz = np.linspace(-4, 4, 200)
    a.plot(zz, np.exp(-zz**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="normal curve")
    a.set(xlim=(-4, 4), xlabel="standardised sum (x − mean) / std", ylabel="density",
          title="Central limit theorem: sums of dice\nturn into a bell curve")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("assets/distributions.png", dpi=150)


def bayes() -> None:
    b = R["bayes"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    # 1000 people as dots, using the expected counts per 1000
    k = b["per_1000"]
    n_tp, n_fn, n_fp = round(k["true_pos"]), round(k["sick"] - k["true_pos"]), round(k["false_pos"])
    labels = ["sick, test +"] * n_tp + ["sick, test −"] * n_fn + ["healthy, test +"] * n_fp
    labels += ["healthy, test −"] * (1000 - len(labels))
    colors = {"sick, test +": RED, "sick, test −": "#7f1d1d", "healthy, test +": ORANGE, "healthy, test −": "#e2e8f0"}
    order = np.random.default_rng(1).permutation(1000)
    xs, ys = order % 40, order // 40
    for lab, col in colors.items():
        m = np.array([l == lab for l in labels])
        cnt = int(m.sum())
        a1.scatter(xs[m], ys[m], s=26, color=col, label=f"{lab}: {cnt}")
    a1.set(title=f"1000 people, disease in 1 %, test 99 % sensitive, 95 % specific\n"
                 f"positive tests: {n_tp} truly sick + {n_fp} healthy → P(sick | +) = {b['posterior_formula']:.1%}",
           xticks=[], yticks=[])
    a1.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    a1.set_aspect("equal")
    for s in a1.spines.values():
        s.set_visible(False)

    c = b["curve"]
    a2.semilogx([r["prior"] for r in c], [r["posterior"] for r in c], color=BLUE, lw=2, label="after one positive test")
    p = b["prior"]
    a2.plot([p], [b["posterior_formula"]], "o", color=RED)
    a2.annotate(f"1 % sick → {b['posterior_formula']:.1%}", (p, b["posterior_formula"]), xytext=(8, -14), textcoords="offset points", fontsize=9)
    a2.plot([p], [b["posterior_two_positives_formula"]], "s", color=GREEN, label="after a second positive test")
    a2.annotate(f"retest → {b['posterior_two_positives_formula']:.1%}", (p, b["posterior_two_positives_formula"]), xytext=(8, -4), textcoords="offset points", fontsize=9)
    a2.set(xlabel="prior: how common the disease is (log)", ylabel="P(sick | positive test)",
           title="Same test, different prior → very different answer", ylim=(0, 1.02))
    a2.grid(alpha=0.3, which="both")
    a2.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig("assets/bayes.png", dpi=150)


def roc_vs_pr() -> None:
    rows = R["imbalance"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for r, col in zip(rows, RATE_COLORS):
        lab = f"{r['positive_rate']:.1%} positives"
        axes[0].plot(r["roc"]["fpr"], r["roc"]["tpr"], color=col, lw=2, label=f"{lab}: AUC {r['roc_auc']:.3f}")
        axes[1].plot(r["pr"]["recall"], r["pr"]["precision"], color=col, lw=2, label=f"{lab}: AP {r['pr_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].set(xlabel="false positive rate (share of negatives flagged)", ylabel="true positive rate (recall)",
                title="ROC: the 4 curves lie on top of each other")
    axes[1].set(xlabel="recall", ylabel="precision (share of alarms that are real)",
                title="Precision–recall: same model, collapses as positives get rare", ylim=(0, 1.02))
    for a in axes[:2]:
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="lower left" if a is axes[1] else "lower right")

    a = axes[2]
    x = np.arange(len(rows))
    a.bar(x - 0.2, [r["roc_auc"] for r in rows], 0.4, color=GRAY, label="ROC-AUC")
    a.bar(x + 0.2, [r["precision_at_recall_80"] for r in rows], 0.4, color=RED, label="precision at 80 % recall")
    for i, r in enumerate(rows):
        a.text(i + 0.2, r["precision_at_recall_80"] + 0.02, f"{r['false_alarms_per_true_alarm']:.0f} false\nper true" if r["false_alarms_per_true_alarm"] >= 1 else f"{r['false_alarms_per_true_alarm']:.1f} false\nper true", ha="center", fontsize=8)
    a.set_xticks(x, [f"{r['positive_rate']:.1%}" for r in rows])
    a.set(xlabel="share of positives in the data", ylim=(0, 1.1), title="ROC-AUC barely moves, precision collapses")
    a.legend(fontsize=9)
    a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("assets/roc_vs_pr.png", dpi=150)


if __name__ == "__main__":
    distributions()
    bayes()
    roc_vs_pr()
