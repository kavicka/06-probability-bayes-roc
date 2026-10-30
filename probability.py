"""Step 06 · Probability, distributions, Bayes and why ROC-AUC lies on imbalanced data.

Three experiments, all simulated with a fixed seed so every number in the README is reproducible:
1. Distributions & variance: law of large numbers, variance of a mean shrinks like 1/n, central limit theorem.
2. Bayes: a 99 % accurate test for a disease 1 % of people have. How likely is "sick" after a positive test?
3. ROC vs precision-recall: the SAME classifier on data with 50 % … 0.1 % positives.

Run:  uv run python probability.py   → results.json
      uv run python plot.py          → assets/*.png
"""

import json

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

rng = np.random.default_rng(0)


# --- 1. Distributions & variance --------------------------------------------------
def coin_running_means(n: int = 5000, runs: int = 5, p: float = 0.5) -> list[list[float]]:
    """Law of large numbers: the share of heads drifts towards p as n grows."""
    flips = rng.random((runs, n)) < p
    return (np.cumsum(flips, 1) / np.arange(1, n + 1)).tolist()


def std_of_mean(p: float = 0.5, sizes=(10, 100, 1_000, 10_000), repeats: int = 2000) -> list[dict]:
    """Variance of a mean = σ² / n, so its standard deviation shrinks like 1/√n."""
    rows = []
    for n in sizes:
        means = rng.binomial(n, p, repeats) / n
        rows.append({"n": n, "empirical_std": float(means.std()), "theory_std": float(np.sqrt(p * (1 - p) / n))})
    return rows


def dice_sums(counts=(1, 2, 5, 30), samples: int = 100_000) -> dict:
    """Central limit theorem: a sum of many dice looks like a normal (bell) curve."""
    out = {}
    for k in counts:
        s = rng.integers(1, 7, (samples, k)).sum(1)
        out[str(k)] = {"values": np.bincount(s, minlength=6 * k + 1)[k:].tolist(), "start": k,
                       "mean": float(s.mean()), "var": float(s.var()), "theory_mean": 3.5 * k, "theory_var": 35 / 12 * k}
    return out


# --- 2. Bayes ----------------------------------------------------------------------
def bayes_posterior(prior: float, sens: float, spec: float) -> float:
    """P(sick | +) = P(+ | sick) P(sick) / P(+)."""
    p_pos = sens * prior + (1 - spec) * (1 - prior)
    return sens * prior / p_pos


def bayes_experiment(prior=0.01, sens=0.99, spec=0.95, people: int = 1_000_000) -> dict:
    sick = rng.random(people) < prior
    positive = np.where(sick, rng.random(people) < sens, rng.random(people) > spec)
    second = np.where(sick, rng.random(people) < sens, rng.random(people) > spec)  # an independent retest
    p1 = bayes_posterior(prior, sens, spec)
    return {
        "prior": prior, "sensitivity": sens, "specificity": spec,
        "posterior_formula": p1,
        "posterior_simulated": float(sick[positive].mean()),
        "positives": int(positive.sum()), "true_positives": int((sick & positive).sum()),
        "posterior_two_positives_formula": bayes_posterior(p1, sens, spec),  # yesterday's posterior = today's prior
        "posterior_two_positives_simulated": float(sick[positive & second].mean()),
        "per_1000": {"sick": 1000 * prior, "true_pos": 1000 * prior * sens, "false_pos": 1000 * (1 - prior) * (1 - spec)},
        "curve": [{"prior": float(q), "posterior": bayes_posterior(q, sens, spec)} for q in np.logspace(-4, -0.3, 60)],
    }


# --- 3. ROC vs PR on imbalanced data ----------------------------------------------
def imbalance_experiment(rates=(0.5, 0.1, 0.01, 0.001), n: int = 400_000, separation: float = 2.0) -> list[dict]:
    """One fixed classifier: scores ~ N(0,1) for negatives and N(2,1) for positives.
    Only the share of positives changes between rows."""
    rows = []
    for rate in rates:
        y = rng.random(n) < rate
        score = rng.normal(0, 1, n) + separation * y
        fpr, tpr, _ = roc_curve(y, score)
        prec, rec, _ = precision_recall_curve(y, score)
        i = np.argmin(np.abs(rec - 0.8))  # threshold that catches 80 % of positives
        tp = rec[i] * y.sum()
        fp = tp / prec[i] - tp
        step = max(1, len(fpr) // 400)
        rows.append({
            "positive_rate": rate, "positives": int(y.sum()), "n": n,
            "roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
            "precision_at_recall_80": float(prec[i]),
            "false_alarms_per_true_alarm": float(fp / tp),
            "accuracy_always_negative": float(1 - y.mean()),
            "roc": {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()},
            "pr": {"recall": rec[::step].tolist(), "precision": prec[::step].tolist()},
        })
        print(f"positives {rate:6.1%}: ROC-AUC {rows[-1]['roc_auc']:.3f}  PR-AUC {rows[-1]['pr_auc']:.3f}  "
              f"precision@recall80 {prec[i]:.3f}  false alarms per true alarm {fp / tp:6.1f}  "
              f"'always no' accuracy {1 - y.mean():.3%}")
    return rows


def main() -> None:
    res = {
        "coin_running_means": coin_running_means(),
        "std_of_mean": std_of_mean(),
        "dice_sums": dice_sums(),
        "bayes": bayes_experiment(),
        "imbalance": imbalance_experiment(),
    }
    for r in res["std_of_mean"]:
        print(f"n={r['n']:6d}  std of mean: simulated {r['empirical_std']:.4f}  theory {r['theory_std']:.4f}")
    b = res["bayes"]
    print(f"P(sick | +) formula {b['posterior_formula']:.4f}  simulated {b['posterior_simulated']:.4f}")
    print(f"P(sick | +,+) formula {b['posterior_two_positives_formula']:.4f}  simulated {b['posterior_two_positives_simulated']:.4f}")
    json.dump(res, open("results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
