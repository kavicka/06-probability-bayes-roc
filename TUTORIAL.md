# Tutorial · 06 · Probability, Bayes & ROC-AUC on imbalanced data

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial goes through [`probability.py`](probability.py) experiment by experiment: what is simulated, why it is simulated that way, and what it teaches. The README has the final tables; this file explains how they come about.

**Goal:** be able to explain why ROC-AUC can look excellent on imbalanced data while the model is useless in practice. That needs three ingredients, built in order: noise and sample size, Bayes' rule, and what ROC and precision–recall curves measure.

**You need:** NumPy and scikit-learn (`uv add numpy scikit-learn matplotlib`).

**Why simulate instead of only using formulas:** every formula below is checked against a simulation with a fixed seed (`rng = np.random.default_rng(0)`). If the simulation and the formula agree, you understand the formula. If not, one of them is wrong. And the fixed seed makes every number in the README reproducible.

---

## Part 1 · Noise shrinks slowly: distributions and variance

### Step 1 · Watch the law of large numbers

```python
def coin_running_means(n=5000, runs=5, p=0.5):
    flips = rng.random((runs, n)) < p
    return np.cumsum(flips, 1) / np.arange(1, n + 1)
```

**How it works:** `rng.random() < 0.5` is a fair coin (True = heads). `cumsum / count` is the share of heads after 1, 2, …, n flips. Five independent runs are drawn so you see how differently they start.

**What it shows:** early on the share jumps around wildly; with more flips all runs settle at 0.5. The **mean** (expected value) is the long-run average.

### Step 2 · Measure how fast the noise goes away

```python
def std_of_mean(p=0.5, sizes=(10, 100, 1_000, 10_000), repeats=2000):
    for n in sizes:
        means = rng.binomial(n, p, repeats) / n         # 2000 experiments of n flips
        empirical = means.std()
        theory = np.sqrt(p * (1 - p) / n)
```

**How it works:** for each n we repeat "flip n coins, take the mean" 2000 times and measure how much those means vary. `rng.binomial(n, p)` is a shortcut for "count heads in n flips".

**The theory:** one flip has variance `p(1−p) = 0.25`. The mean of n independent flips has variance `σ²/n`, so its standard deviation is `0.5/√n`. Simulation: 0.157 → 0.0049 for n = 10 → 10,000; theory 0.158 → 0.0050.

**Why it matters for ML:** **100× more data gives only 10× less noise.** A test set of 100 examples has a standard deviation of about ±4 percentage points on accuracy, so it cannot tell 80 % from 83 %. This is why later projects use several seeds and larger test sets before claiming one model beats another.

### Step 3 · Central limit theorem with dice

```python
def dice_sums(counts=(1, 2, 5, 30), samples=100_000):
    s = rng.integers(1, 7, (samples, k)).sum(1)
```

One die is flat (each face 1/6). The sum of 2 dice is a triangle, the sum of 30 dice is a bell curve. **The sum of many independent things looks normal, whatever one thing looks like.** That is why averages, errors and many measurements behave like normal distributions. Check: mean 105.00 and variance 87.46 vs theory 105 and 87.5 (`3.5·k`, `35/12·k`).

---

## Part 2 · Bayes' rule: a positive result needs a base rate

### Step 4 · The formula

```python
def bayes_posterior(prior, sens, spec):
    p_pos = sens * prior + (1 - spec) * (1 - prior)   # all ways to test positive
    return sens * prior / p_pos
```

- **prior** = P(sick) before the test: 1 %.
- **sensitivity** = P(+ | sick): 99 %, the test catches almost every sick person.
- **specificity** = P(− | healthy): 95 %, so 5 % of healthy people test positive by mistake.
- **P(+)** is the denominator: positives come from sick people (`sens·prior`) *and* from healthy people (`(1−spec)·(1−prior)`).

Result: **P(sick | +) = 16.7 %**.

**Intuition with 1,000 people:** 10 are sick, ~10 test positive. 990 are healthy, 5 % of them ≈ 50 test positive. So ~60 positives, only ~10 real. The test is fine; the **base rate** is small, so false positives from the large healthy group swamp the true ones.

### Step 5 · Check it on a million simulated people

```python
sick     = rng.random(people) < prior
positive = np.where(sick, rng.random(people) < sens, rng.random(people) > spec)
posterior_simulated = sick[positive].mean()
```

**How it works:** decide who is sick, then simulate each person's test with the right probability depending on their true state. `sick[positive].mean()` = "among those who tested positive, what share is sick". Simulated 16.5 % vs formula 16.7 %.

### Step 6 · Update with a second test

```python
second = np.where(sick, rng.random(people) < sens, rng.random(people) > spec)
bayes_posterior(p1, sens, spec)          # yesterday's posterior = today's prior
```

**How it works:** after one positive, your new prior is 16.7 %. Plug that into the same formula and a second independent positive gives **79.8 %** (simulated 79.3 %). This "update the belief with each new piece of evidence" is the core of Bayesian thinking.

The script also computes the posterior for priors from 0.01 % to 50 % (`curve`), plotted in `assets/bayes.png`: the same test means 2 % sick at a 0.1 % prior and 69 % sick at 10 %.

---

## Part 3 · ROC-AUC vs precision–recall

### Step 7 · Build one fixed classifier, vary only the imbalance

```python
def imbalance_experiment(rates=(0.5, 0.1, 0.01, 0.001), n=400_000, separation=2.0):
    for rate in rates:
        y = rng.random(n) < rate                       # share of positives
        score = rng.normal(0, 1, n) + separation * y   # negatives ~ N(0,1), positives ~ N(2,1)
```

**Why a synthetic classifier:** we want to change *only one thing*, the share of positives. A real model trained on each dataset would also change. Here the "model" is literally the same in every row: negatives get scores from N(0, 1), positives from N(2, 1). Any change in a metric is caused by the imbalance alone.

### Step 8 · Compute both curves

```python
fpr, tpr, _   = roc_curve(y, score)
prec, rec, _  = precision_recall_curve(y, score)
roc_auc = roc_auc_score(y, score)
pr_auc  = average_precision_score(y, score)
```

A classifier says "positive" when `score > threshold`. Sweeping the threshold gives a curve:

- **ROC:** true positive rate (recall: share of real positives caught) vs false positive rate (share of *negatives* wrongly flagged). **Each rate is computed within one class**, so class sizes cancel out. ROC-AUC = probability that a random positive scores above a random negative.
- **Precision–recall:** precision = TP / (TP + FP), "of my alarms, how many are real". **It mixes both classes**, so it feels the imbalance.

### Step 9 · Pick an operating point and count false alarms

```python
i  = np.argmin(np.abs(rec - 0.8))      # threshold that catches 80 % of positives
tp = rec[i] * y.sum()
fp = tp / prec[i] - tp                 # from precision = tp / (tp + fp)
```

**Why this number:** AUCs summarise the whole curve, but a real system runs at one threshold. "To catch 80 % of the cases, how many false alarms do I get per real one?" is what a user feels.

### Step 10 · Read the table

| Positives | ROC-AUC | PR-AUC | Precision at 80 % recall | False alarms per true alarm |
|---|---|---|---|---|
| 50 % | 0.921 | 0.921 | 0.866 | 0.2 |
| 1 % | 0.920 | 0.279 | 0.061 | 15.3 |
| 0.1 % | 0.915 | 0.052 | 0.006 | 177.0 |

**Why ROC-AUC "lies":** it answers *"can the model rank positives above negatives?"*, and that really doesn't change. On rare positives, the question that matters is *"when it raises an alarm, is it real?"*: with 0.1 % positives even a small false positive rate on 399,600 negatives produces 177 false alarms per real one. And accuracy is worse still: "always say no" is 99.9 % accurate.

**Rules taken into project 07 (churn):** report PR-AUC next to ROC-AUC, compare PR-AUC with the base rate (a random model has PR-AUC = share of positives), and choose the threshold from the costs of each kind of mistake.

---

## Step 11 · Save and plot

`main()` writes everything to `results.json`; [`plot.py`](plot.py) draws `distributions.png`, `bayes.png` and `roc_vs_pr.png` from it.

```bash
uv run python probability.py
uv run python plot.py
```

The same idea on real (simulated) driving data is Racing Lab chapter 06: crashes are 6.3 % of frames, ROC-AUC 0.910, but only 25 % of alarms are real at 80 % recall.

---

## Try it yourself

1. Change `separation` from 2.0 to 3.0. Both AUCs rise; how many false alarms per true alarm remain at 0.1 %?
2. Change specificity to 99.9 % in the Bayes example. How much does one positive test now mean?
3. In `std_of_mean`, use `p = 0.9`. Why is the noise smaller than at `p = 0.5`?

## Recap

| Step | Idea |
|---|---|
| 1–2 | Noise of a mean falls like 1/√n: small test sets give noisy metrics |
| 3 | Sums of many independent things become normal (CLT) |
| 4–6 | Bayes: posterior depends on the prior; update with each new piece of evidence |
| 7 | Change one thing at a time: same classifier, only the imbalance varies |
| 8–10 | ROC is within-class and ignores imbalance; precision feels it |
