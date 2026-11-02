# Tutoriál · 06 · Pravdepodobnosť, Bayes a ROC-AUC na nevyvážených dátach

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál prejde [`probability.py`](probability.py) experiment po experimente: čo sa simuluje, prečo práve takto a čo z toho plynie. README má výsledné tabuľky; tento súbor vysvetľuje, ako vzniknú.

**Cieľ:** vedieť vysvetliť, prečo ROC-AUC môže na nevyvážených dátach vyzerať výborne, kým model je v praxi nepoužiteľný. Na to treba tri ingrediencie v poradí: šum a veľkosť vzorky, Bayesovo pravidlo a to, čo merajú ROC a precision–recall krivky.

**Potrebuješ:** NumPy a scikit-learn (`uv add numpy scikit-learn matplotlib`).

**Prečo simulovať a nie len počítať vzorce:** každý vzorec nižšie sa overí simuláciou s pevným seedom (`rng = np.random.default_rng(0)`). Ak sa simulácia a vzorec zhodnú, vzorcu rozumieš. Ak nie, jedno z nich je zle. A pevný seed robí každé číslo v README reprodukovateľným.

---

## Časť 1 · Šum klesá pomaly: rozdelenia a rozptyl

### Krok 1 · Pozri sa na zákon veľkých čísel

```python
def coin_running_means(n=5000, runs=5, p=0.5):
    flips = rng.random((runs, n)) < p
    return np.cumsum(flips, 1) / np.arange(1, n + 1)
```

**Ako to funguje:** `rng.random() < 0.5` je férová minca (True = hlava). `cumsum / počet` je podiel hláv po 1, 2, …, n hodoch. Kreslí sa päť nezávislých behov, aby si videl, aké rôzne začínajú.

**Čo ukazuje:** na začiatku podiel divoko skáče; s viac hodmi sa všetky behy ustália na 0,5. **Stredná hodnota** (očakávaná hodnota) je dlhodobý priemer.

### Krok 2 · Zmeraj, ako rýchlo šum mizne

```python
def std_of_mean(p=0.5, sizes=(10, 100, 1_000, 10_000), repeats=2000):
    for n in sizes:
        means = rng.binomial(n, p, repeats) / n         # 2000 pokusov po n hodoch
        empirical = means.std()
        theory = np.sqrt(p * (1 - p) / n)
```

**Ako to funguje:** pre každé n zopakujeme „hoď n mincí, zober priemer“ 2000-krát a zmeriame, ako veľmi sa tie priemery líšia. `rng.binomial(n, p)` je skratka pre „spočítaj hlavy v n hodoch“.

**Teória:** jeden hod má rozptyl `p(1−p) = 0,25`. Priemer n nezávislých hodov má rozptyl `σ²/n`, takže jeho smerodajná odchýlka je `0,5/√n`. Simulácia: 0,157 → 0,0049 pre n = 10 → 10 000; teória 0,158 → 0,0050.

**Prečo na tom záleží v ML:** **100× viac dát dá len 10× menej šumu.** Testovacia sada so 100 príkladmi má smerodajnú odchýlku presnosti asi ±4 percentuálne body, takže nerozlíši 80 % od 83 %. Preto neskoršie projekty používajú viac seedov a väčšie testovacie sady, kým tvrdia, že jeden model je lepší ako druhý.

### Krok 3 · Centrálna limitná veta na kockách

```python
def dice_sums(counts=(1, 2, 5, 30), samples=100_000):
    s = rng.integers(1, 7, (samples, k)).sum(1)
```

Jedna kocka je plochá (každá stena 1/6). Súčet 2 kociek je trojuholník, súčet 30 kociek je zvonová krivka. **Súčet mnohých nezávislých vecí vyzerá normálne, nech jedna vec vyzerá akokoľvek.** Preto sa priemery, chyby a mnohé merania správajú ako normálne rozdelenie. Kontrola: stredná hodnota 105,00 a rozptyl 87,46 vs teória 105 a 87,5 (`3,5·k`, `35/12·k`).

---

## Časť 2 · Bayesovo pravidlo: pozitívny výsledok potrebuje základnú mieru

### Krok 4 · Vzorec

```python
def bayes_posterior(prior, sens, spec):
    p_pos = sens * prior + (1 - spec) * (1 - prior)   # všetky cesty k pozitívnemu testu
    return sens * prior / p_pos
```

- **prior** = P(chorý) pred testom: 1 %.
- **senzitivita** = P(+ | chorý): 99 %, test zachytí takmer každého chorého.
- **špecificita** = P(− | zdravý): 95 %, takže 5 % zdravých má omylom pozitívny test.
- **P(+)** je menovateľ: pozitívne testy pochádzajú od chorých (`sens·prior`) *aj* od zdravých (`(1−spec)·(1−prior)`).

Výsledok: **P(chorý | +) = 16,7 %**.

**Intuícia na 1 000 ľuďoch:** 10 je chorých, ~10 má pozitívny test. 990 je zdravých, 5 % z nich ≈ 50 má pozitívny test. Takže ~60 pozitívnych, len ~10 skutočných. Test je v poriadku; **základná miera** je malá, takže falošne pozitívni z veľkej zdravej skupiny prehlušia tých skutočných.

### Krok 5 · Over to na milióne simulovaných ľudí

```python
sick     = rng.random(people) < prior
positive = np.where(sick, rng.random(people) < sens, rng.random(people) > spec)
posterior_simulated = sick[positive].mean()
```

**Ako to funguje:** rozhodni, kto je chorý, potom každému nasimuluj test so správnou pravdepodobnosťou podľa jeho skutočného stavu. `sick[positive].mean()` = „medzi tými s pozitívnym testom, aký podiel je chorý“. Simulácia 16,5 % vs vzorec 16,7 %.

### Krok 6 · Aktualizuj druhým testom

```python
second = np.where(sick, rng.random(people) < sens, rng.random(people) > spec)
bayes_posterior(p1, sens, spec)          # včerajší posterior = dnešný prior
```

**Ako to funguje:** po jednom pozitívnom teste je tvoj nový prior 16,7 %. Dosaď ho do toho istého vzorca a druhý nezávislý pozitívny test dá **79,8 %** (simulácia 79,3 %). Toto „aktualizuj presvedčenie každým novým dôkazom“ je jadro bayesovského uvažovania.

Skript tiež spočíta posterior pre priory od 0,01 % po 50 % (`curve`), nakreslené v `assets/bayes.png`: ten istý test znamená 2 % chorých pri priore 0,1 % a 69 % pri 10 %.

---

## Časť 3 · ROC-AUC vs precision–recall

### Krok 7 · Postav jeden pevný klasifikátor a meň len nevyváženosť

```python
def imbalance_experiment(rates=(0.5, 0.1, 0.01, 0.001), n=400_000, separation=2.0):
    for rate in rates:
        y = rng.random(n) < rate                       # podiel pozitívnych
        score = rng.normal(0, 1, n) + separation * y   # negatívne ~ N(0,1), pozitívne ~ N(2,1)
```

**Prečo syntetický klasifikátor:** chceme meniť *len jednu vec*, podiel pozitívnych. Skutočný model natrénovaný na každom datasete by sa tiež zmenil. Tu je „model“ doslova rovnaký v každom riadku: negatívne dostanú skóre z N(0, 1), pozitívne z N(2, 1). Akúkoľvek zmenu metriky spôsobí len nevyváženosť.

### Krok 8 · Spočítaj obe krivky

```python
fpr, tpr, _   = roc_curve(y, score)
prec, rec, _  = precision_recall_curve(y, score)
roc_auc = roc_auc_score(y, score)
pr_auc  = average_precision_score(y, score)
```

Klasifikátor povie „pozitívny“, keď `score > prah`. Posúvanie prahu dá krivku:

- **ROC:** true positive rate (recall: podiel zachytených skutočných pozitívnych) vs false positive rate (podiel *negatívnych* omylom označených). **Každá miera sa počíta v rámci jednej triedy**, takže veľkosti tried sa vykrátia. ROC-AUC = pravdepodobnosť, že náhodný pozitívny dostane vyššie skóre než náhodný negatívny.
- **Precision–recall:** precision = TP / (TP + FP), „z mojich poplachov, koľko je skutočných“. **Mieša obe triedy**, takže nevyváženosť cíti.

### Krok 9 · Vyber pracovný bod a spočítaj falošné poplachy

```python
i  = np.argmin(np.abs(rec - 0.8))      # prah, ktorý zachytí 80 % pozitívnych
tp = rec[i] * y.sum()
fp = tp / prec[i] - tp                 # z precision = tp / (tp + fp)
```

**Prečo toto číslo:** AUC zhrnie celú krivku, ale skutočný systém beží na jednom prahu. „Aby som zachytil 80 % prípadov, koľko falošných poplachov dostanem na jeden skutočný?“ je to, čo pocíti používateľ.

### Krok 10 · Prečítaj tabuľku

| Pozitívnych | ROC-AUC | PR-AUC | Precision pri 80 % recall | Falošných poplachov na skutočný |
|---|---|---|---|---|
| 50 % | 0,921 | 0,921 | 0,866 | 0,2 |
| 1 % | 0,920 | 0,279 | 0,061 | 15,3 |
| 0,1 % | 0,915 | 0,052 | 0,006 | 177,0 |

**Prečo ROC-AUC „klame“:** odpovedá na *„vie model zoradiť pozitívne nad negatívne?“* a to sa naozaj nemení. Pri zriedkavých pozitívnych je dôležitá otázka *„keď spustí poplach, je skutočný?“*: pri 0,1 % pozitívnych aj malá miera falošne pozitívnych na 399 600 negatívnych vyrobí 177 falošných poplachov na jeden skutočný. A accuracy je ešte horšia: „vždy povedz nie“ má 99,9 %.

**Pravidlá, ktoré si berieme do projektu 07 (churn):** uvádzaj PR-AUC vedľa ROC-AUC, porovnávaj PR-AUC so základnou mierou (náhodný model má PR-AUC = podiel pozitívnych) a prah vyberaj podľa cien jednotlivých chýb.

---

## Krok 11 · Ulož a nakresli

`main()` zapíše všetko do `results.json`; [`plot.py`](plot.py) z neho nakreslí `distributions.png`, `bayes.png` a `roc_vs_pr.png`.

```bash
uv run python probability.py
uv run python plot.py
```

Tá istá myšlienka na skutočných (simulovaných) jazdách je kapitola 06 v Racing Lab: havárie sú 6,3 % snímok, ROC-AUC 0,910, ale pri 80 % recall je skutočných len 25 % poplachov.

---

## Vyskúšaj sám

1. Zmeň `separation` z 2,0 na 3,0. Obe AUC stúpnu; koľko falošných poplachov na skutočný zostane pri 0,1 %?
2. Zmeň v Bayesovom príklade špecificitu na 99,9 %. Koľko teraz znamená jeden pozitívny test?
3. V `std_of_mean` použi `p = 0.9`. Prečo je šum menší než pri `p = 0.5`?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 1–2 | Šum priemeru klesá ako 1/√n: malé testovacie sady dávajú zašumené metriky |
| 3 | Súčty mnohých nezávislých vecí sú normálne (CLV) |
| 4–6 | Bayes: posterior závisí od prioru; aktualizuj každým novým dôkazom |
| 7 | Meň jednu vec naraz: rovnaký klasifikátor, mení sa len nevyváženosť |
| 8–10 | ROC je v rámci triedy a nevyváženosť ignoruje; precision ju cíti |
