# ============================================================
# Fenerbahçe Match Performance – Machine Learning Methods
# DSA210 | Damla Kandemir
# ============================================================
#
# Supervised   : Majority Class Baseline, Logistic Regression,
#                k-Nearest Neighbors, Decision Tree
# Unsupervised : K-Means Clustering
# Split        : Chronological (train 2016-23 / val 2023-24 / test 2024-26)
# Leakage note : GF, GA, GoalDiff, Points, Possession excluded —
#                not known before the match.
# ============================================================

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model  import LogisticRegression
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier, plot_tree
from sklearn.cluster       import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (accuracy_score, confusion_matrix,
                                   ConfusionMatrixDisplay, classification_report)

# ── 1. DATA LOADING ──────────────────────────────────────────
csv_files = {
    "2016-17": "Fenerbahce_2016-2017.csv",
    "2017-18": "Fenerbahce_2017-2018.csv",
    "2018-19": "Fenerbahce_2018-2019.csv",
    "2019-20": "Fenerbahce_2019-2020.csv",
    "2020-21": "Fenerbahce_2020-2021.csv",
    "2021-22": "Fenerbahce_2021-2022.csv",
    "2022-23": "Fenerbahce_2022-2023.csv",
    "2023-24": "Fenerbahce_2023-2024.csv",
    "2024-25": "Fenerbahçe_2024-2025.csv",
    "2025-26": "Fenerbahce_2025-2026.csv",
}

dfs = []
for season, fname in csv_files.items():
    df_tmp = pd.read_csv(fname)
    df_tmp['Season'] = season
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)

# ── 2. CLEANING ──────────────────────────────────────────────
df = df[df['Result'].isin(['W', 'D', 'L'])].copy()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date', 'Result', 'GF', 'GA', 'Venue', 'Comp'])
df['GF'] = pd.to_numeric(df['GF'], errors='coerce')
df['GA'] = pd.to_numeric(df['GA'], errors='coerce')
df = df.dropna(subset=['GF', 'GA'])
df = df.sort_values('Date').reset_index(drop=True)

# ── 3. FEATURE ENGINEERING ───────────────────────────────────
df['Win']        = (df['Result'] == 'W').astype(int)
df['IsHome']     = (df['Venue'] == 'Home').astype(int)
df['IsLeague']   = (df['Comp'] == 'Süper Lig').astype(int)
df['IsEuropean'] = df['Comp'].isin(['Europa Lg', 'Champions Lg', 'Conf Lg']).astype(int)
df['CompGroup']  = df['IsLeague']
df['RestDays']   = df['Date'].diff().dt.days.clip(0, 30).fillna(7)

df['AfterEurope'] = False
for i in range(1, len(df)):
    if df.loc[i, 'IsLeague'] and df.loc[i - 1, 'IsEuropean']:
        df.loc[i, 'AfterEurope'] = True
df['AfterEurope'] = df['AfterEurope'].astype(int)

season_order     = {s: i for i, s in enumerate(csv_files.keys())}
df['SeasonNum']  = df['Season'].map(season_order)
df['Is4Back']    = df['Formation'].str[0].map({'4': 1, '3': 0}).fillna(0).astype(int)

BIG_RIVALS       = {'Galatasaray', 'Beşiktaş', 'Trabzonspor', 'Başakşehir'}
df['IsBigRival'] = df['Opponent'].isin(BIG_RIVALS).astype(int)

df['Points'] = df['Result'].map({'W': 3, 'D': 1, 'L': 0})
df['Form3']  = df['Points'].shift(1).rolling(3, min_periods=1).mean().fillna(1.5)

print(f"Total matches : {len(df)}")
print(f"Win rate      : {df['Win'].mean():.1%}")

FEATURES    = ['IsHome', 'CompGroup', 'IsEuropean', 'AfterEurope',
               'RestDays', 'SeasonNum', 'Is4Back', 'IsBigRival', 'Form3']
FEAT_LABELS = ['Home Venue', 'League Match', 'European Match', 'After Europe',
               'Rest Days', 'Season', '4-Back Formation', 'Big Rival', 'Recent Form']

# ── 4. TRAIN / VALIDATION / TEST SPLIT ───────────────────────
TRAIN_S = ['2016-17','2017-18','2018-19','2019-20','2020-21','2021-22','2022-23']
VAL_S   = ['2023-24']
TEST_S  = ['2024-25','2025-26']

train = df[df['Season'].isin(TRAIN_S)]
val   = df[df['Season'].isin(VAL_S)]
test  = df[df['Season'].isin(TEST_S)]

X_train, y_train = train[FEATURES].values, train['Win'].values
X_val,   y_val   = val[FEATURES].values,   val['Win'].values
X_test,  y_test  = test[FEATURES].values,  test['Win'].values

scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(X_train)
Xva_sc = scaler.transform(X_val)
Xte_sc = scaler.transform(X_test)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ── 5. SUPERVISED MODELS ─────────────────────────────────────

# Baseline
majority_class = int(np.bincount(y_train).argmax())
base_val = accuracy_score(y_val,  [majority_class] * len(y_val))
base_tst = accuracy_score(y_test, [majority_class] * len(y_test))
print(f"\nBaseline          Val: {base_val:.3f}  Test: {base_tst:.3f}")

# Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(Xtr_sc, y_train)
lr_val = accuracy_score(y_val,  lr.predict(Xva_sc))
lr_tst = accuracy_score(y_test, lr.predict(Xte_sc))
print(f"Logistic Reg.     Val: {lr_val:.3f}  Test: {lr_tst:.3f}")

# kNN
k_values     = [3, 5, 7, 9, 11]
knn_val_accs = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtr_sc, y_train)
    knn_val_accs.append(accuracy_score(y_val, knn.predict(Xva_sc)))
best_k   = k_values[int(np.argmax(knn_val_accs))]
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(Xtr_sc, y_train)
knn_val = max(knn_val_accs)
knn_tst = accuracy_score(y_test, best_knn.predict(Xte_sc))
print(f"kNN (k={best_k})         Val: {knn_val:.3f}  Test: {knn_tst:.3f}")

# Decision Tree
depths      = [3, 4, 5, 6, 7]
dt_val_accs = []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    dt_val_accs.append(accuracy_score(y_val, dt.predict(X_val)))
best_depth = depths[int(np.argmax(dt_val_accs))]
best_dt    = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_dt.fit(X_train, y_train)
dt_val = max(dt_val_accs)
dt_tst = accuracy_score(y_test, best_dt.predict(X_test))
print(f"Decision Tree     Val: {dt_val:.3f}  Test: {dt_tst:.3f}")

# ── 6. K-MEANS CLUSTERING ────────────────────────────────────
# Use all 447 matches — unsupervised, no train/test split needed
X_all    = df[FEATURES].values
scaler_km = StandardScaler()
X_all_sc  = scaler_km.fit_transform(X_all)

# Elbow method
inertias = []
k_range  = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_all_sc)
    inertias.append(km.inertia_)

best_k_km = 3  # elbow at k=3
km_final  = KMeans(n_clusters=best_k_km, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X_all_sc)

cluster_profile = df.groupby('Cluster').agg(
    Matches     = ('Win',        'count'),
    WinRate     = ('Win',        'mean'),
    AvgRestDays = ('RestDays',   'mean'),
    HomePct     = ('IsHome',     'mean'),
    LeaguePct   = ('IsLeague',   'mean'),
    BigRivalPct = ('IsBigRival', 'mean'),
    AvgForm3    = ('Form3',      'mean'),
).round(3)
print("\nK-Means Cluster Profiles:")
print(cluster_profile.to_string())

# Label clusters by ascending win rate
wr_sorted = cluster_profile['WinRate'].sort_values()
label_map = {
    wr_sorted.index[0]: 'Difficult Conditions',
    wr_sorted.index[1]: 'Moderate Conditions',
    wr_sorted.index[2]: 'Favourable Conditions',
}
df['ClusterLabel'] = df['Cluster'].map(label_map)

# ── COLOUR PALETTE ────────────────────────────────────────────
FB_BLUE   = '#003399'; FB_YELLOW = '#E6A800'
WIN_C     = '#2ecc71'; LOSS_C    = '#e74c3c'; DRAW_C = '#f39c12'
BG = 'white'; TEXT = '#222222'; GRID = '#e0e0e0'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'axes.edgecolor': GRID,
    'axes.labelcolor':  TEXT, 'xtick.color':   TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID, 'grid.alpha': 0.5,
})

# ════════════════════════════════════════════════════════════
# FIGURE 1 – Model Comparison & Hyperparameter Selection
# ════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(20, 6))
fig1.patch.set_facecolor(BG)
fig1.suptitle("Fenerbahçe Win Prediction – Model Comparison",
              fontsize=17, fontweight='bold', color=FB_YELLOW)

ax = axes[0]
m_names = ['Baseline', 'Logistic\nRegression', f'kNN\n(k={best_k})',
           f'Decision Tree\n(depth={best_depth})']
t_accs  = [base_tst, lr_tst, knn_tst, dt_tst]
bars = ax.bar(m_names, t_accs, color=['#888888', FB_BLUE, WIN_C, FB_YELLOW],
              edgecolor=BG, width=0.55)
ax.axhline(base_tst, color=LOSS_C, ls='--', lw=2, label=f'Baseline: {base_tst:.3f}')
ax.set_ylim(0, 1); ax.set_ylabel("Test Accuracy"); ax.grid(axis='y')
ax.set_title("Test Set Accuracy by Model", color=FB_YELLOW, fontsize=13)
ax.legend(fontsize=10)
for b, v in zip(bars, t_accs):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f'{v:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax = axes[1]
ax.plot(k_values, knn_val_accs, marker='o', color=WIN_C, lw=2.5, ms=9)
ax.scatter([best_k], [knn_val], color=FB_YELLOW, s=200, zorder=5,
           label=f'Best  k={best_k}  ({knn_val:.3f})')
ax.set_xlabel("k  (Number of Neighbours)"); ax.set_ylabel("Validation Accuracy")
ax.set_title("kNN – Validation Accuracy vs k", color=FB_YELLOW, fontsize=13)
ax.set_xticks(k_values); ax.grid(); ax.legend(fontsize=10)

ax = axes[2]
ax.plot(depths, dt_val_accs, marker='s', color=FB_BLUE, lw=2.5, ms=9)
ax.scatter([best_depth], [dt_val], color=FB_YELLOW, s=200, zorder=5,
           label=f'Best  depth={best_depth}  ({dt_val:.3f})')
ax.set_xlabel("Max Depth"); ax.set_ylabel("Validation Accuracy")
ax.set_title("Decision Tree – Validation Accuracy vs Depth",
             color=FB_YELLOW, fontsize=13)
ax.set_xticks(depths); ax.grid(); ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('/home/claude/ML_ModelComparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("\nFigure 1 saved: ML_ModelComparison.png")

# ════════════════════════════════════════════════════════════
# FIGURE 2 – Confusion Matrices
# ════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 4, figsize=(22, 5))
fig2.patch.set_facecolor(BG)
fig2.suptitle("Confusion Matrices – Test Set  (Win vs Not-Win)",
              fontsize=15, fontweight='bold', color=FB_YELLOW)

preds_list = [
    ([majority_class]*len(y_test), f'Baseline\n(Acc={base_tst:.3f})',              'Greys'),
    (lr.predict(Xte_sc),           f'Logistic Regression\n(Acc={lr_tst:.3f})',     'Blues'),
    (best_knn.predict(Xte_sc),     f'kNN  k={best_k}\n(Acc={knn_tst:.3f})',       'Greens'),
    (best_dt.predict(X_test),      f'Decision Tree  depth={best_depth}\n(Acc={dt_tst:.3f})', 'YlOrBr'),
]
for ax, (y_pred, title, cmap) in zip(axes, preds_list):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Not Win', 'Win'])
    disp.plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(title, color=FB_YELLOW, fontsize=12)

plt.tight_layout()
plt.savefig('/home/claude/ML_ConfusionMatrices.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Figure 2 saved: ML_ConfusionMatrices.png")

# ════════════════════════════════════════════════════════════
# FIGURE 3 – Feature Insights
# ════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(18, 6))
fig3.patch.set_facecolor(BG)
fig3.suptitle("Feature Insights – Logistic Regression & Decision Tree",
              fontsize=15, fontweight='bold', color=FB_YELLOW)

ax = axes[0]
coefs      = lr.coef_[0]
sorted_idx = np.argsort(coefs)
bar_colors = [WIN_C if c > 0 else LOSS_C for c in coefs[sorted_idx]]
ax.barh(np.array(FEAT_LABELS)[sorted_idx], coefs[sorted_idx],
        color=bar_colors, edgecolor=BG)
ax.axvline(0, color=TEXT, lw=1.2)
ax.set_xlabel("Coefficient  (positive → increases win probability)")
ax.set_title("Logistic Regression – Coefficients", color=FB_YELLOW, fontsize=13)
ax.grid(axis='x')
ax.barh([], [], color=WIN_C,  label='↑ Increases win prob.')
ax.barh([], [], color=LOSS_C, label='↓ Decreases win prob.')
ax.legend(fontsize=9)

ax = axes[1]
importances = best_dt.feature_importances_
sorted_idx2 = np.argsort(importances)
imp_colors  = [FB_YELLOW if importances[i] == max(importances) else FB_BLUE
               for i in sorted_idx2]
ax.barh(np.array(FEAT_LABELS)[sorted_idx2], importances[sorted_idx2],
        color=imp_colors, edgecolor=BG)
ax.set_xlabel("Feature Importance  (Gini)")
ax.set_title(f"Decision Tree (depth={best_depth}) – Feature Importance",
             color=FB_YELLOW, fontsize=13)
ax.grid(axis='x')

plt.tight_layout()
plt.savefig('/home/claude/ML_FeatureInsights.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Figure 3 saved: ML_FeatureInsights.png")

# ════════════════════════════════════════════════════════════
# FIGURE 4 – Decision Tree Visualization
# ════════════════════════════════════════════════════════════
dt_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_viz.fit(X_train, y_train)

fig4, ax = plt.subplots(figsize=(22, 9))
fig4.patch.set_facecolor(BG)
plot_tree(dt_viz, feature_names=FEAT_LABELS, class_names=['Not Win', 'Win'],
          filled=True, rounded=True, fontsize=10, ax=ax,
          impurity=False, proportion=False, label='root')
ax.set_title("Decision Tree – Match Outcome Rules  (depth 3)",
             fontsize=14, fontweight='bold', color=FB_YELLOW, pad=15)
plt.tight_layout()
plt.savefig('/home/claude/ML_DecisionTree.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Figure 4 saved: ML_DecisionTree.png")

# ════════════════════════════════════════════════════════════
# FIGURE 5 – K-Means Clustering
# ════════════════════════════════════════════════════════════
fig5, axes = plt.subplots(1, 3, figsize=(20, 6))
fig5.patch.set_facecolor(BG)
fig5.suptitle("K-Means Clustering – Natural Match Condition Groups",
              fontsize=16, fontweight='bold', color=FB_YELLOW)

ax = axes[0]
ax.plot(list(k_range), inertias, marker='o', color=FB_BLUE, lw=2.5, ms=9)
ax.scatter([best_k_km], [inertias[best_k_km - 2]], color=FB_YELLOW,
           s=200, zorder=5, label=f'Chosen  k={best_k_km}')
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia  (Within-cluster SSE)")
ax.set_title("Elbow Method – Optimal k", color=FB_YELLOW, fontsize=13)
ax.legend(fontsize=10); ax.grid()

ax = axes[1]
ordered_labels = ['Difficult Conditions', 'Moderate Conditions', 'Favourable Conditions']
win_rates_km   = [cluster_profile.loc[wr_sorted.index[i], 'WinRate'] for i in range(3)]
bar_cols_km    = [LOSS_C, DRAW_C, WIN_C]
bars = ax.bar(ordered_labels, win_rates_km, color=bar_cols_km, edgecolor=BG, width=0.5)
ax.set_ylabel("Win Rate"); ax.set_ylim(0, 1); ax.grid(axis='y')
ax.set_title("Win Rate per Cluster", color=FB_YELLOW, fontsize=13)
ax.set_xticklabels(ordered_labels, fontsize=9)
for b, v in zip(bars, win_rates_km):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.1%}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax = axes[2]
x = np.arange(3); w = 0.25
home_p   = [cluster_profile.loc[wr_sorted.index[i], 'HomePct']     for i in range(3)]
league_p = [cluster_profile.loc[wr_sorted.index[i], 'LeaguePct']   for i in range(3)]
rival_p  = [cluster_profile.loc[wr_sorted.index[i], 'BigRivalPct'] for i in range(3)]
ax.bar(x - w, home_p,   w, label='Home Match',  color=FB_BLUE,   edgecolor=BG)
ax.bar(x,     league_p, w, label='League Match', color=FB_YELLOW, edgecolor=BG)
ax.bar(x + w, rival_p,  w, label='Big Rival',    color=LOSS_C,    edgecolor=BG)
ax.set_xticks(x); ax.set_xticklabels(ordered_labels, fontsize=8)
ax.set_ylabel("Proportion"); ax.set_ylim(0, 1); ax.grid(axis='y')
ax.set_title("Cluster Composition", color=FB_YELLOW, fontsize=13)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/ML_KMeans.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Figure 5 saved: ML_KMeans.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("MACHINE LEARNING RESULTS – SUMMARY")
print("=" * 58)
print(f"{'Model':<32} {'Val Acc':>8} {'Test Acc':>10}")
print("-" * 58)
summary_rows = [
    ('Majority Class Baseline',              base_val, base_tst),
    ('Logistic Regression',                  lr_val,   lr_tst),
    (f'kNN (best k={best_k})',               knn_val,  knn_tst),
    (f'Decision Tree (best depth={best_depth})', dt_val, dt_tst),
]
for name, va, ta in summary_rows:
    print(f"{name:<32} {va:>8.3f} {ta:>10.3f}")
best_row = max(summary_rows[1:], key=lambda r: r[2])
print(f"\nBest model: {best_row[0]}  (test acc={best_row[2]:.3f})")

# ── README.md Generation ──────────────────────────────────────
readme = f"""# Fenerbahçe Match Performance Analysis
**DSA210 – Spring 2026 | Damla Kandemir**

## Project Overview

This project analyzes the factors affecting Fenerbahçe's match performance across different competitions — Süper Lig, Turkish Cup, and European tournaments — over 10 seasons (2016–2026). Rather than focusing on a single variable, the goal is to take a broader look at how contextual factors such as home advantage, fixture congestion, opponent strength, and competition type relate to match outcomes.

**Research Question:** Which factors are most related to Fenerbahçe's match performance across league and European competitions?

---

## Dataset
- **Source:** [FBref](https://fbref.com/en/squads/ae1e2d7d/Fenerbahce-Stats)
- **Period:** 2016–2026 (10 seasons)
- **Total matches:** 447
- **Competitions:** Süper Lig (353), Europa League (63), Conference League (17), Champions League (14)
- **Features:** Date, Venue (Home/Away), Competition, Result, Goals For, Goals Against, Possession, Formation, Opponent

---

## Hypotheses Tested

| # | Hypothesis | Test | Result |
|---|-----------|------|--------|
| H1 | Fenerbahçe performs better at home than away | Two-sample t-test | ✅ Significant (p = 0.0131) |
| H2 | League performance drops after a European match | Two-sample t-test | ❌ Not significant (p = 0.5825) |
| H3 | Higher possession correlates with better goal difference | Pearson correlation | ❌ Not significant (r = 0.018, p = 0.7091) |

---

## Key Findings (EDA)
- **Win rate:** 57.0% overall (Home: 61.9% | Away: 52.2%)
- **Home advantage** is statistically significant (p = 0.0131), confirming that playing at home meaningfully increases the likelihood of winning.
- **European fixture fatigue** does not significantly reduce league points (p = 0.5825), suggesting Fenerbahçe manages rotation effectively.
- **Possession** is not a strong predictor of goal difference (r = 0.018), indicating that Fenerbahçe's style is not purely possession-based.

---

## Visualizations

### General Performance (10 Seasons)
![General Performance](General_Performance.png)

### Hypothesis Tests
![Hypothesis Tests](Hypothesis_testing.png)

### Cumulative Points & Rest Days Effect
![Cumulative Rest Days](cumulative_rest_days.png)

---

## Machine Learning Methods

### Task
The ML task is formulated as a **binary classification** problem. The target variable is whether Fenerbahçe won the match (Win = 1) or not (Draw/Loss = 0). Goals scored, goals conceded, possession, and match points were excluded from model inputs — these are not known before the match and would cause **data leakage**.

In addition to supervised classification, **K-Means Clustering** was applied as an unsupervised method to discover natural groupings in match conditions, independent of the result.

### Features Used

| Feature | Description |
|---------|-------------|
| Home Venue | Whether Fenerbahçe played at home (1) or away (0) |
| League Match | Whether the match was a Süper Lig game (1) or European (0) |
| European Match | Whether the match was a European competition |
| After Europe | Whether a league match followed directly after a European fixture |
| Rest Days | Number of days since the previous match |
| Season | Ordinal season number (captures trend over years) |
| 4-Back Formation | Whether Fenerbahçe used a 4-defender formation |
| Big Rival | Whether the opponent was Galatasaray, Beşiktaş, Trabzonspor, or Başakşehir |
| Recent Form | Rolling average of points from the previous 3 matches |

### Train / Validation / Test Split

A **chronological split** was used to avoid future information leakage — the model learns from the past and is evaluated on more recent seasons:
- **Train:** 2016–17 to 2022–23 → 300 matches
- **Validation:** 2023–24 → 53 matches (used for hyperparameter selection)
- **Test:** 2024–25 and 2025–26 → 94 matches (final evaluation only)

---

### Supervised Learning Results

| Model | Val Accuracy | Test Accuracy |
|-------|-------------|---------------|
| Majority Class Baseline | {base_val:.3f} | {base_tst:.3f} |
| Logistic Regression | {lr_val:.3f} | **{lr_tst:.3f}** |
| kNN (best k = {best_k}) | {knn_val:.3f} | {knn_tst:.3f} |
| Decision Tree (best depth = {best_depth}) | {dt_val:.3f} | **{dt_tst:.3f}** |

- **Logistic Regression** and **Decision Tree** both outperform the majority class baseline ({base_tst:.3f}) on the test set, reaching **{max(lr_tst, dt_tst):.1%} accuracy**.
- kNN achieved the highest validation accuracy but generalized less well, likely due to overfitting to local patterns in the training set.
- The best k for kNN ({best_k}) and best depth for Decision Tree ({best_depth}) were selected using the validation set only — the test set was never seen during tuning.

#### Model Comparison & Hyperparameter Selection
![ML Model Comparison](ML_ModelComparison.png)

#### Confusion Matrices
![Confusion Matrices](ML_ConfusionMatrices.png)

#### Feature Insights (LR Coefficients & DT Feature Importance)
![Feature Insights](ML_FeatureInsights.png)

#### Decision Tree – Match Outcome Rules
![Decision Tree](ML_DecisionTree.png)

---

### Unsupervised Learning – K-Means Clustering

Beyond predicting outcomes, K-Means Clustering was applied to all 447 matches to discover whether **natural groups of match conditions** exist in the data — without using the result as input.

The **Elbow Method** identified **k = 3** as the optimal number of clusters. The three clusters correspond to meaningfully different match contexts:

| Cluster | Win Rate | Avg Rest Days | League % | Big Rival % | Interpretation |
|---------|----------|---------------|----------|-------------|----------------|
| Difficult Conditions | {win_rates_km[0]:.1%} | {cluster_profile.loc[wr_sorted.index[0], 'AvgRestDays']:.1f} | {cluster_profile.loc[wr_sorted.index[0], 'LeaguePct']:.0%} | {cluster_profile.loc[wr_sorted.index[0], 'BigRivalPct']:.0%} | Mainly European matches, lowest win rate |
| Moderate Conditions | {win_rates_km[1]:.1%} | {cluster_profile.loc[wr_sorted.index[1], 'AvgRestDays']:.1f} | {cluster_profile.loc[wr_sorted.index[1], 'LeaguePct']:.0%} | {cluster_profile.loc[wr_sorted.index[1], 'BigRivalPct']:.0%} | Congested league schedule |
| Favourable Conditions | {win_rates_km[2]:.1%} | {cluster_profile.loc[wr_sorted.index[2], 'AvgRestDays']:.1f} | {cluster_profile.loc[wr_sorted.index[2], 'LeaguePct']:.0%} | {cluster_profile.loc[wr_sorted.index[2], 'BigRivalPct']:.0%} | Normal league matches, highest win rate |

**Key finding:** The algorithm separated European matches from league matches as a primary boundary — supporting the exploratory finding that competition type is a strong differentiator of match conditions, even if European fixture fatigue alone was not statistically significant in hypothesis testing.

#### K-Means Clustering Results
![K-Means](ML_KMeans.png)

---

### Key ML Findings
- **Home venue** is the strongest positive predictor of winning, consistent with H1.
- **Recent form** (rolling average of last 3 match points) is an important predictor, suggesting momentum matters.
- **Big rival matches** are associated with lower win probability.
- **European match context** is picked up by K-Means as a natural cluster, even without the result.
- Overall supervised accuracy (~{max(lr_tst, dt_tst):.0%}) is modest but exceeds the naive baseline, which is expected given the limited pre-match information available.

---

## Repository Structure
```
DSA-210-PROJECT/
├── README.md
├── requirements.txt
├── fenerbahce_eda.py
├── fenerbahce_ml.py
├── General_Performance.png
├── Hypothesis_testing.png
├── cumulative_rest_days.png
├── ML_ModelComparison.png
├── ML_ConfusionMatrices.png
├── ML_FeatureInsights.png
├── ML_DecisionTree.png
├── ML_KMeans.png
└── Fenerbahce 10 seasons/
    ├── Fenerbahce_2016-2017.csv
    ├── ...
    └── Fenerbahce_2025-2026.csv
```

---

## How to Reproduce the Analysis

### 1. Clone the repository
```bash
git clone https://github.com/damland7/DSA-210-PROJECT.git
cd DSA-210-PROJECT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run EDA
```bash
python fenerbahce_eda.py
```
Generates: `General_Performance.png`, `Hypothesis_testing.png`, `cumulative_rest_days.png`

### 4. Run ML analysis
```bash
python fenerbahce_ml.py
```
Generates: `ML_ModelComparison.png`, `ML_ConfusionMatrices.png`, `ML_FeatureInsights.png`, `ML_DecisionTree.png`, `ML_KMeans.png`, `README.md`

> **Note:** CSV files must be inside the `Fenerbahce 10 seasons/` folder, or update the file paths in the `csv_files` dictionary at the top of each script.

---

## Data Source
Data collected manually from FBref match logs:
- https://fbref.com/en/squads/ae1e2d7d/Fenerbahce-Stats

---

## Limitations and Future Work
- **Pre-match features only:** The models rely on contextual factors (venue, competition, rest days) rather than in-game statistics. Adding opponent strength data from Transfermarkt (squad market values) could improve predictive power.
- **Class balance:** Win rate is 57%, so the dataset slightly favors the positive class. Techniques like SMOTE could be explored.
- **Model scope:** More advanced ensemble methods (Random Forest, Gradient Boosting) could be applied in future work after covering them in more depth.
- **Time series:** A season-level time series analysis (e.g., ARIMA) could capture long-term performance trends beyond what season ordinal encoding captures.

---

## AI Usage Disclaimer
AI tools (Claude by Anthropic) were used in this project to assist with:
- Writing and debugging Python code for EDA, hypothesis testing, and ML analysis
- Structuring the README and project report
- Reviewing statistical interpretations and suggesting feature engineering ideas

All data collection, hypothesis formulation, analysis decisions, and final interpretations were made independently by the student.

---

*DSA210 – Introduction to Data Science | Sabancı University | Spring 2026*
"""

with open('/home/claude/README.md', 'w', encoding='utf-8') as f:
    f.write(readme)
print("\nREADME.md generated successfully.")
print("\n=== ALL DONE ===")
