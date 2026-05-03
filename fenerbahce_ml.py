# ============================================================
# Fenerbahçe Match Performance – Machine Learning Methods
# DSA210 | Damla Kandemir
# ============================================================
#
# Task   : Binary classification → Win (1) vs Not-Win (0)
# Models : Majority Class Baseline, Logistic Regression,
#          k-Nearest Neighbors, Decision Tree
# Split  : Chronological (train 2016-23 / val 2023-24 / test 2024-26)
# Note   : GF, GA, GoalDiff, Points, Possession are intentionally
#          excluded — they are not known before the match (data leakage).
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
# Target variable
df['Win'] = (df['Result'] == 'W').astype(int)

# Venue: Home = 1, Away = 0
df['IsHome'] = (df['Venue'] == 'Home').astype(int)

# Competition flags
df['IsLeague']   = (df['Comp'] == 'Süper Lig').astype(int)
df['IsEuropean'] = df['Comp'].isin(['Europa Lg', 'Champions Lg', 'Conf Lg']).astype(int)
df['CompGroup']  = df['IsLeague']  # 1 = League, 0 = Europe/Other

# Rest days between consecutive matches (capped at 30)
df['RestDays'] = df['Date'].diff().dt.days.clip(0, 30).fillna(7)

# Flag: league match played directly after a European match
df['AfterEurope'] = False
for i in range(1, len(df)):
    if df.loc[i, 'IsLeague'] and df.loc[i - 1, 'IsEuropean']:
        df.loc[i, 'AfterEurope'] = True
df['AfterEurope'] = df['AfterEurope'].astype(int)

# Season as ordinal number (captures performance trend over years)
season_order = {s: i for i, s in enumerate(csv_files.keys())}
df['SeasonNum'] = df['Season'].map(season_order)

# Formation: 4-back line (1) vs 3-back line (0)
df['Is4Back'] = df['Formation'].str[0].map({'4': 1, '3': 0}).fillna(0).astype(int)

# Big rival flag (main title challengers in Süper Lig)
BIG_RIVALS = {'Galatasaray', 'Beşiktaş', 'Trabzonspor', 'Başakşehir'}
df['IsBigRival'] = df['Opponent'].isin(BIG_RIVALS).astype(int)

# Recent form: rolling average of points from the previous 3 matches
# shift(1) ensures the current match result is never used → no leakage
df['Points'] = df['Result'].map({'W': 3, 'D': 1, 'L': 0})
df['Form3']  = df['Points'].shift(1).rolling(3, min_periods=1).mean().fillna(1.5)

print(f"Total matches : {len(df)}")
print(f"Win rate      : {df['Win'].mean():.1%}")
print(f"Seasons       : {df['Season'].nunique()}")

# ── 4. FEATURES ──────────────────────────────────────────────
# Pre-match only — GF, GA, Possession, GoalDiff, Points excluded (leakage)
FEATURES = ['IsHome', 'CompGroup', 'IsEuropean', 'AfterEurope',
            'RestDays', 'SeasonNum', 'Is4Back', 'IsBigRival', 'Form3']
FEAT_LABELS = ['Home Venue', 'League Match', 'European Match', 'After Europe',
               'Rest Days', 'Season', '4-Back Formation', 'Big Rival', 'Recent Form']

# ── 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT ─────────
TRAIN_S = ['2016-17','2017-18','2018-19','2019-20','2020-21','2021-22','2022-23']
VAL_S   = ['2023-24']
TEST_S  = ['2024-25','2025-26']

train = df[df['Season'].isin(TRAIN_S)]
val   = df[df['Season'].isin(VAL_S)]
test  = df[df['Season'].isin(TEST_S)]

X_train, y_train = train[FEATURES].values, train['Win'].values
X_val,   y_val   = val[FEATURES].values,   val['Win'].values
X_test,  y_test  = test[FEATURES].values,  test['Win'].values

print(f"\nTrain : {len(X_train)} matches | "
      f"Val : {len(X_val)} matches | "
      f"Test : {len(X_test)} matches")

# Scale for LR and kNN (not required for Decision Tree)
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(X_train)  # fit only on train → no leakage
Xva_sc = scaler.transform(X_val)
Xte_sc = scaler.transform(X_test)

# ════════════════════════════════════════════════════════════
# MODEL 1 — Majority Class Baseline
# ════════════════════════════════════════════════════════════
majority_class   = int(np.bincount(y_train).argmax())
base_val = accuracy_score(y_val,  [majority_class] * len(y_val))
base_tst = accuracy_score(y_test, [majority_class] * len(y_test))
print(f"\nBaseline (always predict {majority_class}) "
      f"Val: {base_val:.3f}  Test: {base_tst:.3f}")

# ════════════════════════════════════════════════════════════
# MODEL 2 — Logistic Regression
# ════════════════════════════════════════════════════════════
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(Xtr_sc, y_train)
lr_val = accuracy_score(y_val,  lr.predict(Xva_sc))
lr_tst = accuracy_score(y_test, lr.predict(Xte_sc))
print(f"Logistic Regression           Val: {lr_val:.3f}  Test: {lr_tst:.3f}")

# ════════════════════════════════════════════════════════════
# MODEL 3 — kNN  (best k chosen on validation set)
# ════════════════════════════════════════════════════════════
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
print(f"kNN (best k={best_k})                 Val: {knn_val:.3f}  Test: {knn_tst:.3f}")

# ════════════════════════════════════════════════════════════
# MODEL 4 — Decision Tree  (best depth chosen on validation set)
# ════════════════════════════════════════════════════════════
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
print(f"Decision Tree (depth={best_depth})        Val: {dt_val:.3f}  Test: {dt_tst:.3f}")

# ── Colour palette (matches EDA style) ───────────────────────
FB_BLUE  = '#003399'; FB_YELLOW = '#E6A800'
WIN_C    = '#2ecc71'; LOSS_C    = '#e74c3c'
BG = 'white'; TEXT = '#222222'; GRID = '#e0e0e0'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'axes.edgecolor': GRID,
    'axes.labelcolor':  TEXT, 'xtick.color':  TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID, 'grid.alpha': 0.5,
})

# ════════════════════════════════════════════════════════════
# FIGURE 1 – Model Comparison & Hyperparameter Selection
# ════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(20, 6))
fig1.patch.set_facecolor(BG)
fig1.suptitle("Fenerbahçe Win Prediction – Model Comparison",
              fontsize=17, fontweight='bold', color=FB_YELLOW)

# 1a — Test accuracy bar chart
ax = axes[0]
m_names  = ['Baseline', 'Logistic\nRegression', f'kNN\n(k={best_k})',
             f'Decision Tree\n(depth={best_depth})']
t_accs   = [base_tst, lr_tst, knn_tst, dt_tst]
bar_cols = ['#888888', FB_BLUE, WIN_C, FB_YELLOW]
bars = ax.bar(m_names, t_accs, color=bar_cols, edgecolor=BG, width=0.55)
ax.axhline(base_tst, color=LOSS_C, ls='--', lw=2,
           label=f'Baseline: {base_tst:.3f}')
ax.set_ylim(0, 1); ax.set_ylabel("Test Accuracy"); ax.grid(axis='y')
ax.set_title("Test Set Accuracy by Model", color=FB_YELLOW, fontsize=13)
ax.legend(fontsize=10)
for b, v in zip(bars, t_accs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# 1b — kNN: validation accuracy vs k
ax = axes[1]
ax.plot(k_values, knn_val_accs, marker='o', color=WIN_C, lw=2.5, ms=9)
ax.scatter([best_k], [knn_val], color=FB_YELLOW, s=200, zorder=5,
           label=f'Best  k={best_k}  ({knn_val:.3f})')
ax.set_xlabel("k  (Number of Neighbours)"); ax.set_ylabel("Validation Accuracy")
ax.set_title("kNN – Validation Accuracy vs k", color=FB_YELLOW, fontsize=13)
ax.set_xticks(k_values); ax.grid(); ax.legend(fontsize=10)

# 1c — Decision Tree: validation accuracy vs max_depth
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
# FIGURE 2 – Confusion Matrices (all 4 models)
# ════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 4, figsize=(22, 5))
fig2.patch.set_facecolor(BG)
fig2.suptitle("Confusion Matrices – Test Set  (Win vs Not-Win)",
              fontsize=15, fontweight='bold', color=FB_YELLOW)

preds_list = [
    ([majority_class] * len(y_test),  f'Baseline\n(Acc={base_tst:.3f})', 'Greys'),
    (lr.predict(Xte_sc),              f'Logistic Regression\n(Acc={lr_tst:.3f})', 'Blues'),
    (best_knn.predict(Xte_sc),        f'kNN  k={best_k}\n(Acc={knn_tst:.3f})', 'Greens'),
    (best_dt.predict(X_test),         f'Decision Tree  depth={best_depth}\n(Acc={dt_tst:.3f})', 'YlOrBr'),
]
for ax, (y_pred, title, cmap) in zip(axes, preds_list):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Not Win', 'Win'])
    disp.plot(ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(title, color=FB_YELLOW, fontsize=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig('/home/claude/ML_ConfusionMatrices.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Figure 2 saved: ML_ConfusionMatrices.png")

# ════════════════════════════════════════════════════════════
# FIGURE 3 – Feature Insights (LR Coefficients + DT Importance)
# ════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(18, 6))
fig3.patch.set_facecolor(BG)
fig3.suptitle("Feature Insights – Logistic Regression & Decision Tree",
              fontsize=15, fontweight='bold', color=FB_YELLOW)

# 3a — Logistic Regression coefficients
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

# 3b — Decision Tree feature importance
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
# FIGURE 4 – Decision Tree Visualization (depth=3 for readability)
# ════════════════════════════════════════════════════════════
dt_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_viz.fit(X_train, y_train)

fig4, ax = plt.subplots(figsize=(22, 9))
fig4.patch.set_facecolor(BG)
plot_tree(dt_viz,
          feature_names=FEAT_LABELS,
          class_names=['Not Win', 'Win'],
          filled=True, rounded=True, fontsize=10, ax=ax,
          impurity=False, proportion=False, label='root')
ax.set_title("Decision Tree – Match Outcome Rules  (visualized at depth 3)",
             fontsize=14, fontweight='bold', color=FB_YELLOW, pad=15)
plt.tight_layout()
plt.savefig('/home/claude/ML_DecisionTree.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Figure 4 saved: ML_DecisionTree.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MACHINE LEARNING RESULTS – SUMMARY")
print("=" * 60)
print(f"{'Model':<32} {'Val Acc':>8} {'Test Acc':>10}")
print("-" * 60)
summary_rows = [
    ('Majority Class Baseline',            base_val, base_tst),
    ('Logistic Regression',                lr_val,   lr_tst),
    (f'kNN (best k={best_k})',             knn_val,  knn_tst),
    (f'Decision Tree (best depth={best_depth})', dt_val, dt_tst),
]
for name, va, ta in summary_rows:
    print(f"{name:<32} {va:>8.3f} {ta:>10.3f}")

best_row = max(summary_rows[1:], key=lambda r: r[2])
print(f"\nBest model on test set: {best_row[0]}  (acc={best_row[2]:.3f})")

print("\nClassification Report – Logistic Regression (test set):")
print(classification_report(y_test, lr.predict(Xte_sc),
                             target_names=['Not Win', 'Win']))

print("\n=== ML ANALYSIS COMPLETE ===")
