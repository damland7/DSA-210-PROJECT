# ============================================================
# Fenerbahçe Match Performance – EDA & Hypothesis Tests
# DSA210 | Damla Kandemir
# ============================================================

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 1. DATA LOADING ──────────────────────────────────────────
# Place all CSV files in the same folder as this script
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

# ── 2. DATA CLEANING ─────────────────────────────────────────
df = df[df['Result'].isin(['W','D','L'])].copy()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date','Result','GF','GA','Venue','Comp'])
df['GF']   = pd.to_numeric(df['GF'],   errors='coerce')
df['GA']   = pd.to_numeric(df['GA'],   errors='coerce')
df['Poss'] = pd.to_numeric(df['Poss'], errors='coerce')
df = df.dropna(subset=['GF','GA'])
df = df.sort_values('Date').reset_index(drop=True)

# ── 3. FEATURE ENGINEERING ───────────────────────────────────
df['Points']     = df['Result'].map({'W':3,'D':1,'L':0})
df['GoalDiff']   = df['GF'] - df['GA']
df['IsEuropean'] = df['Comp'].isin(['Europa Lg','Champions Lg','Conf Lg'])
df['IsLeague']   = df['Comp'] == 'Süper Lig'
df['RestDays']   = df['Date'].diff().dt.days.clip(0, 30)
df['CompGroup']  = df['Comp'].map(
    {'Süper Lig':'Süper Lig','Europa Lg':'Europe','Champions Lg':'Europe','Conf Lg':'Europe'}
).fillna('Other')

# Flag: league match played right after a European match
df['AfterEurope'] = False
for i in range(1, len(df)):
    if df.loc[i,'IsLeague'] and df.loc[i-1,'IsEuropean']:
        df.loc[i,'AfterEurope'] = True

print(f"Total matches: {len(df)}  |  Seasons: {df['Season'].nunique()}")
print(df['Result'].value_counts())

# ── 4. COLOR PALETTE ─────────────────────────────────────────
FB_BLUE='#003399'; FB_YELLOW='#E6A800'
WIN_C='#2ecc71';   DRAW_C='#f39c12'; LOSS_C='#e74c3c'
BG='white'; TEXT='#222222'; GRID='#e0e0e0'

plt.rcParams.update({
    'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':GRID,
    'axes.labelcolor':TEXT,'xtick.color':TEXT,'ytick.color':TEXT,
    'text.color':TEXT,'grid.color':GRID,'grid.alpha':0.5,
})

seasons = list(csv_files.keys())

# ════════════════════════════════════════════════════════════
# FIGURE 1 – General Performance Overview
# ════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
fig1.patch.set_facecolor(BG)
fig1.suptitle("Fenerbahçe Match Performance – 10 Seasons (2016–2026)",
              fontsize=18, fontweight='bold', color=FB_YELLOW, y=1.01)

# 1a – Win/Draw/Loss pie chart
ax = axes[0,0]
wdl = df['Result'].value_counts()[['W','D','L']]
ax.pie(wdl, labels=['Win','Draw','Loss'],
       colors=[WIN_C,DRAW_C,LOSS_C], autopct='%1.1f%%', startangle=90,
       wedgeprops=dict(width=0.55,edgecolor=BG,linewidth=2),
       textprops={'color':TEXT,'fontsize':11})
ax.set_title("Overall Result Distribution", color=FB_YELLOW, fontsize=13, pad=10)

# 1b – Average points per season
ax = axes[0,1]
sp = df.groupby('Season')['Points'].mean()
bars = ax.bar(range(len(sp)), sp.values, color=FB_BLUE, edgecolor=FB_YELLOW, linewidth=0.8)
ax.set_xticks(range(len(sp))); ax.set_xticklabels(sp.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Avg Points / Match")
ax.set_title("Average Points per Season", color=FB_YELLOW, fontsize=13)
ax.axhline(sp.mean(), color=FB_YELLOW, ls='--', lw=1.5, label=f'Avg: {sp.mean():.2f}')
ax.legend(fontsize=9); ax.grid(axis='y')
for b,v in zip(bars, sp.values):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

# 1c – Home vs Away win rate
ax = axes[0,2]
vwr = df.groupby('Venue').apply(lambda x:(x['Result']=='W').mean()*100).reindex(['Home','Away'])
b2 = ax.bar(['Home','Away'], vwr.values, color=[FB_YELLOW,FB_BLUE], edgecolor=TEXT, width=0.5)
ax.set_ylabel("Win Rate (%)"); ax.set_title("Home vs Away Win Rate", color=FB_YELLOW, fontsize=13)
ax.set_ylim(0,100); ax.grid(axis='y')
for b,v in zip(b2, vwr.values):
    ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 1d – Win rate by competition
ax = axes[1,0]
cgwr = df.groupby('CompGroup').apply(lambda x:(x['Result']=='W').mean()*100)
b3 = ax.bar(cgwr.index, cgwr.values, color=[FB_BLUE,FB_YELLOW,'#888'][:len(cgwr)], edgecolor=TEXT, width=0.5)
ax.set_ylabel("Win Rate (%)"); ax.set_title("Win Rate by Competition", color=FB_YELLOW, fontsize=13)
ax.set_ylim(0,100); ax.grid(axis='y')
for b,v in zip(b3, cgwr.values):
    ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 1e – Goals scored vs conceded per season
ax = axes[1,1]
gf_avg = df.groupby('Season')['GF'].mean()
ga_avg = df.groupby('Season')['GA'].mean()
x = np.arange(len(gf_avg)); w=0.35
ax.bar(x-w/2, gf_avg.values, w, label='Goals Scored',   color=WIN_C,  edgecolor=BG)
ax.bar(x+w/2, ga_avg.values, w, label='Goals Conceded', color=LOSS_C, edgecolor=BG)
ax.set_xticks(x); ax.set_xticklabels(gf_avg.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Avg Goals / Match"); ax.set_title("Goals Scored vs Conceded per Season", color=FB_YELLOW, fontsize=13)
ax.legend(fontsize=10); ax.grid(axis='y')

# 1f – Possession distribution
ax = axes[1,2]
pd_df = df.dropna(subset=['Poss'])
ax.hist(pd_df['Poss'], bins=20, color=FB_BLUE, edgecolor=FB_YELLOW, linewidth=0.7)
ax.axvline(pd_df['Poss'].mean(), color=FB_YELLOW, ls='--', lw=2, label=f"Avg: {pd_df['Poss'].mean():.1f}%")
ax.axvline(50, color='white', ls=':', lw=1.5, label='50%')
ax.set_xlabel("Possession (%)"); ax.set_ylabel("Number of Matches")
ax.set_title("Possession Distribution", color=FB_YELLOW, fontsize=13)
ax.legend(fontsize=10); ax.grid(axis='y')

plt.tight_layout()
plt.savefig('General_Performance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Figure 1 saved.")

# ════════════════════════════════════════════════════════════
# FIGURE 2 – Hypothesis Tests
# ════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 3, figsize=(18,6))
fig2.patch.set_facecolor(BG)
fig2.suptitle("Hypothesis Tests", fontsize=18, fontweight='bold', color=FB_YELLOW)

# H1 – Home vs Away points
ax = axes[0]
home_pts = df[df['Venue']=='Home']['Points']
away_pts = df[df['Venue']=='Away']['Points']
_, p1 = stats.ttest_ind(home_pts, away_pts, alternative='greater')
bp = ax.boxplot([home_pts.values, away_pts.values], patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor(FB_YELLOW); bp['boxes'][1].set_facecolor(FB_BLUE)
ax.set_xticklabels(['Home','Away'], fontsize=12)
ax.set_ylabel("Points (0/1/3)")
ax.set_title(f"H1: Home vs Away\np = {p1:.4f}  {'✓ Significant' if p1<0.05 else '✗ Not Significant'}",
             color=FB_YELLOW, fontsize=12)
ax.text(0.5,-0.15, f"Two-sample t-test  |  Home avg: {home_pts.mean():.2f}   Away avg: {away_pts.mean():.2f}",
        transform=ax.transAxes, ha='center', fontsize=9, color=TEXT)
ax.grid(axis='y')

# H2 – League performance after European match
ax = axes[1]
after_eu = df[(df['IsLeague']) & (df['AfterEurope'])]['Points']
normal   = df[(df['IsLeague']) & (~df['AfterEurope'])]['Points']
_, p2 = stats.ttest_ind(after_eu, normal, alternative='less')
bp2 = ax.boxplot([after_eu.values, normal.values], patch_artist=True, widths=0.5,
                  medianprops=dict(color='black', linewidth=2))
bp2['boxes'][0].set_facecolor(LOSS_C); bp2['boxes'][1].set_facecolor(WIN_C)
ax.set_xticklabels(['After\nEurope','Normal'], fontsize=11)
ax.set_ylabel("Points (0/1/3)")
ax.set_title(f"H2: European Fixture Effect\np = {p2:.4f}  {'✓ Significant' if p2<0.05 else '✗ Not Significant'}",
             color=FB_YELLOW, fontsize=12)
ax.text(0.5,-0.15, f"Two-sample t-test  |  After: {after_eu.mean():.2f}   Normal: {normal.mean():.2f}",
        transform=ax.transAxes, ha='center', fontsize=9, color=TEXT)
ax.grid(axis='y')

# H3 – Possession vs Goal Difference
ax = axes[2]
poss_df = df.dropna(subset=['Poss','GoalDiff'])
r, p3 = stats.pearsonr(poss_df['Poss'], poss_df['GoalDiff'])
ax.scatter(poss_df['Poss'], poss_df['GoalDiff'],
           alpha=0.4, color=FB_BLUE, edgecolors=FB_YELLOW, linewidth=0.3, s=40)
m, b = np.polyfit(poss_df['Poss'], poss_df['GoalDiff'], 1)
xs = np.linspace(poss_df['Poss'].min(), poss_df['Poss'].max(), 100)
ax.plot(xs, m*xs+b, color=FB_YELLOW, lw=2, label=f'r = {r:.3f}')
ax.axhline(0, color='white', ls=':', lw=1)
ax.set_xlabel("Possession (%)"); ax.set_ylabel("Goal Difference")
ax.set_title(f"H3: Possession vs Goal Difference\np = {p3:.4f}  {'✓ Significant' if p3<0.05 else '✗ Not Significant'}",
             color=FB_YELLOW, fontsize=12)
ax.legend(fontsize=10); ax.grid()

plt.tight_layout()
plt.savefig('Hypothesis_testing.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Figure 2 saved.")

# ════════════════════════════════════════════════════════════
# FIGURE 3 – Cumulative Points & Rest Days Effect
# ════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(16,6))
fig3.patch.set_facecolor(BG)
fig3.suptitle("Cumulative Performance & Rest Days Effect", fontsize=16, fontweight='bold', color=FB_YELLOW)

# Cumulative points – last 5 seasons (Süper Lig only)
ax = axes[0]
for season, color in zip(seasons[-5:], ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']):
    s_df = df[(df['Season']==season) & (df['IsLeague'])].sort_values('Date').copy()
    if len(s_df) == 0: continue
    s_df['CumPts'] = s_df['Points'].cumsum()
    ax.plot(range(len(s_df)), s_df['CumPts'].values, label=season, color=color, lw=2)
ax.set_xlabel("Match Number"); ax.set_ylabel("Cumulative Points")
ax.set_title("Last 5 Seasons – Cumulative Points (Süper Lig)", color=FB_YELLOW, fontsize=12)
ax.legend(fontsize=9); ax.grid()

# Rest days between matches vs average points
ax = axes[1]
rest_df = df.dropna(subset=['RestDays'])
rest_bins = pd.cut(rest_df['RestDays'], bins=[0,3,5,7,30], labels=['1-3','4-5','6-7','7+'])
rest_pts = rest_df.groupby(rest_bins, observed=True)['Points'].mean()
b4 = ax.bar(rest_pts.index.astype(str), rest_pts.values,
            color=[LOSS_C,DRAW_C,WIN_C,FB_BLUE], edgecolor=TEXT, linewidth=0.8, width=0.5)
ax.set_xlabel("Rest Days Between Matches"); ax.set_ylabel("Avg Points / Match")
ax.set_title("Rest Days vs Performance", color=FB_YELLOW, fontsize=12)
ax.grid(axis='y')
for b,v in zip(b4, rest_pts.values):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('cumulative_rest_days.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Figure 3 saved.")

print("\n=== ANALYSIS COMPLETE ===")
