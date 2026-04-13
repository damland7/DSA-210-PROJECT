# Fenerbahçe Match Performance Analysis
**DSA210 – Spring 2026 | Damla Kandemir**

## Project Overview
This project analyzes the factors affecting Fenerbahçe's match performance across different competitions (Süper Lig, European competitions) over 10 seasons (2016–2026).

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

## Key Findings
- **Win rate:** 57.0% overall (Home: 61.9% | Away: 52.2%)
- **Home advantage** is statistically significant (p = 0.0131)
- **European fixture fatigue** does not significantly reduce league points (p = 0.5825)
- **Possession** is not a strong predictor of goal difference for Fenerbahçe (r = 0.018)

---

## Visualizations

### General Performance (10 Seasons)
![General Performance](General_Performance.png)

### Hypothesis Tests
![Hypothesis Tests](Hypothesis_testing.png)

### Cumulative Points & Rest Days Effect
![Cumulative Rest Days](cumulative_rest_days.png)

---

## Repository Structure
```
DSA-210-PROJECT/
├── README.md
├── requirements.txt
├── fenerbahce_eda.py
├── General_Performance.png
├── Hypothesis_testing.png
├── cumulative_rest_days.png
└── data/
    ├── Fenerbahce_2016-2017.csv
    ├── Fenerbahce_2017-2018.csv
    ├── Fenerbahce_2018-2019.csv
    ├── Fenerbahce_2019-2020.csv
    ├── Fenerbahce_2020-2021.csv
    ├── Fenerbahce_2021-2022.csv
    ├── Fenerbahce_2022-2023.csv
    ├── Fenerbahce_2023-2024.csv
    ├── Fenerbahçe_2024-2025.csv
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

### 3. Run the analysis
```bash
python fenerbahce_eda.py
```

This will generate 3 figure files:
- `General_Performance.png`
- `Hypothesis_testing.png`
- `cumulative_rest_days.png`

> **Note:** The CSV data files must be in the same folder as `fenerbahce_eda.py`, or update the file paths in the `csv_files` dictionary at the top of the script.

---

## Data Source
Data collected manually from FBref match logs:
- https://fbref.com/en/squads/ae1e2d7d/Fenerbahce-Stats

---

## Next Steps
- Integrate opponent strength data from Transfermarkt (squad market values)
- Apply machine learning methods (Logistic Regression, Random Forest) to predict match outcomes
