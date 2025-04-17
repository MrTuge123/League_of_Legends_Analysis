# Categorization Model of Predicting LOL Game Outcomes

**Authors:**
- Xingzhi Cui ([tigercui@umich.edu](mailto:tigercui@umich.edu))
- Yun Jong Na ([kevinyjn@umich.edu](mailto:kevinyjn@umich.edu))
**Email:** tigercui@umich.edu, kevinyjn@umich.edu

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)  
3. [Framing a Prediction Problem](#framing-a-prediction-problem)  
4. [Baseline Model](#baseline-model)  
5. [Final Model](#final-model)  
6. [Installation & Usage](#installation--usage)

---

## Introduction

In this project, we explore whether in-game features at the 25 minute cutoff can accurately predict the outcome (win/loss) of League of Legends matches. Our dataset consists of approximately **868** matches in 2024 sourced from OraclesElixir, which is a public dataset under Riot Games, containing information on champion selections, team compositions, player roles, and match metadata.

- **Central Question:** Can a classification model leverage in-game kills, gold, experience information at the 25 minute checkpoint to predict the game result?
- **Motivation:** Predictive insights can inform esports strategy and enhance spectator engagement by offering data-driven match forecasts.  
- **Dataset Details:**  
  - **Rows:** 1736, 2 rows per game
  - **Features:**
    - `goldat25`: Team's total gold collected by minute 25
    - `xpat25`: Team's total experience points gained by minute 25
    - `csat25`: Team's total creep score (minion kills) by minute 25
    - `opp_goldat25`: Opponent team's total gold at minute 25
    - `opp_xpat25`: Opponent team's total experience at minute 25
    - `opp_csat25`: Opponent team's total creep score at minute 25
    - `killsat25`: Total kills achieved by the team by minute 25
    - `opp_killsat25`: Total kills achieved by the opponent team by minute 25
  - **Target Variable:** `result` (win = 1, loss = 0)  
  - **Relevance:** Accurate prediction models support coaches and analysts in optimizing in-game decisions and contribute to the broader field of sports analytics.
---

## Data Cleaning and Exploratory Data Analysis

1. **Row Selection** 
- **League filter:** Restrict to the six major leagues (WLDs, LCK, LPL, LEC, LTA, LCP).  
- **Level filter:** From the original 12 rows per match, keep only the two **team‑level** records (`data = data[data['position'] == 'team']`); discard all player‑level entries.

2. **Feature Selection**  
- Restricted to features measured at 25 minutes.  
- Removed time‑static variables (e.g. total dragons slain).  
- Dropped perfectly collinear stats (ex: `killsAs25` vs. `opp_deathsAt25`) to aid interpretability.

3. **Missing‑Value Handling**
- **Assessment of missingness:** We found that rows lacking any 25‑minute metric (gold, kills, objectives, etc.) account for roughly **46.58%** of our team‑level data.
- **Rationale for dropping:** A missing 25‑minute value indicates that the game state wasn’t recorded at that cut‑off, so these rows contain no usable mid‑game information. Imputing them would introduce unfounded assumptions, so we removed all records with missing 25‑minute features. ​

### TODO: Univariate Analysis

### TODO: Bivariate Analysis


