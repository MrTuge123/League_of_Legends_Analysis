# Categorization Model of Predicting LOL Game Outcomes

**Author:** Xingzhi Cui, Yun Jong Na 
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

In this project, we explore whether in-game features at the 25 minute cutoff can accurately predict the outcome (win/loss) of League of Legends matches. Our dataset consists of approximately **1625** matches in 2024 sourced from OraclesElixir, which is a public dataset under Riot Games, containing information on champion selections, team compositions, player roles, and match metadata.

- **Central Question:** Can a classification model leverage in-game kills, gold, experience information at the 25 minute checkpoint to predict the game result?
- **Motivation:** Predictive insights can inform esports strategy and enhance spectator engagement by offering data-driven match forecasts.  
- **Dataset Details:**  
  - **Rows:** 3250
  - **Features:** `goldat25`, `xpat25`, `csat25`, `opp_goldat25`, `opp_xpat25`, `opp_csat25`, `killsat25`, `opp_killsat25`
  - **Target Variable:** `result` (win = 1, loss = 0)  
- **Relevance:** Accurate prediction models support coaches and analysts in optimizing in-game decisions and contribute to the broader field of sports analytics.
---


## Data Cleaning and Exploratory Data Analysis
