# Categorization Model of Predicting LOL Game Outcomes

**Authors:**
- Xingzhi Cui ([tigercui@umich.edu](mailto:tigercui@umich.edu))
- Yun Jong Na ([kevinyjn@umich.edu](mailto:kevinyjn@umich.edu))

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)  
3. [Framing a Prediction Problem](#framing-a-prediction-problem)  
4. [Baseline Model](#baseline-model)  
5. [Testing for Potential Improvement](#potential-improvement)  
6. [Model Comparison and Final Selection](#final-model)

---

## Introduction

In this project, we explore whether in-game features at the 25 minute cutoff can accurately predict the outcome (win/loss) of League of Legends matches. Our full dataset consists of approximately **9800** matches in 2024 sourced from OraclesElixir, which is a public dataset under Riot Games, containing information on champion selections, team compositions, player roles, and match metadata.

- **Central Question:** Can a classification model leverage in-game kills, gold, experience information at the 25 minute checkpoint to predict the game result?
- **Motivation:** Predictive insights can inform esports strategy and enhance spectator engagement by offering data-driven match forecasts.  
- **Dataset Details:**  
  - **Rows:** 117600, 2 rows per game
  - **Relavant Features:**
    - `gameid`: A unique identifier for each game
    - `goldat25`: Team’s total gold collected by minute 25  
    - `xpat25`: Team’s total experience points gained by minute 25  
    - `csat25`: Team’s total creep score (minion kills) by minute 25  
    - `killsat25`: Total kills achieved by the team by minute 25  
    - `assistsat25`: Total assists by the team by minute 25  
    - `deathsat25`: Total deaths suffered by the team by minute 25 
    - `opp_goldat25`: Opponent team’s total gold collected by minute 25  
    - `opp_xpat25`: Opponent team’s total experience points gained by minute 25  
    - `opp_csat25`: Opponent team’s total creep score by minute 25  
    - `opp_killsat25`: Total kills achieved by the opponent team by minute 25  
    - `opp_assistsat25`: Total assists by the opponent team by minute 25  
    - `opp_deathsat25`: Total deaths suffered by the opponent team by minute 25 
    - `golddiffat25`: Gold difference at minute 25 (`goldat25 – opp_goldat25`)  
    - `xpdiffat25`: Experience difference at minute 25 (`xpat25 – opp_xpat25`)  
    - `csdiffat25`: Creep score difference at minute 25 (`csat25 – opp_csat25`)  
    - `side`: Team side designation (Blue or Red), which can influence draft priority and map control  
    - `gamelength`: Total duration of the match, in seconds, capturing overall game pace  
    - `league`: Regional league or tournament in which the match was played (e.g., LCK, LPL, LEC)  
    
  - **Target Variable:** `result` (win = 1, loss = 0)  
  
  - **Relevance:** Accurate prediction models support coaches and analysts in optimizing in‑game decisions and contribute to the broader field of sports analytics.

---

## Data Cleaning and Exploratory Data Analysis

1. **Row Selection** 
- **Level filter:** From the original 12 rows per match, keep only the two **team‑level** records (`data = data[data['position'] == 'team']`); discard all player‑level entries.

2. **Feature Selection**  
- Restricted to features measured at 25 minutes.  
- Removed time‑static variables (e.g. total dragons slain).  
- Dropped perfectly collinear stats (ex: `killsAs25` vs. `opp_deathsAt25`) to aid interpretability.

3. **Imputation Strategies and Missing‑Value Handling**
- **Assessment of missingness:** We found that rows lacking any 25‑minute metric (gold, kills, objectives, etc.) account for roughly **46.58%** of our team‑level data.
- **Rationale for dropping:** A missing 25‑minute value indicates that the game state wasn’t recorded at that cut‑off, so these rows contain no usable mid‑game information. Imputing them would introduce unfounded assumptions, so we removed all records with missing 25‑minute features. ​

**Head of Cleaned Data:**

| gameid          |   result |   goldat25 |   xpat25 |   csat25 |   killsat25 |   deathsat25 |   opp_goldat25 |   opp_xpat25 |   opp_csat25 |
|:----------------|---------:|-----------:|---------:|---------:|------------:|-------------:|---------------:|-------------:|-------------:|
| LOLTMNT06_13630 |        0 |      45581 |    53080 |      904 |           9 |            7 |          44394 |        55632 |          899 |
| LOLTMNT06_13630 |        1 |      44394 |    55632 |      899 |           7 |            9 |          45581 |        53080 |          904 |
| LOLTMNT06_12701 |        0 |      40305 |    50828 |      864 |           4 |            9 |          44748 |        57191 |          878 |
| LOLTMNT06_12701 |        1 |      44748 |    57191 |      878 |           9 |            4 |          40305 |        50828 |          864 |
| LOLTMNT06_13667 |        0 |      43673 |    55802 |      900 |           6 |            7 |          42984 |        54096 |          893 |


**TODO: Univariate Analysis**
<iframe src="images/fig_uni1.html" width="800" height="600" frameborder="0" ></iframe>

**TODO: Bivariate Analyses and Aggregations**
<iframe src="images/figbi1.html" width="800" height="600" frameborder="0" ></iframe>

<iframe src="images/aggregate_table.html" width="800" height="400" frameborder="0"></iframe>



-**Issues with Multicollinearity**

Even after dropping the most obvious redundancies, many of our 25‑minute features remain highly correlated. For example, teams that secure more kills typically accumulate more gold and experience, while opponents with higher death counts correspondingly lag behind. This interdependence can lead to unstable coefficient estimates in a standard logistic regression. Therefore, in the following regression models, we will need to either address these issues or interpret the results with caution. 

![Correlation Matrix of Features](images/corr_matrix.png)



## Framing a Prediction Problem

- **Task:** Binary classification of match outcome, `result` (1 = win, 0 = loss).  
- **Features:** All in‑game metrics available at the 25 minute mark **after** the filtering and cleaning steps described in [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis). We exclude any variables that aren’t known at minute 25 (e.g. final kill totals).  
- **Evaluation Metric:** **Accuracy**, defined as the proportion of correct predictions:
  $$
    \text{Accuracy}
    = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
    = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\bigl(y_i = \hat y_i\bigr),
  $$
  where \(y_i\) is the true label and \(\hat y_i\) is the model’s prediction.  
- **Prediction Question:**  
  > Can a classifier, given only kills, gold, XP, creep‑score, and objective metrics at minute 25, accurately predict which team will win the match?



## Baseline Model

To establish a performance benchmark, we trained a **simple logistic regression** on our 25‑minute features for the team, exluding the information of their opponents.  

**1. Data Partitioning**  
We split the cleaned dataset into training (70%) and test (30%) sets, preserving the win/loss balance via stratification:

```python
from sklearn.model_selection import train_test_split

y = data['result']
X = data[[
    'goldat25', 'xpat25', 'csat25',
    'killsat25', 'deathsat25'
]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=123
)
```

**Baseline Model Pipeline:**

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        penalty='l2',
        solver='liblinear',
        random_state=398
    )
)

pipeline.fit(X_train, y_train)

# Predictions
y_test_pred = pipeline.predict(X_test)
y_train_pred = pipeline.predict(X_train)

# Metrics
test_acc   = accuracy_score(y_test,   y_test_pred)
train_acc  = accuracy_score(y_train,  y_train_pred)
roc_auc    = roc_auc_score(y_test,    pipeline.decision_function(X_test))
num_coeffs = np.count_nonzero(pipeline.named_steps['logisticregression'].coef_)

# Tabulate results
results = {
    'Train Accuracy': train_acc,
    'Test Accuracy':  test_acc,
    'ROC‑AUC':        roc_auc,
    'Non‑zero Coeff': num_coeffs
}
```

**Model Information**

| Data Type    | Features                                                                                                       | Processing Method |
|--------------|----------------------------------------------------------------------------------------------------------------|-------------------|
| Quantitative | `goldat25`, `xpat25`, `csat25`, `killsat25`, `deathsat25`     | `StandardScaler`  |
| Ordinal      | –                                                                                                              | –                 |
| Nominal      | –                                                                                                              | –                 |


**Results Output**

| Model Name                          |   num_non_zero |   training_accuracy |   testing_accuracy |   ROC_accuracy |
|:------------------------------------|---------------:|--------------------:|-------------------:|---------------:|
| Baseline_Simple_Logistic_Regression |              5 |            0.809433 |           0.812473 |       0.897624 |


**Model Analysis**

The baseline logistic regression achieves 80.94% accuracy on the training set and 81.25% on the test set, with a ROC‑AUC of 0.8976—indicating strong overall discrimination between wins and losses. However, the presence of multicollinearity among mid‑game features, as indicated in [Issues with Multicollinearity](#issues-with-multicollinearity), suggests that regularized models or dimensionality‑reduction techniques may further improve stability and performance in subsequent modeling steps. Moreover, including more features may improve the predictive power of the model.



## Testing for Potential Improvement

-**Feature Engineering**

To enrich our predictive signal, we extended the feature set as follows:

- **Categorical Variables:**  
  - **League** and **Side** were one‑hot encoded via `ColumnTransformer` + `OneHotEncoder` to capture structural differences across competitive regions and team assignments.  
- **Continuous Variables:**  
  - **Game Length** was retained as a numeric feature and standardized alongside other in‑game metrics.  
- **Opponent Metrics:**  
  - We introduced `opp_goldat25`, `opp_xpat25`, and `opp_csat25` to account for the opposing team’s mid‑game performance, which can directly influence outcome probabilities.

---

-**Addressing Multicollinearity**

Our initial OLS‑based logistic regression revealed unstable coefficient estimates under high predictor correlation. Multicollinearity inflates variance, undermines interpretability, and can degrade out‑of‑sample performance.

To mitigate these issues, we evaluate three modeling strategies expressly designed to handle correlated features:

1. **Ridge Logistic Regression (L2 Regularization)**  
   Applies an L2 penalty to shrink coefficient magnitudes, reducing variance without discarding predictors.

2. **LASSO Logistic Regression (L1 Regularization)**  
   Introduces an L1 penalty to drive some coefficients to zero, thus performing automatic variable selection and alleviating redundancy.

3. **Recursive Feature Elimination (RFE)**  
   Recursively removes the least important features based on model performance, yielding a parsimonious set of predictors.

Each approach offers a robust mechanism for stabilizing estimates and improving generalization in the presence of multicollinearity. Subsequent sections detail their implementation and comparative evaluation.  


## Model Comparison and Final Selection

| Model Name                                   |   num_non_zero |   training_accuracy |   testing_accuracy |   ROC_accuracy |
|:---------------------------------------------|---------------:|--------------------:|-------------------:|---------------:|
| Baseline_Simple_Logistic_Regression          |              5 |            0.809433 |           0.812473 |       0.897624 |
| LogisticRegressionCV_L1                      |              9 |            0.835888 |           0.8361   |       0.921376 |
| Ridge_LogisticRegressionCV                   |             57 |            0.835979 |           0.836526 |       0.921788 |
| LogisticRegression_RFECV_BackwardElimination |              5 |            0.835523 |           0.833972 |       0.920703 |


**Key Observations:**

- The **L1‑penalized** (LASSO) model matches the highest test accuracy (83.61%) and AUC (0.9214), using only **9** non‑zero coefficients.  
- The **Ridge** model attains similar performance metrics but retains **57** coefficients, increasing complexity and overfitting risk.  
- The **RFECV** approach yields a very sparse model (5 features) but with slightly lower test accuracy (83.40%) and AUC (0.9207).  
- The **Baseline** logistic regression lags behind on both accuracy and discrimination (AUC).

**Final Model Choice:**  
We select the **LASSO Logistic Regression** as our final model based on its superior balance of performance, sparsity, and stability:

1. **Strong performance lift**—achieves **83.61%** test accuracy (up from 81.25% with the baseline) and **ROC‑AUC = 0.9214** (up from 0.8976), nearly matching Ridge’s performance.  
2. **Controlled complexity**—retains only **9** non‑zero coefficients versus **57** in the Ridge model, dramatically reducing overfitting risk.  
3. **Better than RFECV**—outperforms the backward‑elimination approach (5 features, 83.40% accuracy) while still providing a sparse, interpretable solution.  
4. **Multicollinearity mitigation**—automatically zeros out redundant mid‑game metrics, stabilizing coefficient estimates and enhancing generalization.  



```python
data['side_binary'] = data['side'].map({'Blue': 0, 'Red': 1})
cols = [['gameid','result','goldat25', 'xpat25', 'csat25', 'killsat25','deathsat25', 'opp_goldat25', 'opp_xpat25', 'opp_csat25', 'side_binary' ,'gamelength', 'league']]
data = data[cols]
y = data['result']
X = data[['goldat25', 'xpat25', 'csat25', 'killsat25','deathsat25', 'opp_goldat25', 'opp_xpat25', 'opp_csat25', 'side_binary' ,'gamelength', 'league']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=103) 

numeric_feats = ['goldat25', 'xpat25', 'csat25', 'killsat25','deathsat25', 'opp_goldat25', 'opp_xpat25', 'opp_csat25', 'side_binary' ,'gamelength']
categorical_feats = ['league']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feats)
], remainder='drop') 

pipeline = make_pipeline(
    preprocessor,
    LogisticRegressionCV(
        cv=5,
        penalty='l1',
        solver='liblinear',
        random_state=398
    )
)

pipeline.fit(X_train, y_train)

y_test_pred = pipeline.predict(X_test)
y_test_score = pipeline.decision_function(X_test)
testing_accuracy = accuracy_score(y_test, y_test_pred)
roc_auc          = roc_auc_score(y_test, y_test_score)

log_reg_cv = pipeline.named_steps['logisticregressioncv']

coef_series = pd.Series(
    log_reg_cv.coef_[0],
    index=feature_names
)

selected_features = coef_series[coef_series != 0].index.tolist()
num_non_zero = len(selected_features)

y_train_pred = pipeline.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_pred)

new_row = {
    'Model Name':        'LogisticRegressionCV_L1',
    'num_non_zero':      num_non_zero,
    'training_accuracy': training_accuracy,
    'testing_accuracy':  testing_accuracy,
    'ROC_accuracy':      roc_auc
}

Model_selection_df.loc[len(Model_selection_df)] = new_row
```
-**Model Properties**
| Data Type    | Features                                                                                                                                                                                           | Processing Method      |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| Quantitative | `goldat25`, `xpat25`, `csat25`, `killsat25`, `deathsat25`,<br>`opp_goldat25`, `opp_xpat25`, `opp_csat25`, `opp_killsat25`, `opp_assistsat25`, `opp_deathsat25`,<br>`golddiffat25`, `xpdiffat25`, `csdiffat25`,<br>`gamelength`, `side_binary` | `StandardScaler`       |
| Nominal      | `league` (one‑hot encoded)                                                                                                                                                                        | `OneHotEncoder`        |
| Ordinal      | –   


-**Model Analysis**
The model end up having the following predictors:
'num__goldat25', 'num__xpat25', 'num__csat25', 'num__killsat25', 'num__deathsat25', 'num__opp_goldat25', 'num__opp_xpat25', 'num__opp_csat25', 'num__side_binary'

**Model Analysis**  
The final LASSO logistic regression model retained **9** mid‑game features with non‑zero coefficients:

- `goldat25`  
- `xpat25`  
- `csat25`  
- `killsat25`  
- `deathsat25`  
- `opp_goldat25`  
- `opp_xpat25`  
- `opp_csat25`  
- `side_binary`  

> **Interpretation:**  
> These selected predictors underscore the importance of both a team’s own resource accumulation (gold, experience, creep score, kills, deaths) and the opponent’s resource metrics at 25 minutes, as well as map-side assignment (`side_binary`), in forecasting match outcomes. By focusing on this sparse set of highly informative features, the LASSO model delivers robust predictions while mitigating multicollinearity.  
