# Dataset and Results Files

This README provides information about the files containing our dataset and results.

## datasets/preprocessed/{period_shift}{Tm}{group}.csv

Dataset after transformation to stationary process using fractional differencing and seasonal adjustment. There are four subdirectories: period_shift=original, 1_month_shifted, 2_month_shifted, and 3_month_shifted, each containing four subdirectories: Tm=T1, T2, T3, T4. The CSVs under each directory correspond to $[A_1, A_2, A_3]$ for each group's period_shift and Tm, in the order of competitor1, competitor2, and target.

## adf_test_result.csv

ADF test results and information on fractional differencing for all activity data used in the experiment. Each row corresponds to $A_i$ of each group for period shift and $T_m$.

Columns:
- **period_shift**: Period shift corresponding to the data
- **Tm**: $T_m$ corresponding to the data
- **group**: Group name corresponding to the data
- **repo**: Repository corresponding to the data
- **statistic**: ADF test statistic
- **pvalue**: p-value of the ADF test
- **fracdiff-n**: Fractional difference $n$. $n=0$ means level.

## var_estimation_result.csv

Estimation results of VAR estimated in the experiment and results of Ljung-Box test. The estimation unit is the target of the VAR model. Each row corresponds to VAR for period shift and $T_m$.

Columns:
- **period_shift**: Period shift corresponding to the estimated VAR
- **Tm**: $T_m$ corresponding to the estimated VAR
- **group**: Group name of the estimated VAR
- **nobs**: Number of samples in the estimated VAR
- **lag**: Optimal Lag order of the estimated VAR using `statsmodels.tsa.vector_ar.var_model.VAR.select_order`
- **aic**: AIC for Optimal Lag order (AIC is adopted in the experiment)
- **bic**: BIC for Optimal Lag order
- **hqic**: HQIC for Optimal Lag order
- **whiteness_test_lag**: Maximum lag order for the Ljung-Box test
- **whiteness_test_statistic**: Ljung-Box test statistic for the error terms of the estimated SVAR
- **whiteness_test_pvalue**: p-value of the Ljung-Box test for the error terms of the estimated SVAR

## miao_score_tables/{group}.csv

This file contains the score table for each group obtained by MIAO.

Columns:
- **miao_score**: Represents $AMS_{ij}$ in the paper

## decision_tree_dataset.csv

This dataset is used for training and testing the decision tree model.

Columns:
- **label**: Correct label (target variable)
- **group**: Group name
- **target**: Name of the target OSS
- **c2 -> c1**: $AMS_{c2,c1}$
- **t -> c1**: $AMS_{t,c1}$
- **c1 -> c2**: $AMS_{c1,c2}$
- **t -> c2**: $AMS_{t,c2}$
- **c1 -> t**: $AMS_{c1,t}$
- **c2 -> t**: $AMS_{c2,t}$

## classification_results.csv

This file contains the classification results produced by the decision tree model.

Columns:
- **group**: Group name
- **rev**: True value of whether it's REV or not
- **predicted**: Predicted value by the decision tree

Settings of Decision Tree Model: 

```python
params = {
    'max_depth': 3, 
    'min_samples_split': 10, 
    'min_samples_leaf': 3,
    'random_state': 1
}

clf = DecisionTreeClassifier(**params)
```