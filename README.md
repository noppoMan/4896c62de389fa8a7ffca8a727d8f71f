# Dataset and Results Files

This README provides information about the files containing our dataset and results.

## datasets/original/daily/{org}/{name}.csv

These are datasets obtained from https://github.com/{org}/{name} repositories by cloning them and collecting daily commit counts. These datasets serve as the foundation for our research.

## datasets/preprocessed/{period_shift}{Tm}{group}.csv

Dataset after transformation to stationary process using fractional differencing and seasonal adjustment. There are four subdirectories: period_shift=original, 1_month_shifted, 2_month_shifted, and 3_month_shifted, each containing four subdirectories: Tm=T1, T2, T3, T4. The CSVs under each directory correspond to $[A_1, A_2, A_3]$ for each group's period_shift and Tm, in the order of competitor1, competitor2, and target.

The process of creating datasets under the preprocessed directory is implemented in the #Auto-detecting-$T_m$ and #Preparation-for-VAR-Analysis sections of demo.ipynb. Additionally, the SVAR estimation and IRF calculation using these datasets are implemented in the #SVAR and #IRF-Calculation sections of demo.ipynb.

## repo_state.csv

A CSV file summarizing information about OSS projects in the dataset, based on data up
to June 30, 2024.

Columns:
- **n_commits**: Total number of commits
- **n_stars**: Number of GitHub stars
- **code_size**: Total code size of all files in the repository
- **contributors**: Unique count of contributors listed in commit logs
- **repo_state**: One of the following five values:
   - alive: Surviving until the latest date
   - archived: Archived on GitHub
   - deprecated: Marked as deprecated
   - abandonment: No commits in the last year
   - dormant: Average monthly commits in the last 12 months falls below 1.5
- **repo_state_val**: For abandonment cases, number of days between the last commit date and June 30, 2024 (latest date). For dormant cases, average monthly commits over the last 12 months.
- **rev**: 1 if designated as REV, 0 otherwise

## adf_test_result.csv

ADF test results and information on fractional differencing for all activity data used in the experiment. Each row corresponds to $A_i$ of each group for period shift and $T_m$. Since 2,884 $A_i$ were obtained in the experiment, this CSV consists of 2,884 rows + header row.

This is the log of results from executing the `miao.prepare_var` function for all period shifts, $T_m$.

Columns:
- **period_shift**: Period shift corresponding to the data
- **Tm**: $T_m$ corresponding to the data
- **group**: Group name corresponding to the data
- **repo**: Repository corresponding to the data
- **statistic**: ADF test statistic
- **pvalue**: p-value of the ADF test
- **fracdiff-n**: Fractional difference $n$. $n=0$ means level. The fractional differencing transformation was performed using Python's fracdiff package. See: https://github.com/fracdiff/fracdiff

### Verification of Unit Root Presence in the Dataset

This code allows you to verify whether the time series data $A_i$ in the datasets/preprocessed directory contains any unit root processes. If unit root processes are present, the unit_root_count value will be greater than 0. In MIAO theory, since all $A_i$ are required to be stationary, the unit_root_count should be 0.

```python
from glob import glob
from statsmodels.tsa.stattools import adfuller

unit_root_count = 0
i = 0

files = glob("./datasets/preprocessed/**/**/*.csv")
for file in files:
    df = pd.read_csv(file, index_col=0)
    for col in df.columns:
        i += 1
        results = adfuller(df[col], regression='c')
        if results[1] >= 0.05:
            unit_root_count += 1

    print(f"total scanned: {i}, unit_root_count: {unit_root_count}", end='\r')
```

## var_estimation_result.csv

Estimation results of VAR estimated in the experiment and results of Ljung-Box test. Each row shows the estimation values obtained from the SVAR model at the corresponding Period shift and Tm. Since 952 SVARs were obtained in the experiment, this CSV consists of 952 rows + header row.

This is the log of results from executing the `miao.prepare_var` function for all period shifts, $T_m$.

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

## miao_score_tables/{group}_{n_split}.csv

This file contains the score table for each group obtained by MIAO.

Columns:
- **miao_score**: Represents $\mathrm{AMS}_{ij}$ in the paper

**The code for calculating MIAO scores can be found in the #MIAO-Score-Calculation section of `demo.ipynb`.**

## decision_tree_dataset.csv

This dataset is used for training and testing the decision tree model. All data have been normalized.

Columns:
- **label**: Correct label (target variable)
- **group**: Group name
- **target**: Name of the target OSS
- **c2 -> c1**: $\mathrm{AMS}_{c2,c1}$
- **t -> c1**: $\mathrm{AMS}_{t,c1}$
- **c1 -> c2**: $\mathrm{AMS}_{c1,c2}$
- **t -> c2**: $\mathrm{AMS}_{t,c2}$
- **c1 -> t**: $\mathrm{AMS}_{c1,t}$
- **c2 -> t**: $\mathrm{AMS}_{c2,t}$

The dataset version of $\mathrm{AMS}_{ij}$ without normalization (not divided by m) is non_normalized_decision_tree_dataset.csv. This is equivalent to the CSV outputs under miao_score_tables.

## classification_results.csv

This file contains the classification results produced by the decision tree model for 63 groups in the evaluation section.

Columns:
- **group**: Group name
- **rev**: True value of whether it's REV or not
- **predicted**: Predicted value by the decision tree

**All code used in the experiment (hyperparameters, training, and performance evaluation) can be found in the #Decision Tree section of `demo.ipynb`**

## miao.py

A Python File containing programs for executing MIAO. 