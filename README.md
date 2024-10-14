# Dataset and Results Files

This README provides information about the files containing our dataset and results.

## File Descriptions

### datasets/preprocessed/{period_shift}{Tm}{group}.csv

Dataset after transformation to stationary process using fractional differencing and seasonal adjustment. There are four subdirectories: period_shift=original, 1_month_shifted, 2_month_shifted, and 3_month_shifted, each containing four subdirectories: Tm=T1, T2, T3, T4. The CSVs under each directory correspond to $[A_1, A_2, A_3]$ for each group's period_shift and Tm, in the order of competitor1, competitor2, and target.

### miao_score_tables/{group}.csv

This file contains the score table for each group obtained by MIAO.

Columns:
- **miao_score**: Represents $AMS_{ij}$ in the paper

### decision_tree_dataset.csv

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

### classification_results.csv

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