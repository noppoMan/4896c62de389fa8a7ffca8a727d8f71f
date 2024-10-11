# Dataset and Results Files

This README provides information about the files containing our dataset and results.

## File Descriptions

### miao_score_table/{group}.csv

This file contains the score table for each group obtained by MIAO.

Columns:
- **miao_score**: Represents AMS_ij in the paper

### decision_tree_dataset.csv

This dataset is used for training and testing the decision tree model.

Columns:
- **label**: Correct label (target variable)
- **group**: Group name
- **target**: Name of the target OSS
- **c2 -> c1**: AMS_c2,c1
- **t -> c1**: AMS_t,c1
- **c1 -> c2**: AMS_c1,c2
- **t -> c2**: AMS_t,c2
- **c1 -> t**: AMS_c1,t
- **c2 -> t**: AMS_c2,t

### classification_results.csv

This file contains the classification results produced by the decision tree model.

Columns:
- **group**: Group name
- **rev**: True value of whether it's REV or not
- **predicted**: Predicted value by the decision tree