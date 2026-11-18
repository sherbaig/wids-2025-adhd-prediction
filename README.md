# WiDS 2025 Datathon: Diagnosis Prediction from fMRI and Questionnaire Data

End-to-end ML pipeline predicting a binary diagnosis outcome for 1,213 participants, using fMRI connectome features, demographics and behavioural questionnaires.

## What it does
- Cleans and merges multi-modal data (~19,900 features)
- Engineers features, selects the top 30 with SelectKBest and reduces them with PCA
- Handles class imbalance with SMOTE, applied to the training split only
- Compares 8 classifiers with 5-fold stratified cross-validation
- Tunes the top 3 (Random Forest, SVM, LightGBM) with RandomizedSearchCV
- Builds a stacking ensemble and interprets a LightGBM model with SHAP on named features

## Results (held-out test set)
| Model | ROC-AUC |
|---|---|
| Random Forest | 0.831 |
| LightGBM | 0.836 |
| SVM | 0.840 |
| Stacking ensemble | 0.846 |

## Known limitations
- Scaling, feature selection and PCA are fitted before the train/test split, so held-out scores may be slightly optimistic. A leakage-free pipeline version is in progress.
- A partial dependence plot is skipped because of a plotting error.

## Tech
Python, pandas, scikit-learn, imbalanced-learn, XGBoost, LightGBM, SHAP

## Data
Competition data is not included. Download it from the WiDS Datathon 2025 page and set `BASE_PATH` in the notebook.
