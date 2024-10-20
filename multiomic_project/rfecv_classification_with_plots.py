import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer

import os
from utils import load_discovery_data_for_classification
from configs import MY_DIR
#
# MY directory 
os.makedirs(MY_DIR, exist_ok=True)

# Load data
data = load_discovery_data_for_classification()
#data = data.iloc[:, :100]

# Split your data into features (X) and target (y)
X = data.drop(columns=['Target']) 
y = data['Target'] 
print("Number of classes(y):", np.unique(y))

# ------------------- 

# Global variables
cv = 3
seed = 42
# -------------------

# Function to allow custom XGBoost parameter tuning
def build_xgboost_model(custom_params=None):
    """
    Build an XGBoost model .
    
    :custom_params: Dictionary of XGBoost parameters (optional).
    :return: XGBoost model
    """
    default_params = {
        'use_label_encoder': False, 
        'eval_metric': 'auc', 
        'objective': 'multi:softprob',  # Multiclass classification objective for probabilities
        'num_class': len(set(y)),
        'learning_rate': 0.05,
        'n_estimators': 200,
        'max_depth': 3,
        'subsample': 0.8,
        'min_child_weight': 4,
        'n_jobs': -1,
        'reg_lambda': 1,
        'random_state': seed
    }
    if custom_params:
        default_params.update(custom_params)
    
    return xgb.XGBClassifier(**default_params)

#-------------------------- 
# Split to train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

#----- ----- ----- ----- ----- Model 1: RFECV ----- ----- ----- ----- ----- ----- ----- ----- -----
#  Perform RFECV and extract optimal features
def perform_rfecv(model, X_train, y_train, cv):
    """
    :param model: Estimator model for RFECV.
    :param cv: Number of cross-validation folds.
    :return:[1]rfecv.support_ (Boolean mask of selected features):indicates which features have been selected by the RFECV).
            [2]rfecv(anking of each feature,number of features considered optimal by the RFECV,CV results)
    """
    #--------------------------------
    # scorer for AUC 
    model_scorer = 'roc_auc_ovr'
    #--------------------------------
    
    cv_rfecv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    rfecv = RFECV(
        estimator=model,
        step=1, 
        cv=cv_rfecv, 
        scoring=model_scorer, 
        n_jobs=-1,  
        verbose=2
    )
    
    # Fit RFECV to the training data
    rfecv.fit(X_train, y_train)

    # Extract cross-validation scores
    cv_scores = rfecv.cv_results_['mean_test_score']
    print(f"Number of optimal features: {rfecv.n_features_}")
    print(f"Feature ranking: {rfecv.ranking_}")
    print(f"Selected features: {rfecv.support_}")

    # Generate and save the plot for RFECV-selected features
    plot_rfecv_scores(cv_scores, rfecv.n_features_)

    # Save the selected features to a DataFrame and CSV (based on RFECV `support_`)
    selected_features = X_train.columns[rfecv.support_]
    selected_features_df = pd.DataFrame(selected_features, columns=["Selected Features"])
    selected_features_df.to_csv(os.path.join(MY_DIR, 'selected_features.csv'), index=False)

    return rfecv.support_, rfecv

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# plot of RFECV cross-validation scores
def plot_rfecv_scores(cv_scores, n_features_selected):
    """
    :param cv_scores: Cross-validation scores from RFECV.
    :param n_features_selected: Number of optimal features selected by RFECV.
    """
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross-validation score (ROC-AUC)")
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, color='royalblue')
    plt.title("RFECV - Number of Features vs Cross-Validation ROC-AUC")
    plt.axvline(x=n_features_selected, color='red', linestyle='--')  # Add 1 since `n_features` is 0-indexed
    plt.savefig(os.path.join(MY_DIR, 'RFECV_number_of_features_vs_roc_auc.png'), bbox_inches='tight', dpi=300)
    plt.close()

#----- ----- ----- ----- ----- Model 2: Classification(cv=3) ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# Train the model using CV and plot ROC-AUC with class-specific colors
def train_model_with_cv(model, X_train, y_train, selected_features, n_splits=cv, custom_n_features=None):
    """
    Train the model using cross-validation (CV) and plot ROC-AUC with custom feature selection.

    :param model: XGBoost model.
    :param selected_features: Boolean mask of features selected by RFECV.
    :param n_splits: Number of CV folds.
    :param custom_n_features: Integer, optional. Number of top features to select manually.
    """
    # If custom_n_features is specified, select top 'n' features based on ranking
    if custom_n_features:
        top_n_features = np.argsort(rfecv.ranking_)[:custom_n_features]
        X_train_selected = X_train.iloc[:, top_n_features]
    else:
        X_train_selected = X_train.loc[:, selected_features]

    # Function to plot ROC curves for multiclass classification with custom colors
    def plot_multiclass_roc_curves(estimator, X, y, cv, n_classes, class_names):
        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(7, 7))
        tprs, aucs = [], []

        # Define custom colors for each fold of each class
        color_sets = {
            0: ['lightcoral', 'darkred', 'crimson', 'rosybrown'],
            1: ['lightsteelblue', 'royalblue', 'cornflowerblue', 'dodgerblue'],
            2: ['lightgreen', 'green', 'forestgreen', 'darkgreen']
        }

        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_proba = estimator.predict_proba(X.iloc[test_idx])
            
            # Binarize labels for ROC curve calculation
            y_test_binarized = LabelBinarizer().fit_transform(y.iloc[test_idx])
            
            for j in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, j], y_proba[:, j])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                # Use the custom color for each class and fold
                color = color_sets[j][i % len(color_sets[j])]  # Cycle through the colors for each fold
                RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(
                    ax=ax, name=f"Class {class_names[j]} Fold {i+1}", color=color
                )
                
                # Interpolate TPR for mean and CI calculation
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)

        # Calculate the mean and confidence intervals of TPRs
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        
        # Plot the mean ROC curve
        ax.plot(mean_fpr, mean_tpr, color='black', label=r'Mean ROC (AUC = %0.2f)' % (mean_auc), lw=2, alpha=0.8)

        # Plot the chance line
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=0.8)
        
        # Add labels, title, and legend
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC-AUC curves (3-fold CV)')
        ax.legend(loc="lower right")
        
        # Save the figure
        plt.savefig(os.path.join(MY_DIR, 'multiclass_roc_auc_cv.png'), bbox_inches='tight', dpi=300)
        plt.close()

        plot_multiclass_roc_curves(model, X_train_selected, y_train, StratifiedKFold(n_splits=cv), n_classes=len(class_names), class_names=class_names)

#----- ----- ----- ----- ----- Model 3: classification (no split)----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# Train the model without cross-validation and calculate ROC-AUC on the test set
def train_model_without_cv_and_calculate_scores(model, X_train_selected, y_train, X_test_selected, y_test):
    # Train the model on the selected features
    model.fit(X_train_selected, y_train)
    y_pred_proba = model.predict_proba(X_test_selected)

    # Binarize the test labels for ROC-AUC calculation
    y_test_binarized = LabelBinarizer().fit_transform(y_test)
    
    # Calculate different ROC-AUC scores on the test set
    roc_auc_micro = roc_auc_score(y_test_binarized, y_pred_proba, average='micro', multi_class='ovr')
    roc_auc_ovo_macro = roc_auc_score(y_test_binarized, y_pred_proba, average='macro', multi_class='ovo')
    roc_auc_ovo_weighted = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovo')
    roc_auc_ovr_macro = roc_auc_score(y_test_binarized, y_pred_proba, average='macro', multi_class='ovr')
    roc_auc_ovr_weighted = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')

    # Print ROC-AUC scores
    print(f"ROC-AUC (micro, ovr): {roc_auc_micro}")
    print(f"ROC-AUC (macro, ovo): {roc_auc_ovo_macro}")
    print(f"ROC-AUC (weighted, ovo): {roc_auc_ovo_weighted}")
    print(f"ROC-AUC (macro, ovr): {roc_auc_ovr_macro}")
    print(f"ROC-AUC (weighted, ovr): {roc_auc_ovr_weighted}")

#---------------------------------------------------------
# Function to plot and save confusion matrix
def evaluate_confusion_matrix(model, X_test_selected, y_test, class_names):
    """
    Evaluate the model using confusion matrix and various metrics, and plot the confusion matrix.
    
    :param model: Trained model.
    :param X_test_selected: Test feature set (selected features).
    :param y_test: True labels.
    :param class_names: List of class names for the confusion matrix plot.
    """
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_selected)
    
    # Generate confusion matrix
    cnf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1, 2])
    print("\nConfusion Matrix:\n", cnf_matrix)
    
    # Plot and save the confusion matrix
    plot_and_save_confusion_matrix(cnf_matrix, class_names)

    # Calculate FP, FN, TP, TN
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    # Convert to float to avoid integer division
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Calculate metrics
    TPR = TP / (TP + FN)  # Sensitivity / Recall
    TNR = TN / (TN + FP)  # Specificity
    PPV = TP / (TP + FP)  # Precision
    NPV = TN / (TN + FN)  # Negative predictive value
    FPR = FP / (FP + TN)  # False positive rate
    FNR = FN / (TP + FN)  # False negative rate
    FDR = FP / (TP + FP)  # False discovery rate
    ACC = (TP + TN) / (TP + FP + FN + TN)  # Accuracy

    # Print the metrics
    print("Sensitivity (Recall):", TPR) # How well the model can identify the positive class(High is good)
    print("Specificity:", TNR) # Ability of the model to correctly identify the negative cases(High is good)
    print("Precision:", PPV) # Positive predictions that are actually correct.(High is good)
    print("Negative Predictive Value:", NPV) # Negative predictions that are actually correct(High is good)
    print("False Positive Rate:", FPR) # The model is incorrectly predicting the class(High is bad)
    print("False Negative Rate:", FNR) # Model is missing a lot of positive cases (High is bad)
    print("False Discovery Rate:", FDR) # FDR means that many of the positive predictions are incorrect(High is bad)
    print("Overall Accuracy:", ACC) #A high accuracy means that most predictions (both positive and negative) are correct.

    return TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC

# Function to plot and save confusion matrix
def plot_and_save_confusion_matrix(cnf_matrix, class_names):
    """
    Plot and save confusion matrix.
    
    :param cnf_matrix: Confusion matrix (numpy array).
    :param class_names: List of class names.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cnf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', size = 15)
    plt.ylabel('True Label', fontsize=15)
    plt.title('Confusion Matrix', fontsize=15)
    plt.savefig(os.path.join(MY_DIR, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
    plt.xticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.close()

#-------------------------------Define parameters for the models----------------------------------------------------------------

# Parameters for model 1
rfecv_params = {
    'use_label_encoder': False, 
        'eval_metric': 'auc', 
        'objective': 'multi:softprob',  # Multiclass classification objective for probabilities
        'num_class': len(set(y)),
        'learning_rate': 0.05,
        'n_estimators': 150,
        'max_depth': 3,
        'subsample': 0.8,
        'min_child_weight': 3,
        'n_jobs': -1,
        'reg_lambda': 1,
        'random_state': seed
}

# Parameters model 2 
parameter_after_rfe = {
    'use_label_encoder': False, 
        'eval_metric': 'auc', 
        'objective': 'multi:softprob',  # Multiclass classification objective for probabilities
        'num_class': len(set(y)),
        'learning_rate': 0.048100866649421876,
        'n_estimators': 218,
        'max_depth': 5,
        'subsample': 0.7420759170922717,
        'min_child_weight': 1,
        'n_jobs': -1,
        'reg_lambda': 1,
        'random_state': seed
}

# Parameters for final test model.
final_model_params = {
    'use_label_encoder': False, 
        'eval_metric': 'auc', 
        'objective': 'multi:softprob',  # Multiclass classification objective for probabilities
        'num_class': len(set(y)),
        'learning_rate': 0.048100866649421876,
        'n_estimators': 218,
        'max_depth': 5,
        'subsample': 0.7420759170922717,
        'min_child_weight': 1,
        'n_jobs': -1,
        'reg_lambda': 1,
        'random_state': seed
}

#------------------------------------------------------------------------------------------------------------------------
# model 1:
rfecv_model = build_xgboost_model(rfecv_params)
selected_features_mask, rfecv = perform_rfecv(rfecv_model, X_train, y_train, cv=cv)

# model 2:
cv_model = build_xgboost_model(parameter_after_rfe)
train_model_with_cv(cv_model, X_train, y_train, selected_features_mask, n_splits=cv)

# model 3 (test only):
final_model = build_xgboost_model(final_model_params)
X_train_selected = X_train.loc[:, selected_features_mask]
X_test_selected = X_test.loc[:, selected_features_mask]
train_model_without_cv_and_calculate_scores(final_model, X_train_selected, y_train, X_test_selected, y_test)

# Call this function after testing the final model (Model 3)
evaluate_confusion_matrix(final_model, X_test_selected, y_test, class_names=[0, 1, 2])
