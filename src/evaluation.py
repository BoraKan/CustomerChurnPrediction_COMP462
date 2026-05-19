import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)


def get_metrics(y_true, y_pred, y_prob=None):
    results = {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, zero_division=0),
        'MCC':       matthews_corrcoef(y_true, y_pred)
    }
    if y_prob is not None:
        results['ROC-AUC'] = roc_auc_score(y_true, y_prob)
    return results


def plot_confusion_matrix(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title(title)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')


def plot_roc(models, X_test, y_test, ax):
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()


def plot_precision_recall(models, X_test, y_test, ax):
    """Precision-Recall curves — more informative than ROC for imbalanced classes."""
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, label=f'{name} (AP={ap:.3f})')
    baseline = y_test.mean()
    ax.axhline(baseline, color='k', linestyle='--', label=f'No-skill ({baseline:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()


def find_optimal_threshold(y_true, y_prob):
    """Return the probability threshold that maximises F1 on the given set."""
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return thresholds[best_idx], f1s[best_idx]


def plot_business_cost(y_true, y_prob, model_name, ax,
                       cost_fn=500, cost_fp=50):
    """
    Total expected cost across thresholds.
    cost_fn: revenue lost per missed churner (false negative).
    cost_fp: retention campaign cost per non-churner contacted (false positive).
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        fn = cm[1, 0]
        fp = cm[0, 1]
        costs.append(fn * cost_fn + fp * cost_fp)

    best_idx = int(np.argmin(costs))
    ax.plot(thresholds, costs, color='steelblue')
    ax.axvline(thresholds[best_idx], color='red', linestyle='--',
               label=f'Optimal t={thresholds[best_idx]:.2f}\nCost=${costs[best_idx]:,.0f}')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Total Cost ($)')
    ax.set_title(f'Business Cost — {model_name}')
    ax.legend()
    return thresholds[best_idx], costs[best_idx]
