from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from imblearn.metrics import geometric_mean_score
import numpy as np


def get_regression_performance(model_name, fold, is_train, y_true, y_pred):
    # Calculate metrics and update DataFrame
    metrics = {
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true[y_true > 0], y_pred[y_true > 0]),
        'smape': smape(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape_th_0.5': mean_absolute_percentage_error(y_true[y_true > 0.5], y_pred[y_true > 0.5]),
        'mape_th_1': mean_absolute_percentage_error(y_true[y_true > 1], y_pred[y_true > 1]),
        'mape_th_2': mean_absolute_percentage_error(y_true[y_true > 2], y_pred[y_true > 2]),   
    }

    result = {
        "model": model_name,
        "fold": fold,
        "is_train": is_train,
        "rmse": round(metrics['rmse'], 4),
        "mae": round(metrics['mae'], 4),
        "mape": round(metrics['mape'], 4),
        "smape": round(metrics['smape'], 4),
        "r2": round(metrics['r2'], 4), 
        'mape_th_0.5': round(metrics['mape_th_0.5'], 4),
        'mape_th_1': round(metrics['mape_th_1'], 4),
        'mape_th_2': round(metrics['mape_th_2'], 4),
    }

    return result


def get_performance(model_name, fold, is_train, y_true, y_pred):
    # Calculate metrics and update DataFrame
    metrics = {

        'accuracy': accuracy_score(y_true, y_pred),
        'recall_weighted': recall_score(y_true, y_pred, average="weighted"),
        'recall_macro': recall_score(y_true, y_pred, average="macro"),
        'f1_weighted': f1_score(y_true, y_pred, average="weighted"),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'bal_acc': balanced_accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average="weighted"),
        'precision_macro': precision_score(y_true, y_pred, average="macro"),
        'g_mean': geometric_mean_score(y_true, y_pred)
    }

    result = {
        "model": model_name,
        "fold": fold,
        "is_train": is_train,
        'accuracy': round(metrics['accuracy'], 3),
        'recall_weighted': round(metrics['recall_weighted'], 3),
        'recall_macro': round(metrics['recall_macro'], 3),
        'f1_weighted': round(metrics['f1_weighted'], 3),
        'f1_macro': round(metrics['f1_macro'], 3),
        'bal_acc': round(metrics['bal_acc'], 3),
        'precision_weighted': round(metrics['precision_weighted'], 3),
        'precision_macro': round(metrics['precision_macro'], 3),
        'g_mean': round(metrics['g_mean'], 6)
    }

    return result


def smape(y_true, y_pred):
    
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
