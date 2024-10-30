"""Some helper functions for project 1."""

import csv
import numpy as np
import os

from implementations import * 


def get_opt_parameter(metric_name, metrics, ws, parameter):
    """Get the best w from the result the optimization algorithm."""
    if metric_name == 'f1_score':
        metric = metrics[:,0]
        return metric[np.argmax(metric)], parameter[np.argmax(metric)], ws[np.argmax(metric)], np.argmax(metric)
    elif metric_name == 'RMSE':
        metric = metrics[:,1]
        return metric[np.argmin(metric)], parameter[np.argmin(metric)], ws[np.argmin(metric)], np.argmin(metric)
        
def get_eval_metrics(metrics, opt_idx):
    f1_score = metrics[opt_idx, 0]
    rmse = metrics[opt_idx, 1]
    return f1_score, rmse

def print_report(opt_w, is_LR, tx_training_balanced, y_training_balanced, tx_training_imbalanced, y_training_imbalanced, tx_train_validation, y_train_validation, tx_test):

    if is_LR:
        threshold = 0
    else:
        threshold = 0.5
    
    print('True Vs. Predicted positive class (Heart Attack Rate) \n')
    # train set balanced
    sick_train_balanced = np.sum(y_training_balanced == 1)/ len(y_training_balanced)
    y_train_balanced_pred = tx_training_balanced.dot(opt_w)
    y_train_balanced_pred = np.where(y_train_balanced_pred > threshold, 1, 0)
    sick_train_balanced_pred = np.sum(y_train_balanced_pred == 1)/len(y_train_balanced_pred)
    print('Train set balanced: True {t:.3f}, Predicted {p:.3f}.'.format(t=sick_train_balanced, p=sick_train_balanced_pred))
    
    # train set imbalanced
    sick_train_imbalanced = np.sum(y_training_imbalanced == 1)/ len(y_training_imbalanced)
    y_train_imbalanced_pred = tx_training_imbalanced.dot(opt_w)
    y_train_imbalanced_pred = np.where(y_train_imbalanced_pred > threshold, 1, 0)
    sick_train_imbalanced_pred = np.sum(y_train_imbalanced_pred == 1)/ len(y_train_imbalanced_pred)
    print('Train set original: True {t:.3f}, Predicted {p:.3f}.'.format(t=sick_train_imbalanced, p=sick_train_imbalanced_pred))
    
    # validation set
    sick_validation = np.sum(y_train_validation == 1)/ len(y_train_validation)
    y_validation_pred = tx_train_validation.dot(opt_w)
    y_validation_pred = np.where(y_validation_pred > threshold, 1, 0)
    sick_validation_pred = np.sum(y_validation_pred == 1)/ len(y_validation_pred)
    print('Validation set: True {t:.3f}, Predicted {p:.3f}.'.format(t=sick_validation, p=sick_validation_pred))
    
    # test set
    y_test_pred = tx_test.dot(opt_w)
    y_test_pred = np.where(y_test_pred > threshold, 1, 0)
    sick_test_pred = np.sum(y_test_pred == 1)/ len(y_test_pred)
    print('Test set: Predicted {p:.3f}.'.format(p=sick_test_pred))
    
def hyperparam_optimization(metric_name, metrics, ws, params, param_name, tx_training_balanced, y_training_balanced, tx_train_training, y_train_training, tx_train_validation, y_train_validation, tx_test, is_LR):

    opt_metric, opt_param, opt_w, opt_idx = get_opt_parameter(metric_name, metrics, ws, params)
    f1_score, rmse = get_eval_metrics(metrics, opt_idx)

    print('The optimal parameter is {param}={p:.6f} given optimization of the metric {metr} evaluating {m:.5f}.'.format(param = param_name, p=opt_param, metr=metric_name, m=opt_metric))
    print('The optimal weights are w = {}.'.format(opt_w))
    print('f1 score = {f:.5f}, RMSE = {r:.5f}\n'.format(f=f1_score, r=rmse))

    print('*******************************\n')
    
    # True Vs. Predicted positive class (Heart Attack Rate)
    print_report(opt_w, is_LR, tx_training_balanced, y_training_balanced, tx_train_training, y_train_training, tx_train_validation, y_train_validation, tx_test)
    return opt_idx
    

def confusion_matrix_metrics(y_true, y_pred):
    # Ensure that y_true and y_pred are 1D arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Get the unique classes (assuming binary classification: 0 and 1)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    if len(classes) != 2:
        raise ValueError("This function is designed for binary classification (two classes).")
    
    # Initialize counts
    tp = tn = fp = fn = 0

    # Calculate TP, TN, FP, FN
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1  # True Positive
        elif true == 0 and pred == 0:
            tn += 1  # True Negative
        elif true == 0 and pred == 1:
            fp += 1  # False Positive
        elif true == 1 and pred == 0:
            fn += 1  # False Negative

    return tp, tn, fp, fn


def train_vs_valid(tx_training_balanced, y_training_balanced, tx_training_imbalanced, y_training_imbalanced, ws, learning_rate):
    rmse_training_balanced = np.zeros(len(learning_rate))
    rmse_training_imbalanced = np.zeros(len(learning_rate))
    
    for idx, gamma in enumerate(learning_rate):
        
        w = ws[idx]
        
        # Training (balanced) Vs. Validation error
        y_pred_balanced = tx_training_balanced.dot(w)
        y_pred_balanced = np.where(y_pred_balanced > 0, 1, 0)
        rmse_training_balanced[idx] = np.sqrt(calculate_mse(y_training_balanced - y_pred_balanced))
        # Training (imbalanced) Vs. Validation error
        y_pred_imbalanced = tx_training_imbalanced.dot(w)
        y_pred_imbalanced = np.where(y_pred_imbalanced > 0, 1, 0)
        rmse_training_imbalanced[idx] = np.sqrt(calculate_mse(y_training_imbalanced - y_pred_imbalanced))
    
    return rmse_training_balanced, rmse_training_imbalanced