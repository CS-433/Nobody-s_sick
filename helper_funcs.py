"""Some helper functions for project 1."""

import csv
import numpy as np
import os

from implementations import * 

def highly_correlated_features(w, feature_cat_map, feature_cat_encoded_map, feature_cont_map, categorical_features, continuous_features, top_n_cat, top_n_cont):
    
    w_cat = w[1:len(feature_cat_map)+1]
    w_cont = w[len(feature_cat_map)+1:]

    sorted_indices_cat = np.argsort(w_cat)
    top_n_indices_cat = sorted_indices_cat[-top_n_cat:]
    top_n_weights_cat = w_cat[top_n_indices_cat]
    
    sorted_indices_cont = np.argsort(w_cont)
    top_n_indices_cont = sorted_indices_cont[-top_n_cont:]
    top_n_weights_cont = w_cont[top_n_indices_cont]

    cat_feat_idx = feature_cat_map[top_n_indices_cat]
    cont_feat_idx = feature_cont_map[top_n_indices_cont]

    correlated_cat_feat = categorical_features[cat_feat_idx]
    correlated_cont_feat = continuous_features[cont_feat_idx]

    return correlated_cat_feat, feature_cat_encoded_map[top_n_indices_cat], top_n_weights_cat, correlated_cont_feat, top_n_weights_cont
    
def get_opt_parameter(metric_name, metrics, ws, parameter):
    """Get the best w from the result the optimization algorithm."""
    if metric_name == 'f1_score':
        metric = metrics[:,0]
        return metric[np.argmax(metric)], parameter[np.argmax(metric)], ws[np.argmax(metric)], np.argmax(metric)
    elif metric_name == 'RMSE':
        metric = metrics[:,1]
        return metric[np.argmin(metric)], parameter[np.argmin(metric)], ws[np.argmin(metric)], np.argmin(metric)
        

def print_report(opt_w, is_LR, tx_training_balanced, y_training_balanced, tx_train_validation, y_train_validation, tx_test):

    if is_LR:
        threshold = 0
    else:
        threshold = 0.5
    
    print('---------------- True Vs. Predicted positive class (Heart Attack Rate) ---------------- \n')
    
    # train set balanced
    sick_train_balanced = np.sum(y_training_balanced == 1)/ len(y_training_balanced)
    y_train_balanced_pred = tx_training_balanced.dot(opt_w)
    y_train_balanced_pred = np.where(y_train_balanced_pred > threshold, 1, 0)
    sick_train_balanced_pred = np.sum(y_train_balanced_pred == 1)/len(y_train_balanced_pred)
    print('Train set (balanced):\nTrue {t:.3f}, Predicted {p:.3f}.'.format(t=sick_train_balanced, p=sick_train_balanced_pred))
    
    # validation set
    sick_validation = np.sum(y_train_validation == 1)/ len(y_train_validation)
    y_validation_pred = tx_train_validation.dot(opt_w)
    y_validation_pred = np.where(y_validation_pred > threshold, 1, 0)
    sick_validation_pred = np.sum(y_validation_pred == 1)/ len(y_validation_pred)
    print('Validation set:\nTrue {t:.3f}, Predicted {p:.3f}.'.format(t=sick_validation, p=sick_validation_pred))
    
    # test set
    y_test_pred = tx_test.dot(opt_w)
    y_test_pred = np.where(y_test_pred > threshold, 1, 0)
    sick_test_pred = np.sum(y_test_pred == 1)/ len(y_test_pred)
    print('Test set:\nPredicted {p:.3f}.'.format(p=sick_test_pred))
    
def hyperparam_optimization(metric_name, metrics, ws, params, param_name, tx_training_balanced, y_training_balanced, tx_train_validation, y_train_validation, tx_test, is_LR):

    opt_metric, opt_param, opt_w, opt_idx = get_opt_parameter(metric_name, metrics, ws, params)

    print('The optimal parameter is {param}={p:.6f} given optimization of the metric {metr} evaluating {m:.5f}.\n'.format(param = param_name, p=opt_param, metr=metric_name, m=opt_metric))
    # print('The optimal weights are w = {}\n.'.format(opt_w))

    print('*******************************\n')
    
    # True Vs. Predicted positive class (Heart Attack Rate)
    print_report(opt_w, is_LR, tx_training_balanced, y_training_balanced, tx_train_validation, y_train_validation, tx_test)
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


def train_vs_valid(tx_training, y_training, ws, learning_rate, is_LR):

    if is_LR:
        threshold = 0
    else:
        threshold = 1
        
    rmse_training = np.zeros(len(learning_rate))
    for idx, gamma in enumerate(learning_rate):
        
        w = ws[idx]
        
        # Training (balanced) Vs. Validation error
        y_pred = tx_training.dot(w)
        y_pred = np.where(y_pred > threshold, 1, 0)
        rmse_training[idx] = np.sqrt(calculate_mse(y_training - y_pred))
        
    return rmse_training