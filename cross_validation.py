"""Cross validation functions for project 1."""

import numpy as np
import matplotlib.pyplot as plt

from implementations import *

def compute_mse(e):
    """Compute the loss by mean squared error."""
    mse = (e.T.dot(e) / (2 * len(e)))[0, 0]  # Ensures a scalar output
    return mse

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold cross-validation."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_gd(y, x, k_indices, k, gamma):
    """Return the loss of ridge regression for a fold corresponding to k_indices."""
    # Get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    # Gradient Descent
    _, ws = mean_squared_error_gd(y_tr, x_tr, w_initial=np.zeros((x_tr.shape[1],1)), max_iters=50, gamma=gamma)
    w = ws[-1]  
    
    # Calculate the loss for train and test data
    y_pred_tr = x_tr.dot(w)
    y_pred_tr = np.where(y_pred_tr > threshold, 1, 0)

    y_pred_te = x_te.dot(w)
    y_pred_te = np.where(y_pred_te > threshold, 1, 0)
    
    loss_tr = np.sqrt(2 * compute_mse(y_tr - y_pred_tr))
    loss_te = np.sqrt(2 * compute_mse(y_te - y_pred_te))
    return loss_tr, loss_te

def cross_validation_sgd(y, x, k_indices, k, gamma):
    """Return the loss of ridge regression for a fold corresponding to k_indices."""
    # Get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    # Gradient Descent
    _, ws = mean_squared_error_sgd(y_tr, x_tr, w_initial=np.zeros((x_tr.shape[1],1)), max_iters=50, gamma=gamma)
    w = ws[-1]  
    
    # Calculate the loss for train and test data
    y_pred_tr = x_tr.dot(w)
    y_pred_tr = np.where(y_pred_tr > threshold, 1, 0)

    y_pred_te = x_te.dot(w)
    y_pred_te = np.where(y_pred_te > threshold, 1, 0)
    
    loss_tr = np.sqrt(2 * compute_mse(y_tr - y_pred_tr))
    loss_te = np.sqrt(2 * compute_mse(y_te - y_pred_te))
    return loss_tr, loss_te
    
def cross_validation_ridge(y, x, k_indices, k, lambda_):
    """Return the loss of ridge regression for a fold corresponding to k_indices."""
    # Get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    # Ridge regression
    _, w = ridge_regression(y_tr, x_tr, lambda_)
    
    # Calculate the loss for train and test data
    y_pred_tr = x_tr.dot(w)
    y_pred_tr = np.where(y_pred_tr > threshold, 1, 0)

    y_pred_te = x_te.dot(w)
    y_pred_te = np.where(y_pred_te > threshold, 1, 0)
    
    loss_tr = np.sqrt(2 * compute_mse(y_tr - y_pred_tr))
    loss_te = np.sqrt(2 * compute_mse(y_te - y_pred_te))
    return loss_tr, loss_te

def cross_validation_log(y, x, k_indices, k, gamma):
    """Return the loss of ridge regression for a fold corresponding to k_indices."""
    # Get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    # Logistic regression
    _, ws = logistic_regression(y_tr, x_tr, w_initial=np.zeros((x_tr.shape[1],1)), max_iters=50, gamma=gamma)
    w = ws[-1]  
    
    # Calculate the loss for train and test data
    y_pred_tr = x_tr.dot(w)
    y_pred_tr = np.where(y_pred_tr > threshold, 1, 0)

    y_pred_te = x_te.dot(w)
    y_pred_te = np.where(y_pred_te > threshold, 1, 0)
    
    loss_tr = np.sqrt(2 * compute_mse(y_tr - y_pred_tr))
    loss_te = np.sqrt(2 * compute_mse(y_te - y_pred_te))
    return loss_tr, loss_te

def cross_validation_demo(y, x, k_fold, params, param_name, ML_model):
    """Cross-validation over regularization parameter lambda."""
    seed = 12
    k_indices = build_k_indices(y, k_fold, seed)
    rmse_tr = []
    rmse_te = []

    for param in params:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            if ML_model == 'GD':
                loss_tr, loss_te = cross_validation_gd(y, x, k_indices, k, param)
            elif ML_model == 'SGD':
                loss_tr, loss_te = cross_validation_sgd(y, x, k_indices, k, param)
            elif ML_model == 'RIDGE':
                loss_tr, loss_te = cross_validation_ridge(y, x, k_indices, k, param)
            elif ML_model == 'LOGISTIC':
                loss_tr, loss_te = cross_validation_log(y, x, k_indices, k, param)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))

    best_param = params[np.argmin(rmse_te)]
    best_rmse = np.min(rmse_te)

    cross_validation_visualization(param_name, params, rmse_tr, rmse_te)
    print(
        "The best gamma is {:.5f} with a test rmse of {:.3f}.".format(best_param, best_rmse)
    )
    return best_param, best_rmse

def cross_validation_visualization(param_name, params, rmse_tr, rmse_te):
    """Visualization of the curves of rmse_tr and rmse_te."""
    plt.semilogx(params, rmse_tr, marker=".", color="b", label="train error")
    plt.semilogx(params, rmse_te, marker=".", color="r", label="test error")
    plt.xlabel(param_name)
    plt.ylabel("RMSE")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()

def cross_validation_reg_logistic(y, x, k_indices, k, lambda_, gamma):
    """Return the loss of ridge regression for a fold corresponding to k_indices."""
    # Get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    
    # Regularized logistic regression
    _, ws = reg_logistic_regression(y_tr, x_tr, w_initial=np.zeros((x_tr.shape[1], 1)), max_iters=50, gamma=gamma, lambda_=lambda_)
    w = ws[-1]  # Get the last weight vector after training
    
    # Calculate the loss for train and test data
    y_pred_tr = x_tr.dot(w)
    y_pred_tr = np.where(y_pred_tr > threshold, 1, 0)

    y_pred_te = x_te.dot(w)
    y_pred_te = np.where(y_pred_te > threshold, 1, 0)
    
    loss_tr = np.sqrt(2 * compute_mse(y_tr - y_pred_tr))
    loss_te = np.sqrt(2 * compute_mse(y_te - y_pred_te))
    return loss_tr, loss_te

def best_gamma_and_lambda_selection(y, x, k_fold, gammas, lambdas, seed=1):
    """Cross-validation to select the best lambda and gamma for regularized logistic regression."""
    
    # Split data in k-fold
    k_indices = build_k_indices(y, k_fold, seed)

    # Track best lambda and rmse for each gamma
    best_lambdas = []
    best_rmses = []

    # Loop over gamma values
    for gamma in gammas:
        # Track RMSEs for the current gamma over different lambdas
        rmse_te = []
        
        for lambda_ in lambdas:
            rmse_te_tmp = []
            
            # Cross-validation for each fold
            for k in range(k_fold):
                _, loss_te = cross_validation_reg_logistic(y, x, k_indices, k, lambda_, gamma)
                rmse_te_tmp.append(loss_te)
                
            # Average RMSE for this lambda across all folds
            rmse_te.append(np.mean(rmse_te_tmp))

        # Find the best lambda for the current gamma
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])

    # Select the gamma with the minimum test RMSE
    ind_best_gamma = np.argmin(best_rmses)
    best_gamma = gammas[ind_best_gamma]
    best_lambda = best_lambdas[ind_best_gamma]
    best_rmse = best_rmses[ind_best_gamma]

    return best_gamma, best_lambda, best_rmse