"""Implementation functions for project 1."""

import numpy as np

threshold = 1e-4

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def de_standardize(x, mean_x, std_x):
    """De-standardize to the original data set."""
    return x * std_x + mean_x
    
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e**2)

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    epsilon = 1e-15
    pred = 1.0 / (1 + np.exp(-t)) 
    return np.clip(pred, epsilon, 1 - epsilon)

"""Gradient descent"""

def mean_squared_error_gd(y, tx, w_initial, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD
    """
    
    # Define parameters to store w and loss
    ws = [w_initial]
    losses = []
    
    w = w_initial
    for n_iter in range(max_iters):
        # compute loss, gradient
        err = y - tx.dot(w)
        grad = - tx.T.dot(err) / len(err)
        loss = calculate_mse(err)
        # update w by gradient descent
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)

        # convergence criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    print("loss={l}".format(l=losses[-1]))
    return losses, ws
    
"""Stochastic gradient descent"""

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # Number of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, w_initial, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    batch_size = 64
    # Define parameters to store w and loss
    ws = [w_initial]
    losses = []
    w = w_initial

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):
            # compute a stochastic gradient and loss
            err = y_batch - tx_batch.dot(w)
            grad = - tx_batch.T.dot(err) / len(err)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = calculate_mse(err)
            # store w and loss
            ws.append(w)
            losses.append(loss)

            # converge criterion
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
            
    print("loss={l}".format(l=losses[-1]))
    return losses, ws

"""Least squares"""

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    loss = calculate_mse(err)
    return loss, w

"""Ridge regression"""

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    loss = calculate_mse(err)
    return loss, w


""" Logistic regression - gradient descent """

def logistic_regression(y, tx, w_initial, max_iters, gamma):
    
    # Define parameters to store w and loss
    ws = [w_initial]
    losses = []
    
    w = w_initial
    for n_iter in range(max_iters):
        # compute loss, gradient
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y) * (1 / y.shape[0])
        loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        loss = np.squeeze(-loss).item() * (1 / y.shape[0])
        # update w by gradient descent
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)

        # convergence criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    print("loss={l}".format(l=losses[-1]))
    return losses, ws


""" Ridge logistic regression - gradient descent """

def reg_logistic_regression(y, tx, w_initial, max_iters, gamma, lambda_):
    """Regularized logistic regression method.
        
    Args : 
        x = input matrix of the training set (N,D) where N is the number of samples and D the number of features
        y = output vector of the training set(N,) where N is the number of samples
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D, ), for each iteration of GD
    """

    # Define parameters to store w and loss
    ws = [w_initial]
    losses = []
    
    w = w_initial
    for n_iter in range(max_iters):
        # compute loss, gradient
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y) * (1 / y.shape[0]) + 2 * lambda_ * w     
        loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        loss = np.squeeze(-loss).item() * (1 / y.shape[0]) + lambda_ * np.squeeze(w.T.dot(w))
        # update w by gradient descent
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)

        # convergence criterion
        # if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
        #    break

    print("loss={l}".format(l=losses[-1]))
    return losses, ws