#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1.1
    vals = np.asarray(conf['layer_dimensions'])
    layers = vals.shape[0]

    scale = np.sqrt(2 / vals)

    params = {}

    for i in range(1, layers):
        W = np.random.normal(scale=scale[i - 1], size=(vals[i - 1], vals[i]))
        b = np.zeros((vals[i], 1))

        params[f'W_{i}'] = W
        params[f'b_{i}'] = b

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 a)
    if activation_function == 'relu':
        return np.maximum(0, Z)
    elif activation_function == 'leaky_relu':
        return np.maximum(0.01 * Z, Z)
    elif activation_function == 'tanh':
        return np.tanh(Z)
    elif activation_function == 'sigmoid':
        return 1 / (1 + np.exp(-Z))
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 b)
    Z_new = Z - np.max(Z)  # So not to overwrite features[Z_L]
    t = Z_new - np.log(np.sum(np.exp(Z_new), axis=0)[np.newaxis, :])

    return np.exp(t)


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 1.2 c)
    layers = len(conf['layer_dimensions'])

    features = {}
    features['A_0'] = X_batch   # Initializing with the image

    # Hidden layers
    for i in range(1, layers):
        W = params[f'W_{i}']
        a = features[f'A_{i-1}']
        b = params[f'b_{i}']

        Z = W.T @ a + b
        features[f'Z_{i}'] = Z
        features[f'A_{i}'] = activation(Z, conf['activation_function'])

    Y_proposed = softmax(features['Z_' + str(layers - 1)])

    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 1.3
    m = Y_proposed.shape[1]

    cost = - 1 / m * \
        np.sum(np.sum(Y_reference * np.log(Y_proposed), axis=1), axis=0)

    num_correct = np.sum((np.argmax(Y_proposed, axis=0) ==
                          np.argmax(Y_reference, axis=0)))

    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.4 a)
    if activation_function == 'relu':
        return np.heaviside(Z, 1)
    elif activation_function == 'leaky_relu':
        d = np.zeros_like(Z)
        d[Z <= 0] = 0.01
        d[Z > 0] = 1
        return d
    elif activation_function == 'tanh':
        return 1 - np.square(Z)
    elif activation_function == 'sigmoid':
        return Z * (1 - Z)
    else:
        print("Error: Unimplemented derivative of activation function: {}",
              activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 1.4 b)
    layers = len(conf['layer_dimensions'])

    m = Y_proposed.shape[1]
    m_inv = 1 / m   #

    grad_params = {}

    # Last layer
    delta_old = Y_proposed - Y_reference    # Derivative of the cost function wrt z
    grad_params[f'grad_W_{layers - 1}'] = features[f'A_{layers - 2}'] \
        @ delta_old.T * m_inv
    grad_params[f'grad_b_{layers - 1}'] = delta_old @ np.ones((m, 1)) * m_inv

    # All but last layer
    for i in range(layers - 2, 0, -1):
        w_next = params[f'W_{i+1}']
        a_prev = features[f'A_{i-1}']
        z = features[f'Z_{i}']

        g_prime = activation_derivative(z, conf['activation_function'])

        delta = g_prime * (w_next @ delta_old)
        grad_params[f'grad_W_{i}'] = a_prev @ delta.T / m
        grad_params[f'grad_b_{i}'] = delta @ np.ones((m, 1)) / m

        delta_old = delta

    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    # TODO: Task 1.5
    lr = conf['learning_rate']

    updated_params = {}

    for key, vals in params.items():
        updated_params[key] = vals - lr * grad_params[f'grad_{key}']

    return updated_params
