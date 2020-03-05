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

"""Implementation of convolution forward and backward pass"""

import numpy as np


def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_layer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """
    # TODO: Task 2.1
    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    (num_filters, channels_w, height_w, width_w) = weight.shape

    K = pad_size

    height_x_str = int(1 + (height_x + 2 * pad_size - height_w) / stride)
    width_x_str = int(1 + (width_x + 2 * pad_size - width_w) / stride)

    output_layer = np.zeros(
        (batch_size, num_filters, height_x_str, width_x_str))

    input_layer = np.pad(input_layer, ((0, 0), (0, 0),
                                       (K, K), (K, K)),  'constant')

    for j in range(num_filters):
        for i in range(batch_size):
            for p_ind, p in enumerate(range(1, height_x + 1, stride)):
                for q_ind, q in enumerate(range(1, width_x + 1, stride)):
                    sum = 0
                    for r in range(-K, K + 1, 1):
                        for s in range(-K, K + 1, 1):
                            sum += (input_layer[i, :, p + r, q + s]
                                    * weight[j, :, r + 1, s + 1])
                    sum = np.sum(sum, axis=0)   # Summing over channels
                    output_layer[i, j, p_ind, q_ind] = sum + bias[j]

    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    return output_layer


def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    # TODO: Task 2.2
    input_layer_gradient = None

    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    bias_gradient = np.zeros(num_filters)
    weight_gradient = np.zeros((num_filters, channels_w, height_w, width_w))
    input_layer_gradient = np.zeros(
        (batch_size, channels_x, height_x, width_x))

    K = pad_size

    input_layer = np.pad(input_layer, ((0, 0), (0, 0),
                                       (K, K), (K, K)),  'constant')
    output_layer_gradient = np.pad(output_layer_gradient, ((0, 0), (0, 0),
                                                           (K, K), (K, K)),  'constant')

    for j in range(num_filters):
        for k in range(channels_x):
            for r in range(-K, K + 1, 1):
                for s in range(-K, K + 1, 1):
                    b_sum = 0
                    w_sum = 0
                    for p in range(1, height_y + 1):
                        for q in range(1, width_y + 1):
                            w_sum += (output_layer_gradient[:, j, p, q]
                                      * input_layer[:, k, p + r, q + s])
                            b_sum += output_layer_gradient[:, j, p, q]
                    weight_gradient[j, k, r + 1, s + 1] = np.sum(w_sum)
        bias_gradient[j] = np.sum(b_sum)

    for k in range(channels_x):
        for p in range(1, height_x + 1):
            for q in range(1, width_x + 1):
                sum = 0
                for j in range(channels_y):
                    for r in range(-K, K + 1, 1):
                        for s in range(-K, K + 1, 1):
                            sum += (output_layer_gradient[:, j, p + r, q + s]
                                    * weight[j, k, -r + 1, -s + 1])
                input_layer_gradient[:, k, p - 1, q - 1] = sum

    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    return input_layer_gradient, weight_gradient, bias_gradient


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
