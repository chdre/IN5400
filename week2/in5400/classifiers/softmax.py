import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    """
    N = 500, minibatch size (number of pictures)
    D = 3073, pixels in ONE image
    C = 10, labels (classes) (item on photo)
    => X: (500,3073), W: (3073,10), y: (500,)

    Looking at one image - y - we try to predict what the image is by minimizing
    the loss function (softmax loss function).
    """

    n_x = X.shape[0]    # Input dimension
    n_y = W.shape[1]    # Number of classes

    for i in range(n_x):
        z = X[i].dot(W)
        z -= np.max(z)  # Stability

        s_sum = 0
        for j in range(n_y):
            s_sum += np.exp(z[j])

        # Stability, to get rid of division
        t = z - np.log(s_sum)
        prob = np.exp(t)   # Output probability

        loss -= t[y[i]] # np.log(s)   # loss -= t

        for k in range(n_y):
            dW[:,k] += (prob[k] - (k == y[i])) * X[i]  # X[i] "input layer"

    loss /= n_x
    dW /= n_x
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    z = X.dot(W)
    z -= np.max(z)

    s_sum = np.sum(np.exp(z), axis=0)

    t = z - np.log(s_sum[np.newaxis, :])

    prob = np.exp(t)

    loss -= np.sum(t)

    y_hat = np.zeros_like(t)
    y_hat[np.arange(X.shape[0]), y] = 1
    dW = X.T.dot(prob - y_hat)


    loss /= X.shape[0]
    dW /= X.shape[0]

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
