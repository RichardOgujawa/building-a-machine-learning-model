# NOTE, if something is being defined or redefined inside of a function, it will be prefixed with the name of that function for identification purposes

import numpy as np
from prettytable import PrettyTable


def main():
    # Calculate the values for the three nodes
    # Initial Prediction
    # Use .item() method to extract single element from 1 x 1 matrix
    main_yhat = yhat(w2_matrix, a1_matrix(w1_matrix, x)).item()
    # Initial Loss
    print(
        f"Initial Prediction: {main_yhat}, \n Initial Loss: {loss(y, main_yhat)} ")
    backProp(10000, y, w1_matrix, w2_matrix, x)


# DEFINE THE UTILITY FUNCTIONS: SIGMOID AND DSIGMOID
# Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# dsigmoid is the derivative of sigmoid
def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


# First hidden layer which is going to simply return the function required to get a1
def a1_matrix(w1_matrix, x):
    return sigmoid(np.dot(w1_matrix, x))


# The Prediction, yhat
def yhat(w2_matrix, a1_matrix):
    # Reshape it as a column vector
    # Apply sigmoid activation function to the product of the weight matrix and the matrix of activation nodes
    return sigmoid(np.dot(w2_matrix, a1_matrix))


# Loss function
def loss(y, yhat):
    return (y - yhat)**2
    # If there was more than one yhat we'd need to get np.sum of all the (y -yhats)**2


# Gradient Descent for weight 1
def gradw1(y, w2_matrix, w1_matrix, x):
    # Initial Variables
    gw1_a1_matrix = a1_matrix(w1_matrix, x)
    gw1_yhat = yhat(w2_matrix, gw1_a1_matrix)

    # Calculus
    dL_dyhat = np.array(-2*(y-gw1_yhat))  # 1 x 1 - scalar
    # print(dL_dyhat.shape)
    dyhat_dz2_matrix = dsigmoid(np.dot(w2_matrix, gw1_a1_matrix))  # 1 x 3
    # print(dyhat_dz2_matrix.shape)
    dz2_matrix_da1_matrix = w2_matrix  # 3 x 3
    # print(dz2_matrix_da1_matrix.shape)
    da1_matrix_dz1_matrix = dsigmoid(
        np.dot(w2_matrix, gw1_a1_matrix))  # 3 x 3
    # print(da1_matrix_dz1_matrix.shape)
    dz1_matrix_dw1_matrix = x

    #  You need to derive a function for dL/dW1 and return that
    # Since we're doing elementwise multiplication as opposed to matrix multiplication we can just do * instead of np.dot when. Np.dot won't work.
    return (dL_dyhat * dz2_matrix_da1_matrix * dyhat_dz2_matrix * dz2_matrix_da1_matrix
            * da1_matrix_dz1_matrix * dz1_matrix_dw1_matrix).T


# Gradient Descent for Weight 2
def gradw2(y, w2_matrix, w1_matrix, x):
    # Initial Variables
    gw2_a1_matrix = a1_matrix(w1_matrix, x)
    gw2_yhat = yhat(w2_matrix, gw2_a1_matrix)

    # Calculus
    dL_dyhat = np.array(-2*(y-gw2_yhat))  # 1 x 1 - scalar
    # print(dL_dyhat.shape)
    dyhat_dz2_matrix = dsigmoid(np.dot(w2_matrix, gw2_a1_matrix))  # 1 x 3
    # print(dyhat_dz2_matrix.shape)
    dz2_matrix_dw2_matrix = gw2_a1_matrix  # 3 x 3

    #  You need to derive a function for dL/dW1 and return that
    return (dL_dyhat * dyhat_dz2_matrix * dz2_matrix_dw2_matrix).T


"""
# THE LOGIC BEHIND THE COMPOSITION OF THE GRADIENT
w212 = w 2, 1 (2) = weight that links node 2 to output node 1 in the second layer
d stands for differentiation
L = (y-yhat)^2 <== yhat = sigma(z2_matrix) <== z2_matrix = w2_matrix * a1_matrix <== a1_matrix = sigma(z1_matrix) <== z1_matrix = x * w1_matrix <== x
--- Weights in first layer ---
dL/dw1_matrix = dL/dyhat * dyhat/dz2_matrix * dz2_matrix/da1_matrix * da1_matrix/dz1_matrix * dz1_matrix/dw1_matrix
1) L = (y-yhat)^2 || dL/dyhat = 2u * -1 = -2(y-yhat) || 1 x 1 (scalar)
2) yhat = sigma(z2_matrix) || dyhat/dz2_matrix = dsigmoid(u) * 1 = dsigmoid(z2_matrix) || 3 x 1 ==> Transpose ==> 1 x 3, get sigma for each z value
3) z2_matrix = w2_matrix * a1_matrix || dz2_matrix/da1_matrix = w2_matrix || 3 x 3, because you have to differentiate each component of z with respect 
to each component in a. 
4) a1_matrix = sigmoid(z1_matrix) || da1_matrix/dz1_matrix = dsigmoid(z1_matrix) || 3 x 1
5) z1_matrix = x * w1_matrix || dz1_matrix / dw1_matrix = x || 1
-- Because da1_matrix/dz1_matrix is a diagonal matrix then it's more or less just a 3 x 1 matrix which you can multiply by x to still get a 3 x 1 matrix, 
which will be able to minus the original weights from.
--- Weights in second layer --- 
dL/dw2_matrix = dL/dyhat * dyhat/dz2_matrix * dz2_matrix/dw2_matrix 
1) L = (y-yhat)^2 || dL/dyhat = 2u * -1 = -2(y-yhat)
2) yhat = sigma(z2_matrix) || dyhat/dz2_matrix = dsigmoid(u) * 1 = dsigmoid(z2_matrix)
3) z2_matrix = w2_matrix * a1_matrix || dz2_matrix/dw2_matrix = a1_matrix
"""


# Now we can pass in some parameters
# Initial Input
x = 1
# Reshape it as a column vector
w1_matrix = np.array([0.4, 0.3, 0.7]).reshape(3, 1)
# Second level weights
w2_matrix = np.array([[0.8, 0.6, 0.9]])
# Desired output
y = 0.5
# Step Size
eta = 0.1

# Now we can do BackPropagation to update the weight matrix


def backProp(iter, y, w1_matrix, w2_matrix, x):
    table = PrettyTable()
    table.field_names = ["Iteration", "Prediction (ŷ)", "Loss (L)"]
    for i in range(iter):
        w1_matrix = w1_matrix - eta*gradw1(y, w2_matrix, w1_matrix, x)
        w2_matrix = w2_matrix - eta*gradw2(y, w2_matrix, w1_matrix, x)
        yh = yhat(w2_matrix, a1_matrix(w1_matrix, x))
        # Instead of printing out everything, just print when i is divisible by 100 with no remainders.
        if not i % (iter/100):
            table.add_row([i, yh.item(), loss(y, yh).item()])
    print(table)


if __name__ == "__main__":
    main()
