import numpy as np
import math

def calculate_z(w, x, b):
    # z = wx + b
    return w @ x + b

def calculate_x_hidden(z):
    # ReLU function
    x = []

    for elem in z:
        if elem < 0:
            x.append(0)
        else:
            x.append(elem)
    
    return np.transpose(np.array(x))

def calculate_x_output(z):
    # SoftMax function
    x = []
    total = sum(math.exp(elem) for elem in z)

    for elem in z:
        x.append(math.exp(elem)/total)
    
    return np.transpose(np.array(x))


def forward(x0, b1, b2, w1, w2, debug):
    z1 = calculate_z(w1, x0, b1)
    if debug: print("z1=\n",z1.reshape(-1, 1))

    x1 = calculate_x_hidden(z1)
    if debug: print("x1=\n",x1.reshape(-1, 1))

    z2 = calculate_z(w2, x1, b2)
    if debug: print("z2=\n",z2.reshape(-1, 1))

    x2 = calculate_x_output(z2)
    if debug: print("x2=\n",x2.reshape(-1, 1))

    return x1, x2

def derivativeReLU(x):
    # ReLU function derivative -> if elem > 0, then it turns 1
    der_x = []

    for elem in x:
        if elem < 0:
            der_x.append(0)
        else:
            der_x.append(1)
    
    return np.transpose(np.array(der_x))


def backward(t, x1, x2, w2, debug):
    g2 = x2 - t
    if debug: print("g2=\n",g2.reshape(-1, 1))

    g1 = np.transpose(w2) @ g2 * derivativeReLU(x1)
    if debug: print("g1=\n",g1.reshape(-1, 1))

    return g1, g2

def update_weight(w, g, x, learning_rate):
    return w - learning_rate * np.outer(g, x)

def update_bias(b, g, learning_rate):
    return b - learning_rate * g


def learn(x0, t, b1, b2, w1, w2, learning_rate, debug):

    x1, x2 = forward(x0, b1, b2, w1, w2, debug)
    g1, g2 = backward(t, x1, x2, w2, debug)

    w1 = update_weight(w1, g1, x0, learning_rate)
    b1 = update_bias(b1, g1, learning_rate)

    w2 = update_weight(w2, g2, x1, learning_rate)
    b2 = update_bias(b2, g2, learning_rate)

    print("RESULTS")
    print("w1:\n", w1)
    print("b1:\n", b1.reshape(-1, 1))
    print("w2:\n", w2)
    print("b2:\n", b2.reshape(-1, 1))
