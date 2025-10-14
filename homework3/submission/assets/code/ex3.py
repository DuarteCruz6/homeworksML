import numpy as np

def linear_forward(weights, inputs, bias):
    # z = wx + b
    return weights @ inputs + bias

def relu_activation(z):
    # ReLU function
    return np.maximum(0, z)

def softmax_activation(z):
    # SoftMax function
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


def forward(x0, b1, b2, w1, w2, debug):
    z1 = linear_forward(w1, x0, b1)
    if debug: print("z1=\n",z1.reshape(-1, 1))

    x1 = relu_activation(z1)
    if debug: print("x1=\n",x1.reshape(-1, 1))

    z2 = linear_forward(w2, x1, b2)
    if debug: print("z2=\n",z2.reshape(-1, 1))

    x2 = softmax_activation(z2)
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

# data provided in question

w1_0 = np.array([1,0,0,0,0])
w1_1 = np.array([-1,1,-1,1,0])
w1_2 = np.array([1,1,1,1,1])
w1 = np.array([w1_0,w1_1,w1_2])

w2_0 = np.array([0,0,0])
w2_1 = np.array([2,0,0])
w2 = np.array([w2_0,w2_1])

b1 = np.transpose(np.array([0,0,0]))
b2 = np.transpose(np.array([0,0]))

x0 = np.transpose(np.array([1,1,1,1,1]))
t1 = np.array([1,0])

learning_rate = 1

learn(x0, t1, b1, b2, w1, w2, learning_rate, debug = True)