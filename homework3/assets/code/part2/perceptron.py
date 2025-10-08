import numpy as np

#update function
def update(weight, learning_rate, target, output, x):
    multiplication_factor = learning_rate * (target - output)
    return weight + (multiplication_factor * x)

#activation function
def sgn(x):
    if x<0: return 0
    return 1

def calculate_output(x, weight):
    mul = weight @ x # matrix multiplication
    return sgn(mul)

def learn(inputs, targets, weight, learning_rate, num_iterations, debug):

    size = len(inputs)

    for iteration in range(1,num_iterations+1):

        if debug: print("ITERATION", iteration)

        for index in range(size):

            x = inputs[index]
            target = targets[index]
            if debug: print("x\n",x.reshape(-1, 1),"\ntarget:",target)

            output = calculate_output(x, weight)
            if debug: print("output", output)

            if output!=target:
                if debug: print("UPDATE")
                weight = update(weight, learning_rate, target, output, x)
                if debug: print("new weight\n", weight.reshape(-1, 1))

    print("Weight after iterations:\n", weight.reshape(-1, 1))