import numpy as np
import pandas as pd

def compute_mse(b, w, data):
    error = 0
    for line in data:
        x = line[0]
        y = line[1]
        prediction = w * x + b
        error = error + (prediction - y) ** 2

    mse = error / len(data)

    return mse

def step_gradient(b, w, data, alpha):
    x = data[:, 0]
    y = data[:, 1]
    
    loss_function_b = 0
    loss_function_w = 0
    N = len(data)
    for line in data:
        x = line[0]
        y = line[1]

        derivative_w = 2 * x * ((w * x + b) - y)
        derivative_b = 2 * ((w * x + b) - y)

        loss_function_w = loss_function_w +  derivative_w
        loss_function_b = loss_function_b +  derivative_b

    final_b = b - alpha * (loss_function_b / N)
    final_w = w - alpha * (loss_function_w / N)

    return final_b, final_w

def predict(b, w, x):
    return w * x + b
        
def fit(data, b, w, alpha, num_iterations):
    list_b = [b]
    list_w = [w]

    for _ in range(num_iterations):
        b, w = step_gradient(b, w, data, alpha)
        list_b.append(b)
        list_w.append(w)

    return list_b, list_w

alegrete = pd.read_csv('alegrete.csv')
data = alegrete.values

list_b, list_w = fit(data, 1, 1, 0.01, 100)

b = list_b[-1]
w = list_w[-1]
print(predict(b, w, 10))

