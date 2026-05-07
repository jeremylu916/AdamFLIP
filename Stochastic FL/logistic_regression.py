import numpy as np
from SOFL import *

np.random.seed(43)
C = 5
N = [200 for c in range(C)]
nx = 10  # Assuming nx is the dimension of the data points
Dx = []
Dy = []
locs = [np.random.uniform(-1,1) for c in range(C)]

for c in range(C):
    Dcy = []
    Dcx = []
    for i in range(N[c]):
        xi = np.random.normal(loc=locs[c],size=nx)  # sample from normal distribution
        yi = np.random.choice([0, 1])
        Dcx.append(xi)
        Dcy.append(yi)
    Dx.append(Dcx)
    Dy.append(Dcy)

def logistic_loss(Dx, Dy, weights):
    losses = []
    for c in range(C):
        loss = 0
        num_samples = len(Dx[c])
        for i in range(num_samples):
            xi = Dx[c][i]
            yi = Dy[c][i]
            # Logistic loss
            prediction = 1 / (1 + np.exp(-np.dot(weights, xi)))
            if prediction <=0 or prediction >=1:
                print("Alert! Wrong prediction")
            loss += -yi * np.log(prediction) - (1 - yi) * np.log(1 - prediction)
        losses.append(loss / N[c])
    return losses

# losses = logistic_loss(Dx, Dy, weights)
# print(losses)

def gradient_loss(Dx, Dy, weights):
    gradients = []
    for c in range(C):
        grad = np.zeros(nx)
        num_samples = len(Dx[c])
        for i in range(num_samples):
            xi = Dx[c][i]
            yi = Dy[c][i]
            prediction = 1 / (1 + np.exp(-np.dot(weights, xi)))
            error = prediction - yi
            grad += error * xi
        gradients.append(grad / N[c])
    return gradients

def Hessian_loss(Dx, Dy,weights):
    hessian = np.zeros((nx, nx))
    for c in range(C):
        num_samples = len(Dx[c])
        for i in range(num_samples):
            xi = Dx[c][i]
            prediction = 1 / (1 + np.exp(-np.dot(weights, xi)))
            hessian += prediction * (1 - prediction) * np.outer(xi, xi)
    return hessian / sum(N)

def f(x):
    losses = logistic_loss(Dx, Dy, x)
    return np.mean(losses)

def df(x):
    gradients = gradient_loss(Dx, Dy, x)
    return np.mean(gradients, axis=0)
def d2f(x):
    return Hessian_loss(Dx, Dy, x)

def h(x):
    losses = logistic_loss(Dx, Dy, x)
    return np.array(losses) - f(x) #- 0.001#0.07

def dh(x):
    gradients = gradient_loss(Dx, Dy, x)
    return np.array(gradients) - df(x)

def f_sub(x, Dx, Dy):
    losses = logistic_loss(Dx, Dy, x)
    return np.mean(losses)

def df_sub(x, Dx, Dy):
    gradients = gradient_loss(Dx, Dy, x)
    return np.mean(gradients, axis=0)
def d2f_sub(x, Dx, Dy):
    return Hessian_loss(Dx, Dy, x)

def h_sub(x, Dx, Dy):
    losses = logistic_loss(Dx, Dy, x)
    return np.array(losses) - f(x)# - 0.01#0.07

def dh_sub(x, Dx, Dy):
    gradients = gradient_loss(Dx, Dy, x)
    return np.array(gradients) - df_sub(x, Dx, Dy)    

if __name__ == "__main__":
    x_start = np.random.randn(nx,)
    Kp = np.eye(C) #1
    Ki = 0
    eta = 0.1 #0.1
    max_iter = 1000
    tol = 1e-6
    batch_size = 20
    x_opt, history = SQP_eq_stochastic(Dx, Dy, f,h,df,dh,f_sub, h_sub, df_sub, dh_sub, x_start, Kp, Ki, eta, max_iter, tol, batch_size)
    np.savez('logistic_regression.npz', history=np.array(history, dtype=object), x_opt=x_opt)
    plot_optimization_history_2(history, legends=['SOFL'], linewidth=3, savename='logistic_regression.png')