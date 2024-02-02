import numpy as np
from sklearn.linear_model import Ridge
from numpy.linalg import inv

from DataGen import LineGeneratorNoNoise
from SketchGen import NNSketchGenerator



def gauss(A, b, lam, eta, epsilon, delta):
    n, d = A.shape
    sigmaLocal = np.sqrt((4 * eta * np.log(1.25 / delta)) / (epsilon**2))
    #print(f"Local sigma = {sigmaLocal}")
    ANoise = A + np.random.normal(scale=sigmaLocal, size=(n,d))
    bNoise = b + np.random.normal(scale=sigmaLocal, size=n)

    X = inv((ANoise.T @ ANoise) + lam * np.identity(d))
    Y = ANoise.T @ bNoise
    w = X @ Y

    #clf = Ridge(alpha=lam)
    #clf.fit(ANoise, bNoise)

    #return clf.coef_
    return w