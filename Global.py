import numpy as np
from sklearn.linear_model import Ridge
from numpy.linalg import inv

from DataGen import LineGeneratorNoNoise
from Local import gauss
from Ours import Ours
from SketchGen import NNSketchGenerator

def SSP(A, b, lam, eta, epsilon, delta):
    n, d = A.shape
    
    ATA = (A.T @ A)

    #noiseScale1 = (2 * np.sqrt(2 * np.log(4 / delta)) * d * (eta**2)) / epsilon
    noiseScale1 = (2 * np.sqrt(2 * np.log(4 / delta))) / epsilon

    g = np.random.normal(scale=1, size=(d,d))
    gtriu = np.triu(g)
    gauss1 = gtriu.T + np.triu(gtriu, k=1)

    ATAnoise = ATA + (noiseScale1 * gauss1)
    
    first = inv(ATAnoise + lam * np.identity(d))

    ATb = A.T @ b

    #noiseScale2 = (2 * np.sqrt(2 * d * np.log(4 / delta)) * (eta**2)) / epsilon
    noiseScale2 = (2 * np.sqrt(2 * np.log(4 / delta))) / epsilon
    gauss2 = np.random.normal(scale=1, size=d)

    ATbnoise = ATb + noiseScale2 * gauss2

    second = ATbnoise

    return first @ second
