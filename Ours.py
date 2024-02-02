import numpy as np
from sklearn.linear_model import Ridge
from numpy.linalg import inv

from DataGen import LineGeneratorNoNoise


def Ours(A, b, S, lam, eta, epsilon, delta):
    n, d = A.shape
    m, _ = S.shape
    
    sigmaOur = np.sqrt((16 * (eta**2) * np.log((1.25 * d) / delta) * m * (d**2)) / (n * (epsilon**2)))
    #sigmaOur = 50/np.sqrt(n)
    #print(f"sigmaScale = {sigmaOur * np.sqrt(n)}")
    #print(f"our sigma = {sigmaOur}")

    ANoise = A + np.random.normal(scale=sigmaOur, size=(n,d))
    bNoise = b + np.random.normal(scale=sigmaOur, size=n)

    ANoiseSketch = S @ ANoise
    bNoiseSketch = S @ bNoise

    X = inv((ANoiseSketch.T @ ANoiseSketch) + lam * np.identity(d))
    Y = ANoiseSketch.T @ bNoiseSketch
    w = X @ Y

    #clf = Ridge(alpha=lam)
    #clf.fit(ANoiseSketch, bNoiseSketch)
    
    #return clf.coef_
    return w


def OursSimple(A,b,S,lam,epsilon,delta,sigmaScale='none',Sketching=True):
    n, d = A.shape

    localSigma = np.sqrt((np.log(1 / delta)) / (epsilon**2))
    if sigmaScale is 'n':
        localSigma = np.sqrt((np.log(1 / delta)) / (n * (epsilon**2)))
    elif sigmaScale is 'sqrtn':
        localSigma = np.sqrt((np.log(1 / delta)) / (np.sqrt(n) * (epsilon**2)))
    elif sigmaScale is '4sqrtn':
        localSigma = np.sqrt((np.log(1 / delta)) / (np.sqrt(np.sqrt(n)) * (epsilon**2)))
    elif sigmaScale is 'logn':
        localSigma = np.sqrt((np.log(1 / delta)) / (np.log(n) * (epsilon**2)))
    
    ANoise = A + np.random.normal(scale=localSigma, size=(n,d))
    bNoise = b + np.random.normal(scale=localSigma, size=n)

    if Sketching:
        ANoise = S @ ANoise
        bNoise = S @ bNoise

    X = inv((ANoise.T @ ANoise) + (lam * np.identity(d)))
    Y = ANoise.T @ bNoise
    w = X @ Y

    return w


def OursSimplePolyPower(A,b,S,lam,epsilon,delta,polyPower=1,Sketching=True):
    n, d = A.shape

    scale = n ** polyPower
    localSigma = np.sqrt((np.log(1 / delta)) / (scale * (epsilon**2)))
    
    ANoise = A + np.random.normal(scale=localSigma, size=(n,d))
    bNoise = b + np.random.normal(scale=localSigma, size=n)

    if Sketching:
        ANoise = S @ ANoise
        bNoise = S @ bNoise

    X = inv((ANoise.T @ ANoise) + (lam * np.identity(d)))
    Y = ANoise.T @ bNoise
    w = X @ Y

    return w