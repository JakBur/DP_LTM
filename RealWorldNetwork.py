import numpy as np
from DataGen import LineGeneratorGaussXNorm
from numpy.linalg import inv

from Global import SSP
from Ours import OursSimplePolyPower


def ridgeCost(data, target, weights, lam):
    cost = np.sum(((data @ weights) - target) ** 2)
    regularization = np.sum(weights ** 2)
    return cost + lam * regularization

conv = {
    0: lambda x: 0,
}
data = np.loadtxt('regData/3D_spatial_network.txt', delimiter=',', skiprows=1, converters=conv)
print(data)

A = data[:,1:-1]
b = data[:,-1]

print(A)
print(b)
n, d = A.shape
print(n,d)
#print(A)
#print('---------------------')
#print(b)

lam = 10

print("n;d;lambda;epsilon;mechanism;approx;standard deviation")

X = inv((A.T @ A) + (lam * np.identity(d)))
Y = A.T @ b
wOPT = X @ Y

OPT = ridgeCost(A,b,wOPT,lam)   

for epsilon in [0.01,0.03,0.05,0.1]:
    for mechanism in [SSP, 1, 0]:
        
        delta = 1 / n
        approxFactors = []
        for i in range(30):

            mech = ""
            wALG = 0
            if mechanism is SSP:
                wALG = mechanism(A,b,lam,1,epsilon,delta)
                mech = "global"
            else:
                #SketchGen = NNSketchGenerator()
                #S = SketchGen.generate(m,n)
                S = np.array([0])
                wALG = OursSimplePolyPower(A,b,S,lam,epsilon,delta,polyPower=mechanism,Sketching=False)
                mech = "sigmaScale n^" + str(mechanism)
            
            ALG = ridgeCost(A,b,wALG,lam)
            approx = ALG / OPT
            approxFactors.append(approx)

        approxAvg = np.average(approxFactors)
        approxStd = np.std(approxFactors)
        print(f"{n};{d};{lam};{epsilon};{mech};{approxAvg};{approxStd}")