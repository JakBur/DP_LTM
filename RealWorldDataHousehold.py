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
    1: lambda x: 0,
}
#data = np.loadtxt('regData/household_power_consumption.txt', delimiter=';', skiprows=1, converters=conv)

file = open('regData/household_power_consumption.txt', "r")
file.readline()
n = 0
d = 6
while True:
    content=file.readline()
    if '?' in content:
        continue
    if not content:
        break
    n += 1

print(n)

A = np.zeros((n,d))
b = np.zeros(n)

file.close()
print("Done1")

file = open('regData/household_power_consumption.txt', "r")
file.readline()

i = 0
while True:
    content=file.readline()
    if '?' in content:
        continue
    if not content:
        break
    content = content.strip()
    l = content.split(';')
    b[i] = float(l[8])
    lpart = l[2:-1]
    lpartfl = [float(x) for x in lpart]
    A[i] = lpartfl
    i += 1
    
file.close()
print("Done2")

print(A)
print(b)

lam = 10

print("n;d;lambda;epsilon;mechanism;approx;standard deviation")

for epsilon in [0.01,0.03,0.05,0.1]:
    for mechanism in [SSP, 1, 0]:
        
        delta = 1 / n
        approxFactors = []
        for i in range(30):

            X = inv((A.T @ A) + (lam * np.identity(d)))
            Y = A.T @ b
            wOPT = X @ Y

            OPT = ridgeCost(A,b,wOPT,lam)

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