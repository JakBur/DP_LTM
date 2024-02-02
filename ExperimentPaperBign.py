import numpy as np
from sklearn.linear_model import Ridge
from numpy.linalg import inv

from DataGen import LineGeneratorGauss, LineGeneratorGaussXNorm, LineGeneratorNoNoise, LineGeneratorUniform
from Global import SSP
from Local import gauss
from Ours import Ours, OursSimple, OursSimplePolyPower
from SketchGen import NNSketchGenerator

# No SciPy was used here

def ridgeCost(data, target, weights, lam):
    cost = np.sum(((data @ weights) - target) ** 2)
    regularization = np.sum(weights ** 2)
    return cost + lam * regularization

print("alpha;d;lambda;epsilon;mechanism;n;approx;standard deviation")

m = 500
eta = 1
# for alpha in [0,1,2]:
for alpha in [1]:
    # for d in [3,10,50]:
    for d in [10]:
        # for lam in [1,10,100]:
        for lam in [10]:
            for epsilon in [0.1]:
                for mechanism in [0.5]:
                    for n in [int(1000000 * (i+1)) for i in range(40)]:
                        delta = 1 / n
                        approxFactors = []
                        DataGen = LineGeneratorGaussXNorm()
                        A, b = DataGen.generate(n,d,xNoise=(n**alpha))
                        for i in range(30):

                            X = inv((A.T @ A) + (lam * np.identity(d)))
                            Y = A.T @ b
                            wOPT = X @ Y

                            OPT = ridgeCost(A,b,wOPT,lam)

                            mech = ""
                            wALG = 0
                            if mechanism is SSP:
                                wALG = mechanism(A,b,lam,eta,epsilon,delta)
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
                        print(f"{alpha};{d};{lam};{epsilon};{mech};{n};{approxAvg};{approxStd}")