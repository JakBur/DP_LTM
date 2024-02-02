from abc import ABC, abstractmethod
import numpy as np


class DataGenerator(ABC):

    @abstractmethod
    def generate(self, n, d):
        pass


class TestDataGenerator(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n=3, d=2):
        # data = np.empty((n,d))
        A = np.array([[0,0],[1,1],[2,2]])
        b = np.array([0,2,4])
        return A, b
    

class VerticalLineGenerator(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d=1):
        A = np.zeros((n,d))
        b = np.zeros(n)
        for i in range(n):
            A[i] = i
            b[i] = 2
        return A, b
    

class LineGenerator(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d, noise=0.1):
        x = np.random.rand(d)
        A = np.zeros((n,d))
        b = np.zeros(n)
        for i in range(n):
            Ai = np.random.rand(d)
            bi = (Ai @ x) + np.random.normal(scale=noise)
            A[i] = Ai
            b[i] = bi
        return A, b
    

class LineGeneratorGauss(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d, noise=0.1):
        A = np.random.normal(scale=1,size=(n,d))
        x = np.random.normal(scale=1,size=d)
        b = (A @ x) + np.random.normal(scale=noise, size=n)
        return A,b

class LineGeneratorGaussXNorm(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d, xNoise=1):
        A = np.random.normal(scale=1,size=(n,d))
        x = np.random.normal(scale=xNoise,size=d)
        b = (A @ x)
        return A,b


class LineGeneratorUniform(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d, noise=0.1):
        A = np.random.uniform(size=(n,d))
        x = np.random.uniform(size=d)
        b = (A @ x) + np.random.normal(scale=noise, size=n)
        return A,b
    

# Wrong - squares of singular values not adding up to n
class LineGeneratorGaussSingularValues(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d, noise=0.1):
        G = np.random.normal(size=(n,d))
        U, _, V = np.linalg.svd(G, full_matrices=False)
        # S = np.array([[1000,0,0],[0,20,0],[0,0,0.5]])
        singularValues = []
        #for i in range(blocks):
        #    nextBlock = [... for j in range()]
        if d == 3:
            singularValues = [n/2, n/100, n/1000]
        elif d == 10:
            singularValues = [n/2, n/2, n/100, n/100, n/1000, n/1000, n/10000, n/10000, n/100000, n/100000]
        else: # d == 100
            singularValues = []
            for i in range(10):
                [singularValues.append(n/2) if i == 0 else singularValues.append(n/(10**(i+1))) for _ in range(10)]
            #singularValues = [n/2 if i == 0 else n/(10**i) for i in range(d)]
        S = np.diag(singularValues)
        A = (U @ S) @ V.T
        x = np.random.normal(size=d)
        b = (A @ x) + np.random.normal(scale=noise, size=n)
        return A,b
    

# Wrong - squares of singular values not adding up to n
class LineGeneratorGaussSingularValues10dim(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d=10, noise=0.1):
        G = np.random.normal(size=(n,d))
        U, _, V = np.linalg.svd(G, full_matrices=False)
        singularValues = [n/2, n/2, n/2, n/2, n/2, n/1000, n/1000, n/1000, n/1000, n/1000]
        S = np.diag(singularValues)
        A = (U @ S) @ V.T
        x = np.random.normal(size=d)
        b = (A @ x) + np.random.normal(scale=noise, size=n)
        return A,b


# Wrong - squares of singular values not adding up to n
class LineGeneratorGaussSingularValuesGeneral(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d, blocksize, drop, noise=0.1):
        G = np.random.normal(size=(n,d))
        U, _, V = np.linalg.svd(G, full_matrices=False)
        singularValues = []
        blocks = int(d / blocksize + 1)
        for i in range(blocks-1):
            value = n / (drop**i)
            for _ in range(blocksize):
                singularValues.append(value)
        for _ in range(d-len(singularValues)):
            singularValues.append(drop**(blocks-1)) # <-- BUG !! (should have been n / drop**(blocks-1))
        S = np.diag(singularValues)
        A = (U @ S) @ V.T
        x = np.random.normal(size=d)
        b = (A @ x) + np.random.normal(scale=noise, size=n)
        return A,b
    

# Squares of singular values add up to n
class LineGeneratorGaussSV(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d, blocksize, drop, noise=0.1):
        G = np.random.normal(size=(n,d))
        U, _, V = np.linalg.svd(G, full_matrices=False)
        singularValues = []
        blocks = int(d / blocksize + 1)
        initValue = np.sqrt(n / (blocksize * sum([1 / (drop ** (2 * i)) for i in range(blocks)])))
        for i in range(blocks-1):
            value = initValue / (drop**i)
            for _ in range(blocksize):
                singularValues.append(value)
        for _ in range(d-len(singularValues)):
            singularValues.append(initValue / (drop**(blocks-1)))
        S = np.diag(singularValues)
        A = (U @ S) @ V.T
        x = np.random.normal(size=d)
        b = (A @ x) + np.random.normal(scale=noise, size=n)
        return A,b


class LineGeneratorUniformSingularValues(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d, noise=0.1):
        G = np.random.uniform(size=(n,d))
        U, _, V = np.linalg.svd(G, full_matrices=False)
        singularValues = [n/2 if i == 0 else n/(10**i) for i in range(d)]
        S = np.diag(singularValues)
        A = (U @ S) @ V.T
        x = np.random.uniform(size=d)
        b = (A @ x) + np.random.normal(scale=noise, size=n)
        return A,b


class LineGeneratorNoNoise(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d):
        A = np.random.normal(size=(n,d))
        x = np.random.normal(size=d)
        b = (A @ x)
        return A,b
    

class LineGeneratorGaussSVsmallb(DataGenerator):

    def __init__(self):
        pass

    def generate(self, n, d, blocksize, drop, noise=0.1):
        G = np.random.normal(size=(n,d))
        U, _, V = np.linalg.svd(G, full_matrices=False)
        singularValues = []
        blocks = int(d / blocksize + 1)
        initValue = np.sqrt(n / (blocksize * sum([1 / (drop ** (2 * i)) for i in range(blocks)])))
        for i in range(blocks-1):
            value = initValue / (drop**i)
            for _ in range(blocksize):
                singularValues.append(value)
        lastBlock = blocksize
        currentLength = len(singularValues)
        for _ in range(d-currentLength):
            lastBlock = d-currentLength
            singularValues.append(initValue / (drop**(blocks-1)))
        S = np.diag(singularValues)
        A = (U @ S) @ V.T
        x = np.random.normal(size=d)
        sigmai = singularValues[-1]
        b = sum([sigmai * U[:,-i-1] * x[-i-1] for i in range(lastBlock)]) # + np.random.normal(scale=noise, size=n)
        return A,b