from abc import ABC, abstractmethod
import numpy as np


class SketchGenerator(ABC):

    @abstractmethod
    def generate(self, m, n):
        pass


class TestSketchGenerator(SketchGenerator):

    def __init__(self):
        pass

    def generate(self, m=2, n=3):
        sketch = np.array([[1,0,-1],[0,1,0]])
        return sketch


class NNSketchGenerator(SketchGenerator):

    def __init__(self):
        pass

    def generate(self, m, n):
        sketch = np.zeros((m,n))
        for i in range(n):
            row = np.random.randint(m)
            entry = (np.random.randint(2)*2) - 1
            sketch[row,i] = entry
        return sketch