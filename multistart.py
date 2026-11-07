import math
import random
import numpy as np

from BFGS import BFGS

def Rastrigin(x):
    sum_ = 0
    for i in range(0, len(x)):
        sum_ += (x[i]**2 - 10 * math.cos(2 * math.pi * x[i]))

    return 10 * len(x) + sum_

def nabla_rastrigin(x):
    nabla_ = np.zeros(len(x))
    for i in range(0, len(x)):
        nabla_[i] = 2*x[i] + 20 * math.pi * math.sin(2 * math.pi * x[i])

    return nabla_

n_dim = 5
def multistart(n_points):
    better = None

    for point in range(n_points):
        x = np.zeros(4)
        for i in range(4):
            x[i] = random.randint(1, 10)

        algo = BFGS(x, Rastrigin, nabla_rastrigin)
        algo.run()

        if better is None:
            better = algo.x_min

        if Rastrigin(algo.x_min) < Rastrigin(better):
            better = algo.x_min

    
    return better

x = np.array([123.0, 11.0, -1.0, 0.0, 120])
algo = BFGS(x, Rastrigin, nabla_rastrigin)
algo.run()
x_min = algo.x_min

better = multistart(10)


print(Rastrigin(x_min))
print(Rastrigin(better))