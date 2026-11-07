import math
import numpy as np

from BFGS import BFGS


def f(x):
    return (1-x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

def nabla_f(x):
    f_value = x[0]
    s_value = x[1]
    return np.array([-2 * (1 - f_value) - 400 * f_value * (s_value - f_value ** 2), 200 * (s_value - f_value ** 2)])

algo = BFGS(np.array([2.0, 2.0]), f, nabla_f)
algo.run()

print(f(np.array([2.0, 2.0])))
print(algo.x_min)
print(f(algo.x_min))