import math
import random
import numpy as np

from BFGS import BFGS

class MultiStart:
    def __init__(self, f, nabla_f, n_points, optimization_algorithm, n_dim):
        self.algo = optimization_algorithm
        self.f = f
        self.nabla_f = nabla_f
        self.n_dim = n_dim

        self.n_points = n_points
        self.x_min = None

    def _generate_starting_point(self):
        x0 = np.zeros(self.n_dim)
            
        for i in range(self.n_dim):
            x0[i] = random.randint(1, 10)

        return x0

    def run(self):
        for _ in range(self.n_points):
            x0 = self._generate_starting_point()

            local_algo = self.algo(x0, self.f, self.nabla_f)
            local_algo.run()

            if self.x_min is None:
                self.x_min = local_algo.x_min

            if self.f(local_algo.x_min) < self.f(self.x_min):
                self.x_min = local_algo.x_min


if __name__ == "__main__":
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

    x = np.array([123.0, 11.0, -1.0, 0.0, 120])
    algo = BFGS(x, Rastrigin, nabla_rastrigin)
    algo.run()
    x_min = algo.x_min
    print(Rastrigin(x_min))


    multistart = MultiStart(Rastrigin, nabla_rastrigin, 10, BFGS, n_dim)
    multistart.run()

    print(Rastrigin(multistart.x_min))