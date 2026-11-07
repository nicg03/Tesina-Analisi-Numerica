import numpy as np


class BFGS:
    def __init__(self, starting_point, f, nabla_f, rho=0.5, c=1e-4):
        self.rho = rho
        self.c = c
        self.x0 = starting_point

        self.f = f
        self.nabla_f = nabla_f

        self.x_min = 0 

    def line_search(self, alpha, x, p, _):
        if self.f(x + alpha * p) <= self.f(x) + self.c * alpha * self.nabla_f(x).T @ p:
            return alpha
        elif _ >= 100:
            return alpha
        else:
            return self.line_search(alpha * self.rho, x, p, _ + 1)

    def update_B(self, B, s, y):
        if y.T @ s <= 1e-8 or np.linalg.norm(s) <= 1e-8:
            return B
        
        alpha = 1/ (y.T @ s)
        beta = -1 / (s.T @ B @ s)


        return B + alpha * (y.reshape(-1, 1) @ y.reshape(-1, 1).T) + beta * (B @ s.reshape(-1, 1) @ s.reshape(-1, 1).T @ B.T)

    def _bfgs(self, B_k, x_k, alpha, iteration):
        p = -np.linalg.inv(B_k) @ self.nabla_f(x_k)
        alpha = self.line_search(alpha, x_k, p, 0)
        x_new = x_k + p * alpha
        y_new = self.nabla_f(x_new) - self.nabla_f(x_k)
        s = x_new - x_k

        B_new = self.update_B(B_k, s, y_new)

        if iteration == 100:
            return x_new

        return self._bfgs(B_new, x_new, 1.0, iteration + 1)

    def run(self):
        B = np.identity(len(self.x0))

        self.x_min = self._bfgs(B, self.x0, 1.0, 0)


if __name__ == "__main__":
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

    