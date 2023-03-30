import casadi as ca
import numpy as np

# define the 1d class model
# the input of the model is the acceleration


class Vehicle1D():
    def __init__(self, state):
        self.sx = ca.MX.sym('x')
        self.sv = ca.MX.sym('v')
        self.sa = ca.MX.sym('a')
        self.st = ca.MX.sym('t')
        self.su = ca.MX.sym('u')

        self.x = state[0]
        self.v = state[1]
        self.a = state[2]

        self.rhs = ca.vertcat(
            self.sv*self.st+self.sx,
            self.su*self.st+self.sv,
            self.su
        )

        self.f = ca.Function(
            'f', [self.sx, self.sv, self.sa, self.st, self.su], [self.rhs])

    def update(self, u, step_time):
        rhs = np.array(self.f(self.x, self.v, self.a, step_time, u))
        self.x = rhs[0]
        self.v = rhs[1]
        self.a = rhs[2]

    def get_state(self):
        return [self.x, self.v, self.a]
