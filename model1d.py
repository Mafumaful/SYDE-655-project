import casadi as ca
import numpy as np

# define the 1d class model
# the input of the model is the acceleration


class Vehicle1D():
    def __init__(self, state=[0, 0, 0]):
        # define the initial states
        self.x = state[0]
        self.v = state[1]
        self.a = state[2]

        # define the states
        self.st = ca.SX.sym('t')
        self.sx = ca.SX.sym('x')
        self.sv = ca.SX.sym('v')
        self.sa = ca.SX.sym('a')
        self.su = ca.SX.sym('u')
        # define the states of the system
        self.sstate = ca.vertcat(self.sx, self.sv, self.sa)

        # define the state space matrix
        # A matrix
        self.A = ca.DM([
            [0, 1, -1.6],
            [0, 0, -1],
            [0, 0, -2.17391304]
        ])
        # B matrix
        self.B = ca.DM([0, 0, -1.59130435])
        # C matrix
        self.C = ca.DM.eye(3)
        # define the increment of the state
        self.increment = self.A @ self.sstate @ self.st + self.B @ self.su @ self.st

        # runge kutta 4th order
        k1 = self.increment
        k2 = self.increment + k1 * self.st / 2
        k3 = self.increment + k2 * self.st / 2
        k4 = self.increment + k3 * self.st
        self.rhs = self.sstate + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        self.f = ca.Function('f', [self.sstate, self.su, self.st], [self.rhs],
                             ['states', 'controls', 'step_time'], ['RHS'])

    def update(self, u, step_time):
        state = ca.DM([self.x, self.v, self.a])
        updated_state = np.array(self.f(state, u, step_time))
        self.x = updated_state[0]
        self.v = updated_state[1]
        self.a = updated_state[2]

    def get_state(self):
        return [self.x, self.v, self.a]
