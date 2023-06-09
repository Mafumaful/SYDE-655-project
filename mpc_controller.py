import casadi as ca
import numpy as np
from time import time


def DM2Array(x): return np.array(x.full())


class mpc_controller():
    def __init__(self, P=100, C=6, sampling_time=0.1):
        self.time = 0
        # MPC variables
        self.P = P  # prediction horizon
        self.C = C  # control horizon
        h = sampling_time  # sampling time

        # define the states value
        # the distance between actual position and the target position
        delta_d = ca.SX.sym('delta_d')
        # the distance between host car velocity and the preceding car velocity
        delta_v = ca.SX.sym('delta_v')
        # the acceleration of the host car
        a_h = ca.SX.sym('a')
        # the states of the vehicle
        states = ca.vertcat(delta_d, delta_v, a_h)
        self.n_states = states.numel()

        # define the input
        u = ca.SX.sym('u')
        self.n_controls = u.numel()

        # define the increment A transfer matrix
        A = ca.DM([
            [0, 1, -1.6],
            [0, 0, -1],
            [0, 0, -2.17]
        ])
        # B matrix
        B = ca.DM([0, 0, 1.59130435])
        # the increment of the states
        increment = A @ states @ h + B @ u @ h
        # calculate the increment of the states
        self.f = ca.Function('f', [states, u], [increment])

        # define the parameters
        # current initial state and the target state and Q and R
        P = ca.SX.sym('P', self.n_states+self.n_states +
                      self.n_states+self.n_controls)
        X = ca.SX.sym('X', self.n_states, (self.P+1))
        U = ca.SX.sym('U', self.n_controls, self.C)

        # Q and R is the weight of the cost function, we take it from P
        # the first 3 number is the initial state
        initial_state = P[0:self.n_states]
        # the next 3 number is the target state
        reference = P[self.n_states:2*self.n_states]
        # the next 3 number is Q
        Q = ca.diag(P[2*self.n_states:2*self.n_states+self.n_states])
        R = ca.diag(P[-1])  # the last number is R

        # initial state
        st = X[:, 0]
        # initial condition constraints
        g = st - initial_state

        # the cost function and the constraint funtion
        obj = 0
        for k in range(self.P):
            # current state
            st = X[:, k]
            # current control
            ctrl = U[:, min(k, self.C-1)]

            # the cost function
            obj = obj + \
                (st-reference).T @ Q @ (st-reference) + \
                ctrl.T @ R @ ctrl

            # next state
            st_next = X[:, k+1]
            # we can use runge kutta there if the model is nonlinear
            next_st = st + self.f(st, ctrl)
            # the constraint function
            g = ca.vertcat(g, st_next - next_st)

        for k in range(self.C):
            # the control input
            ctrl = U[:, k]
            # the constraint function
            g = ca.vertcat(g, ctrl)

        # opt variables
        self.OPT_variables = ca.vertcat(
            ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp_prob = {'f': obj, 'x': self.OPT_variables, 'g': g, 'p': P}

        # solver options
        opts = {
            'ipopt':
            {
                'max_iter': 1000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        # create solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        # boundaries
        lbx = ca.DM.zeros(self.OPT_variables.shape)
        ubx = ca.DM.zeros(self.OPT_variables.shape)

        lbx[0:self.n_states*(self.P+1):self.n_states] = - \
            5  # delta_d lower bound
        lbx[1:self.n_states*(self.P+1):self.n_states] = - \
            ca.inf  # delta_v lower bound
        lbx[2:self.n_states*(self.P+1):self.n_states] = - \
            ca.inf  # a_h lower bound

        # delta_d upper bound
        ubx[0:self.n_states*(self.P+1):self.n_states] = ca.inf
        # delta_v upper bound
        ubx[1:self.n_states*(self.P+1):self.n_states] = ca.inf
        # a_h upper bound
        ubx[2:self.n_states*(self.P+1):self.n_states] = ca.inf

        lbx[self.n_states*(self.P+1):] = -2.5  # u lower bound
        ubx[self.n_states*(self.P+1):] = 1.5  # u upper bound

        # set the boundaries
        lbg = ca.DM.zeros(self.n_states*(self.P+1)+self.n_controls*self.C)
        ubg = ca.DM.zeros(self.n_states*(self.P+1)+self.n_controls*self.C)

        # set the arguments
        lbg[self.n_states*(self.P+1):] = -1.5
        ubg[self.n_states*(self.P+1):] = 1.5

        self.args = {'lbx': lbx, 'ubx': ubx, 'lbg': lbg, 'ubg': ubg}

    def return_best_u(self, initial_state, reference, Q=ca.DM([50, 1, 1]), R=ca.DM([1])):
        # record the time
        start_time = time()
        x0 = ca.repmat(initial_state, 1, self.P+1)
        # set the initial state and the target state
        p = ca.vertcat(initial_state, reference, Q, R)
        self.args['p'] = p
        self.args['x0'] = ca.vertcat(ca.reshape(
            x0, -1, 1), ca.DM.zeros(self.n_controls*self.C, 1))
        # solve the problem
        sol = self.solver(**self.args)
        # get the optimal control input
        u_opts = sol['x'][self.n_states*(self.P+1):]
        # return the optimal control input the first one
        u_opt = u_opts[0]
        # print the time
        self.time = time()-start_time
        return u_opt.full().flatten()[0]
