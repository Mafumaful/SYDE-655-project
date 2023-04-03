import casadi as ca
import numpy as np


class MPC_ACC_controller():
    def __init__(self, step_time, initial_state=[0, 0, 0]):
        # define the step time
        self.step_time = step_time

        # define the initial state
        self.x = initial_state[0]
        self.v = initial_state[1]
        self.a = initial_state[2]

        # define the safe distance rate
        self.k = 3

        # prediction horizon
        P_horizon = 10

        # setting the weights of the system
        Q_x = 1
        Q_v = 1
        Q_a = 1

        R_u = 1

        # setting the limit
        u_max = 5
        u_min = -5

        v_max = 100
        v_min = 0

        x_max = ca.inf
        x_min = -ca.inf

        # define the states
        sx = ca.SX.sym('x')
        sv = ca.SX.sym('v')
        sa = ca.SX.sym('a')
        su = ca.SX.sym('u')
        # define the states of the system
        states = ca.vertcat(sx, sv, sa)
        n_states = states.numel()

        # define the state space matrix, it's the most important part of the model
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

        # matrix containing all states over all time steps
        X = ca.SX.sym('X', n_states, P_horizon + 1)
        # matrix containing all control actions over all time steps
        U = ca.SX.sym('U', 1, P_horizon)
        # coloumn vector for storing initial state and target state
        P = ca.SX.sym('P', n_states + n_states)
        # state weights matrix (Qx,Qv,Qa)
        Q = ca.diagcat(Q_x, Q_v, Q_a)
        # control weights matrix (Ru)
        R = ca.diagcat(R_u)

        # define the right hand side of the system
        # RHS is the increment of the state
        increment = \
            self.A @ states @ self.step_time + self.B @ su @ self.step_time
        # define the function
        f = ca.Function('f', [states, su], [increment],
                        ['states', 'controls'], ['RHS'])

        # define the cost
        cost = 0
        g = X[:, 0] - P[:n_states]

        # define the objective function
        for i in range(P_horizon):
            current_state = X[:, i]
            current_control = U[:, i]
            cost = cost\
                + (current_state - P[:n_states]).T @ Q @ (current_state - P[:n_states])\
                + current_control.T @ R @ current_control
            next_state = X[:, i + 1]

            # runge kutta 4th order
            k1 = f(current_state, current_control)
            k2 = f(current_state + step_time / 2 * k1, current_control)
            k3 = f(current_state + step_time / 2 * k2, current_control)
            k4 = f(current_state + step_time * k3, current_control)
            x_next_rk4 = current_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            g = ca.vertcat(g, next_state - x_next_rk4)

        OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # define the nonlinear problem
        nlp_prob = {
            'f': cost,
            'x': OPT_variables,
            'g': g,
            'p': P
        }

        # define the nonlinear problem option
        opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        # create the solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        # define the bounds of the system
        lbx = ca.DM.zeros((n_states*(P_horizon+1) + P_horizon, 1))
        ubx = ca.DM.zeros((n_states*(P_horizon+1) + P_horizon, 1))

        lbx[0:n_states*(P_horizon+1):n_states] = x_min
        ubx[0:n_states*(P_horizon+1):n_states] = x_max
        lbx[1:n_states*(P_horizon+1):n_states] = v_min
        ubx[1:n_states*(P_horizon+1):n_states] = v_max
        lbx[2:n_states*(P_horizon+1):n_states] = u_min
        ubx[2:n_states*(P_horizon+1):n_states] = u_max

        # define the bounds of the control input
        lbx[n_states*(P_horizon+1):] = u_min
        ubx[n_states*(P_horizon+1):] = u_max

        # define the initial value of the system
        self.t0 = ca.DM(0)
        self.state_init = ca.DM(initial_state)
        # relative distance, velocity and acceleration
        self.state_target = ca.DM([k*initial_state[1], 0, 0])

        self.u0 = ca.DM.zeros((P_horizon, 1))
        self.x0 = ca.repmat(ca.DM(initial_state), P_horizon+1, 1)  # check this

    # update control input
    def update(self, current_state):
        u = 0
        return u

    # now we need to implement the transfer function of the system and update the control loop for this controller
