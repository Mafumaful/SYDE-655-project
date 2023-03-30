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

        # prediction horizon
        P = 10

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

        states = ca.vertcat(sx, sv, sa)
        n_states = states.numel()

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

        # define the bounds of the system
        lbx[0:n_states*(P+1):n_states] = x_min
        ubx[0:n_states*(P+1):n_states] = x_max
        lbx[1:n_states*(P+1):n_states] = v_min
        ubx[1:n_states*(P+1):n_states] = v_max
        lbx[2:n_states*(P+1):n_states] = u_min
        ubx[2:n_states*(P+1):n_states] = u_max

        # define the bounds of the control input
        lbx[n_states*(P+1):] = u_min
        ubx[n_states*(P+1):] = u_max

    # update control input
    def update():
        pass
