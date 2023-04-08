import casadi as ca


class mpc_controller():
    def __init__(self, sampling_time=0.1):
        # MPC variables
        N = 10  # prediction horizon
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
        n_states = states.numel()

        # define the input
        u = ca.SX.sym('u')
        n_controls = u.numel()
