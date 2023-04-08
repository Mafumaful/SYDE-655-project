import casadi as ca


class VehicleModel():
    def __init__(self, initial_state=[0, 0, 0], dt=0.1):
        # state vector
        a = ca.SX.sym('a')  # acceleration
        v = ca.SX.sym('v')  # velocity
        x = ca.SX.sym('x')  # position
        self.state = ca.vertcat(x, v, a)
        # control vector
        u = ca.SX.sym('u')  # control input

        # define the initial value of the state
        self.x = initial_state[0]
        self.v = initial_state[1]
        self.a = initial_state[2]

    def update(self, u, dt):
        pass
