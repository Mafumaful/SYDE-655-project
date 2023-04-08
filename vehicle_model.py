import casadi as ca


class VehicleModel():
    def __init__(self, initial_state=[0, 0, 0]):
        # state vector
        a = ca.SX.sym('a')  # acceleration
        v = ca.SX.sym('v')  # velocity
        x = ca.SX.sym('x')  # position
        dt = ca.SX.sym('dt')  # sampling time
        a_previous = ca.SX.sym('a_previous')  # previous acceleration
        self.state = ca.vertcat(x, v, a)
        # control vector
        u = ca.SX.sym('u')  # control input

        # define the increment A transfer matrix
        A = ca.DM([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, -2.1]
        ])
        # B matrix
        B = ca.DM([0, 0, 1.59130435])

        # the increment of the states
        increment = A @ self.state @ dt + B @ u @ dt
        # calculate the increment of the states
        self.f = ca.Function('f', [self.state, u, dt], [increment])

        # define the initial value of the state
        self.x_value = initial_state[0]
        self.v_value = initial_state[1]
        self.a_value = initial_state[2]
        self.state_value = [self.x_value, self.v_value, self.a_value]

    def update(self, u, dt=0.1):
        self.state_value = self.state_value + \
            self.f(self.state_value, u, dt)
        self.a_value = self.state_value[2]
        self.v_value = self.state_value[1]
        self.x_value = self.state_value[0]
        return self.state_value
