import numpy as np
import random
import casadi as ca

# import the modules
from plot import Plotter
from mpc_controller import mpc_controller
from vehicle_model import VehicleModel

# define the simulation deteails
t_end = 40
t_cont = 0.1
t_s = 0.1  # sampling time
t = np.arange(0, t_end, t_cont)

# initialize the plotter
plotter = Plotter(t)

# initialize the vehicle
diff_state = ca.DM([10, -5, 3])
ref = ca.DM([0, 0, 0])

h_state = [0, 0, 0]
host_vehicle = VehicleModel(h_state)

p_state = [10, 5, 0]
preceding_vehicle = VehicleModel(p_state)

# initialize the mpc controller
mpc = mpc_controller(sampling_time=t_s)
f = mpc.f


def record_target(state, name):
    plotter.record(state[0], "x(m)", name)
    plotter.record(state[1], "v(m/s)", name)
    plotter.record(state[2], "a(m/s^2)", name)

# create a acceleration reference of the preceding vehicle


us = []

# main loop
if __name__ == "__main__":
    for i in range(len(t)+1):
        if i == 1:
            diff_state[2] = 0.5
        elif i == 200:
            diff_state[2] = -1.5

        u0 = mpc.return_best_u(diff_state, ref)
        print('u0:', u0)
        diff_state = diff_state+f(diff_state, u0)
        host_vehicle.update(diff_state[2])
        # print(preceding_vehicle.state_value)
        record_target(diff_state.full().tolist(), name="diff")
        plotter.record(u0, "u(m/s^2)", "diff")
        record_target(host_vehicle.state_value.full().tolist(), name="host")

    plotter.save_graph()
