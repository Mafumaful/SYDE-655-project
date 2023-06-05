import numpy as np
import random
import casadi as ca
from time import time

# import the modules
from plot import Plotter
from mpc_controller import mpc_controller
from vehicle_model import VehicleModel

# define the simulation deteails
t_end = 20
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
# mpc.__init__(P=100, C=50)
Q = ca.DM([50, 1, 1])
R = ca.DM([0.1])

us = []
times = []

# main loop
if __name__ == "__main__":
    for i in range(len(t)+1):
        start_time = time()
        # update the optimal control input
        print('-----------------------'*5)
        print('Q:', Q)
        print('R:', R)
        u0 = mpc.return_best_u(diff_state, ref, Q, R)
        times.append(mpc.time)
        # print control input
        print('u0:', u0)
        # added a random disturbance
        # disturbance = ca.DM([0, 0, random.uniform(-0.1, 0.1)])
        disturbance = ca.DM([0, 0, 0])
        # disturbance = ca.DM([0, 0, 0.1])
        diff_state = diff_state+f(diff_state, u0) + disturbance
        host_vehicle.update(u0)

        # update Q, R and prediction horizon according to the state
        mpc_state = diff_state.full().tolist()
        [[dd], [dv], [ah]] = mpc_state
        print('mpc_state:', mpc_state)
        # update the Q and R according to the desired performance
        if dd > 2 or dd < -2:
            mpc.__init__(P=50, C=5)
            Q = ca.DM([50, 1, 1])
            R = ca.DM([0.1])
        else:
            mpc.__init__(P=100, C=50)
            Q = ca.DM([1, 1, 1])
            R = ca.DM([1])

        # record the data to plot
        name = "proposed mpc"
        record_target(mpc_state, name=name)
        plotter.record(u0, "u", name)
    plotter.save_graph()
    print('the average time of each iteration:', np.mean(times))
