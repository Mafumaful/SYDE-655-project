import numpy as np
import random

# import the modules
from plot import Plotter
from controller import MPC_ACC_controller
from model1d import Vehicle1D

# define the simulation deteails
t_end = 7
t_cont = 0.01
t_s = 0.1  # sampling time
t = np.arange(0, t_end, t_cont)

# initialize the plotter
plotter = Plotter(t)

# initialize the controller
# controller = MPC_ACC_controller(t_s)

# initialize the model, the initial position of the vehicle is 10m, velocity is constant which is 10m/s
target_vehicle = Vehicle1D([0, 0, 0])
# initialize the host vehicle, the initial position of the vehicle is 0m
host_vehicle = Vehicle1D([0, 0, 0])

# define the recorder to record the state


def record_target(state, name):
    plotter.record(target_vehicle.get_state()
                   [0], "x(m)", name)
    plotter.record(target_vehicle.get_state()
                   [1], "v(m/s)", name)
    plotter.record(target_vehicle.get_state()
                   [2], "a(m/s^2)", name)


# main loop
if __name__ == "__main__":
    for i in range(len(t)+1):
        # create a random acceleration
        target_vehicle.update(1, t_cont)

        # plot
        record_target(target_vehicle.get_state(), "target vehicle state")

    plotter.save_graph()
