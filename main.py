import numpy as np
import random

# import the modules
from plot import Plotter
from mpc_controller import mpc_controller
from vehicle import VehicleModel

# define the simulation deteails
t_end = 7
t_cont = 0.01
t_s = 0.1  # sampling time
t = np.arange(0, t_end, t_cont)

# initialize the plotter
plotter = Plotter(t)


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
        pass

    plotter.save_graph()
