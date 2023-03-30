import numpy as np
from plot import Plotter
from controller import controller
from model1d import Vehicle1D

# define the simulation deteails
t_end = 7
t_cont = 0.01
t_s = 0.2
t = np.arange(0, t_end, t_cont)

# initialize the plotter
plotter = Plotter(t)

# initialize the controller

# initialize the model
vehicle = Vehicle1D([0, 0, 0])

if __name__ == "__main__":
    cnt = 0
    for i in range(len(t)+1):
        cnt += 1
        vehicle.update(1.0, t_cont)
        print(vehicle.get_state())
        plotter.record(vehicle.get_state()[0], "x", "state")
        plotter.record(vehicle.get_state()[1], "v", "state")
        plotter.record(vehicle.get_state()[2], "a", "state")

    plotter.save_graph()
    print(cnt)
