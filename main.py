import numpy as np
from plot import Plotter

# define the simulation deteails
t_end = 7
t_cont = 0.01
t_s = 0.2
t = np.arange(0, t_end, t_cont)
# initialize the plotter
plotter = Plotter(t)


if __name__ == "__main__":
    for i in range(len(t)+1):
        plotter.record(i, legend="time", plot_name="plot the time")
        plotter.record(i, legend="time", plot_name="plot")
        plotter.record(20, legend="20", plot_name="plot the time")
    plotter.save_graph()
