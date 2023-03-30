import matplotlib.pyplot as plt

# this is a class used to record the data according to the time


class Plotter(object):
    def __init__(self, time):
        self.dictionarys = {}
        self.time_stamp = time
        self.output_path = "output/"

    def record(self, record_value, legend, plot_name):
        # find the element in the dictionary,
        # if not exist, create a new one
        if plot_name not in self.dictionarys:
            self.dictionarys[plot_name] = {}
        if legend not in self.dictionarys[plot_name]:
            self.dictionarys[plot_name][legend] = []
        else:
            self.dictionarys[plot_name][legend].append(record_value)

    def save_graph(self):
        print("saving the graph...")
        for key, value in self.dictionarys.items():
            plt.figure()
            for legend, data in value.items():
                plt.plot(self.time_stamp, data, label=legend)
            plt.xlabel("time")
            # grid on
            plt.grid()
            plt.legend()
            plt.savefig(self.output_path+key+".png")
