import numpy as np
import utils
from functools import partial
from scipy.integrate import solve_ivp


class Vehicle():
    def __init__(self):
        self.lr = 1.738
        self.lf = 1.105
        self.dim_n = 4
        self.dim_m = 2
        self.measure_dim = 4
        self.x = np.zeros((self.dim_n))
        self.x_dot = np.zeros((self.dim_n))
        self.measure = np.zeros((self.measure_dim))

        self.Domain = [0, 100, 0, 100, -2*np.pi, 2*np.pi, -50, 50]
        self.u_lim = [-np.pi/4, np.pi/4, -4, 11]

    def integrate(self, u, t_interval):
        dx_dt = partial(self.dynamics, u=u)
        sol = solve_ivp(dx_dt, t_interval, self.x, method='RK45', t_eval=None,
                        rtol=1e-6, atol=1e-6, dense_output=False, events=None, vectorized=False)
        self.x = sol.y[..., -1]
        # self.x[2]=utils.rad_regu(self.x[2])
        self.x[2] = np.sign(self.x[2])*(abs(self.x[2]) % (2*np.pi))
        # return partial(self.dynamics, u=u)

    def dynamics(self, t, x, u):
        X, Y, phi, v = x
        # phi=utils.rad_regu(phi)
        delta, a = u
        # a+=4
        beta = np.arctan((self.lr/(self.lf+self.lr))*np.tan(delta))
        X_dot = v*np.cos(phi+beta)
        Y_dot = v*np.sin(phi+beta)
        phi_dot = (v/self.lr)*np.sin(beta)
        v_dot = a+4

        self.x_dot = [X_dot, Y_dot, phi_dot, v_dot]
        return self.x_dot

    def Read_sensor(self):
        self.measure = self.x
        return self.measure

    def Read_sensor_with_noise(self, sig):
        self.measure = (np.random.normal(0, sig, self.measure_dim) +
                        np.ones((self.measure_dim)))*self.measure
        return self.measure

    def In_domain(self, x):
        in_dom = True
        for i in range(self.dim_n):
            if (x[i] < self.Domain[2*i]) | (x[i] > self.Domain[2*i+1]):
                in_dom = False
                print("x[{}]={} is out of range[{},{}]".format(
                    i, x[i], self.Domain[2*i], self.Domain[2*i+1]))
                break

        return in_dom
