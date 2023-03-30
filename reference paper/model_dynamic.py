import numpy as np
import utils
from functools import partial
from scipy.integrate import solve_ivp


class Vehicle():
    def __init__(self):
        self.lr = 1.738
        self.lf = 1.105
        self.C_alphaf = 262180
        self.C_alphar = 219034
        self.dim_n = 5
        self.dim_m = 1
        self.M = 1200
        self.I_z = 1000
        self.V_x = 16
        self.measure_dim = 5
        self.x = np.zeros((self.dim_n))
        self.x_dot = np.zeros((self.dim_n))
        self.measure = np.zeros((self.measure_dim))

        self.Domain = [0, 100, 0, 100, -10, 10, -10, 10, -np.pi, np.pi]
        self.u_lim = [-np.pi/4, np.pi/4]

    def integrate(self, u, t_interval):
        dx_dt = partial(self.dynamics, u=u)
        sol = solve_ivp(dx_dt, t_interval, self.x, method='RK45', t_eval=None,
                        rtol=1e-6, atol=1e-6, dense_output=False, events=None, vectorized=False)
        self.x = sol.y[..., -1]
        # return partial(self.dynamics, u=u)

    def dynamics(self, t, x, u):
        X, Y, phi, V_y, r = x
        delta = u

        V_y_dot = -((self.C_alphaf*np.cos(delta)+self.C_alphar)/(self.M*self.V_x)) * V_y + ((-self.lf*self.C_alphaf *
                                                                                             np.cos(delta)+self.lr*self.C_alphar)/(self.I_z*self.V_x)) * r + ((self.C_alphaf*np.cos(delta))/self.M) * delta
        r_dot = ((-self.lf*self.C_alphaf*np.cos(delta)+self.lr*self.C_alphar)/(self.M*self.V_x)-self.V_x) * V_y - (((self.lf**2)*self.C_alphaf *
                                                                                                                    np.cos(delta)+(self.lr**2)*self.C_alphar)/(self.I_z*self.V_x)) * r + ((self.lf*self.C_alphaf*np.cos(delta))/self.I_z) * delta
        X_dot = self.V_x*np.cos(phi)-V_y*np.sin(phi)
        Y_dot = self.V_x*np.sin(phi)+V_y*np.cos(phi)
        phi_dot = r

        self.x_dot = [X_dot, Y_dot, phi_dot, V_y_dot, r_dot]
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
                break

        return in_dom
