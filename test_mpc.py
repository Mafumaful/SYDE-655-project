import matplotlib.pyplot as plt
import casadi as ca
import numpy as np
from mpc_controller import mpc_controller


state_init = ca.DM([10, -5, 3])
state_ref = ca.DM([0, 0, 0])

# store the data to be plotted
delta_ds = []
delta_vs = []
a_hs = []
state_inits = []

# define the mpc controller
mpc = mpc_controller(sampling_time=0.1)
f = mpc.f


# main
while ca.norm_2(state_init - state_ref) > 0.01:
    u0 = mpc.return_best_u(state_init, state_ref)
    state_init = state_init+f(state_init, u0)
    state_inits.append(state_init.full())

# average time
print('average time: ', np.mean(mpc.times))

for state_init in state_inits:
    [delta_d] = state_init[0]
    [delta_v] = state_init[1]
    [a_h] = state_init[2]
    delta_ds.append(delta_d)
    delta_vs.append(delta_v)
    a_hs.append(a_h)

# plot the result
plt.plot(delta_ds, label='delta_d')
plt.plot(delta_vs, label='delta_v')
plt.plot(a_hs, label='a_h')
plt.legend()
plt.grid()
plt.savefig('output/result test mpc.png')
