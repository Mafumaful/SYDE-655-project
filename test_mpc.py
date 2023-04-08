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
us = []

# define the mpc controller
mpc = mpc_controller(sampling_time=0.1)
f = mpc.f


# main
while ca.norm_2(state_init - state_ref) > 0.01:
    u0 = mpc.return_best_u(state_init, state_ref)
    state_init = state_init+f(state_init, u0)
    state_inits.append(state_init.full().flatten().tolist())
    us.append(u0)

# average time
print('average calculate time: ', np.mean(mpc.times))

x = np.arange(0, 0.1*len(state_inits), 0.1)

# plot the result
plt.figure()
plt.plot(x, state_inits)
plt.legend(['delta_d', 'delta_v', 'a_h'])
plt.step(x, us, label='u')
plt.xlabel('time(s)')
plt.ylabel('value')
plt.title('MPC result')
plt.grid()
plt.savefig('output/mpc_result.png')
