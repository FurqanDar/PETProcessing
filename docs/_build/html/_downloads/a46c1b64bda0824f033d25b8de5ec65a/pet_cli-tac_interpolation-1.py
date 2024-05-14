import numpy as np
from pet_cli.tac_interpolation import EvenlyInterpolate, EvenlyInterpolateWithMax
import matplotlib.pyplot as plt

# define some dummy TAC times and values
tac_times = np.array([0., 1., 2.5, 4.1, 7., 15.0])
tac_values = np.array([0., 0.8, 2., 1.5, 0.6, 0.0])

# instantiate EvenlyInterpolate object and resample TAC (and add shift for better visualization)
even_interp = EvenlyInterpolate(tac_times=tac_times, tac_values=tac_values+0.25, delta_time=1.0)
resampled_tac = even_interp.get_resampled_tac()

# instantiate EvenlyInterpolateWithMax object and resample TAC (and add shift for better visualization)
even_interp_max = EvenlyInterpolateWithMax(tac_times=tac_times, tac_values=tac_values+0.5, samples_before_max=3)
resampled_tac_max = even_interp_max.get_resampled_tac()

# plot the TAC and the resampled TACs
fig, ax = plt.subplots(1,1, constrained_layout=True, figsize=(8,4))
plt.plot(tac_times, tac_values, 'ko--', label='Raw TAC', zorder=2)
plt.plot(*resampled_tac, 'ro-', label='Evenly Resampled TAC', zorder=1)
plt.plot(*resampled_tac_max, 'bo-', label='Evenly Resampled TAC w/ Max', zorder=0)
ax.text(s='resampled TACS are \nshifted for visual clarity',
        x=0.95, y=0.95, ha='right', va='top', transform=ax.transAxes, fontsize=16)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('TAC Value', fontsize=16)
plt.legend(bbox_to_anchor=(1.0, 0.5), loc='center left')
plt.show()