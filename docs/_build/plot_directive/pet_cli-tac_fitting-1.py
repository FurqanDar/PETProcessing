import numpy as np
import pet_cli.tcms_as_convolutions as pet_tcm
import pet_cli.tac_fitting as pet_fit
import matplotlib.pyplot as plt
import pet_cli.testing_utils as pet_tst

tcm_func = pet_tcm.generate_tac_1tcm_c1_from_tac
pTAC = np.asarray(np.loadtxt("../data/tcm_tacs/fdg_plasma_clamp_evenly_resampled.txt").T)
tTAC = tcm_func(*pTAC, k1=1.0, k2=0.25, vb=0.05)
tTAC[1] = pet_tst.add_gaussian_noise_to_tac_based_on_max(tTAC[1])

fitter = pet_fit.TACFitter(pTAC=pTAC, tTAC=tTAC, tcm_func=tcm_func)
fitter.run_fit()
fit_params = fitter.fit_results[0]
fit_tac = pet_tcm.generate_tac_1tcm_c1_from_tac(*pTAC, *fit_params)

plotter = pet_tst.TACPlots()
plotter.add_tac(*pTAC, label='Input TAC', pl_kwargs={'color':'black', 'ls':'--'})
plotter.add_tac(*tTAC, label='Tissue TAC', pl_kwargs={'color':'blue', 'ls':'', 'marker':'o', 'mec':'k'})
plotter.add_tac(*fit_tac, label='Fit TAC', pl_kwargs={'color':'red', 'ls':'-', 'marker':'', 'lw':2.5})
plt.legend()
plt.show()