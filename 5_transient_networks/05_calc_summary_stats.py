"""Calculate summary statistics.

"""

print("Import packages")
import pickle
import numpy as np
from osl_dynamics.analysis import modes
from osl_dynamics.inference.modes import argmax_time_courses

# Load state probabilities and calculate state time courses
alp = pickle.load(open("results/inf_params/alp.pkl", "rb"))
stc = argmax_time_courses(alp)

# Calculate summary stats
fo = modes.fractional_occupancies(stc)
lt = modes.mean_lifetimes(stc, sampling_frequency=250)
intv = modes.mean_intervals(stc, sampling_frequency=250)
sr = modes.switching_rates(stc, sampling_frequency=250)

np.save("post_hoc/fo.npy", fo)
np.save("post_hoc/lt.npy", lt)
np.save("post_hoc/intv.npy", intv)
np.save("post_hoc/sr.npy", sr)

# Calculate transition probability matrices
tp = modes.calc_trans_prob_matrix(stc, n_states=10)

np.save("post_hoc/tp.npy", tp)
