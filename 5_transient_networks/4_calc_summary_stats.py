"""Calculate summary statistics for the best run.

We also calculate subject specific transition probability matrices.
"""

import pickle
import numpy as np

from osl_dynamics.analysis import modes
from osl_dynamics.inference.modes import argmax_time_courses

# Get best model
best_fe = np.Inf
for run in range(1, 6):
    try:
        with open(f"models/run{run}/loss.dat") as file:
            lines = file.readlines()
        fe = float(lines[1].split("=")[-1].strip())
        print(f"run {run}: {fe}")
        if fe < best_fe:
            best_run = run
            best_fe = fe
    except:
        print(f"run {run} missing")
        pass

print(f"Best run: {best_run}")

# Load state probabilities and calculate Viterbi path
model_dir = f"models/run{best_run}"
alp = pickle.load(open(f"{model_dir}/alp.pkl", "rb"))
stc = argmax_time_courses(alp)

# Calculate summary stats
fo = modes.fractional_occupancies(stc)
lt = modes.mean_lifetimes(stc, sampling_frequency=250)
intv = modes.mean_intervals(stc, sampling_frequency=250)
sr = modes.switching_rates(stc, sampling_frequency=250)

np.save(f"{model_dir}/fo.npy", fo)
np.save(f"{model_dir}/lt.npy", lt)
np.save(f"{model_dir}/intv.npy", intv)
np.save(f"{model_dir}/sr.npy", sr)

# Calculate transition probability matrices
tp = modes.calc_trans_prob_matrix(stc, n_states=10)

np.save(f"{model_dir}/tp.npy", tp)
