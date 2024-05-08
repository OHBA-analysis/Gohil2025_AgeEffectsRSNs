"""Gather data needed to fit a GLM.

"""

import os
import numpy as np

os.makedirs("data", exist_ok=True)

# Directories
base_dir = "/well/woolrich/users/wlo995/Gohil2024_HealthyAgeingRSNs"
model_dir = f"{base_dir}/3_transient_network_analysis/models/run2"
static_dir = f"{base_dir}/2_time_averaged_analysis"

# Load static power
pow_ = np.load(f"{static_dir}/data/pow.npy")

# Load summary stats
fo = np.load(f"{model_dir}/fo.npy")
lt = np.load(f"{model_dir}/lt.npy")
intv = np.load(f"{model_dir}/intv.npy")
sr = np.load(f"{model_dir}/sr.npy")

# Reorder the states
state_psds = np.load(f"{model_dir}/psd.npy")
p = np.mean(state_psds, axis=(0,2,3))
order = np.argsort(p)[::-1]

fo = fo[:, order]
lt = lt[:, order]
intv = intv[:, order]
sr = sr[:, order]

# Save
np.save("data/pow.npy", pow_)
np.save("data/fo.npy", fo)
np.save("data/lt.npy", lt)
np.save("data/intv.npy", intv)
np.save("data/sr.npy", sr)
