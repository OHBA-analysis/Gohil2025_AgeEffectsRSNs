"""Gather data needed to fit a GLM.

"""

import os
import numpy as np

from osl_dynamics.analysis import power

os.makedirs("data", exist_ok=True)

# Directories
base_dir = "/well/woolrich/users/wlo995/Gohil2024_HealthyAgeingRSNs"
model_dir = f"{base_dir}/3_transient_network_analysis/models/run2"
static_dir = f"{base_dir}/2_time_averaged_analysis"

# Load static power
f = np.load(f"{static_dir}/data/f.npy")
static_psd = np.load(f"{static_dir}/data/psd.npy")
static_pow = power.variance_from_spectra(f, static_psd)

# Load summary stats and state PSDs
fo = np.load(f"{model_dir}/fo.npy")

# Reorder the states
state_psds = np.load(f"{model_dir}/psd.npy")
p = np.mean(state_psds, axis=(0,2,3))
order = np.argsort(p)[::-1]
fo = fo[:, order]

# Save
np.save("data/static_pow.npy", static_pow)
np.save("data/fo.npy", fo)
