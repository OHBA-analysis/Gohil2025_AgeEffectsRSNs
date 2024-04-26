"""Calculate static-specific multitaper spectra for the best run.

"""

import pickle
import numpy as np
from glob import glob

from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data

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

base_dir = "/well/woolrich/users/wlo995/Gohil2024_HealthyAgeingRSNs"
model_dir = f"models/run{best_run}"

# Load data
files = sorted(glob(f"{base_dir}/1_preproc_and_source_recon/data/src/*/sflip_parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=16)
data_ = data.trim_time_series(n_embeddings=15, sequence_length=4000)

# Load inferred state probabilities
alpha = pickle.load(open(f"{model_dir}/alp.pkl", "rb"))

# Calculate multitaper
f, psd, coh, w = spectral.multitaper_spectra(
    data=data_,
    alpha=alpha,
    sampling_frequency=250,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=[1, 45],
    standardize=True,
    return_weights=True,
    n_jobs=16,
)
np.save(f"{model_dir}/f.npy", f)
np.save(f"{model_dir}/psd.npy", psd)
np.save(f"{model_dir}/coh.npy", coh)
np.save(f"{model_dir}/w.npy", w)

# Delete temporary directory
data.delete_dir()
