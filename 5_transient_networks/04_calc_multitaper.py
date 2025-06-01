"""Calculate subject/static-specific multitaper spectra.

"""

print("Importing packages")
import os
import pickle
import numpy as np
from glob import glob
from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data

os.makedirs("results/post_hoc", exist_ok=True)

# Load data
files = sorted(glob("../1_preproc_and_source_recon/output/*/*_sflip_lcmv-parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=16)
data_ = data.trim_time_series(n_embeddings=15, sequence_length=400)

# Load inferred state probabilities
alpha = pickle.load(open("results/inf_params/alp.pkl", "rb"))

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
np.save("results/post_hoc/f.npy", f)
np.save("results/post_hoc/psd.npy", psd)
np.save("results/post_hoc/coh.npy", coh)
np.save("results/post_hoc/w.npy", w)
