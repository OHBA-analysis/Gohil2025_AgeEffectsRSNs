"""Data preparation: time-delay embedding and principal component analysis.

"""

import numpy as np
from glob import glob
from osl_dynamics.data import Data

# Load source data
files = sorted(glob("../1_preproc_and_source_recon/output/*/*_sflip_lcmv-parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=16)

# Prepare
methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
    "standardize": {},
}
data.prepare(methods)

# Save
data.save_tfrecord_dataset("data/prepared", sequence_length=400)
np.save("data/pca_components.npy", data.pca_components)
