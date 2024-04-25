"""Data preparation: time-delay embedding and principal component analysis.

"""

from glob import glob

from osl_dynamics.data import Data

# Load data
files = sorted(glob("data/src/*/sflip_parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=16)

# Prepare
methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
    "standardize": {},
}
data.prepare(methods)

# Save
data.save("data/prepared")
data.delete_dir()
