"""Get inferred parameters for the best model.

"""

print("Importing packages")

import pickle
import numpy as np

from osl_dynamics.data import Data
from osl_dynamics.models import load

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

# Load model
model = load(f"models/run{best_run}")
model.summary()

# Load training data
base_dir = "/well/woolrich/users/wlo995/Gohil2024_AgeCognitionEffectsRSNs"
training_data = Data(
    f"{base_dir}/1_preproc_and_source_recon/data/prepared",
    use_tfrecord=True,
    n_jobs=16,
)

# Get inferred alphas
alpha = model.get_alpha(training_data)
pickle.dump(alpha, open(f"models/run{best_run}/alp.pkl", "wb"))

# Get inferred covariances
covs = model.get_covariances()
np.save(f"models/run{best_run}/covs.npy", covs)

# Delete temporary directory
training_data.delete_dir()
