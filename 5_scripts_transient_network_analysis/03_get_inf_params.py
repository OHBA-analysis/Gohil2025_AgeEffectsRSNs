"""Get inferred parameters.

"""

print("Importing packages")
import os
import pickle
import numpy as np
from osl_dynamics.data import load_tfrecord_dataset
from osl_dynamics.models import load

os.makedirs("data/inf_params", exist_ok=True)

# Load training data
dataset = load_tfrecord_dataset(
    "data/prepared",
    config.batch_size,
    buffer_size=1000,
    shuffle=False,
    concatenate=False,
)

# Load model
model = load("results/model")
model.summary()

# Get inferred alphas
alpha = model.get_alpha(dataset)
pickle.dump(alpha, open("results/inf_params/alp.pkl", "wb"))

# Get parameters
initial_state_probs = model.get_initial_state_probs()
trans_prob = model.get_trans_probs()
means, covs = model.get_means_covariances()
np.save("results/inf_params/initial_state_probs.npy", initial_state_probs)
np.save("results/inf_params/trans_prob.npy", trans_prob)
np.save("results/inf_params/means.npy", means)
np.save("results/inf_params/covs.npy", covs)
