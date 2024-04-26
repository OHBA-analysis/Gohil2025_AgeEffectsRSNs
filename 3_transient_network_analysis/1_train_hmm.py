"""Train an HMM on prepared data.

"""

import pickle
from sys import argv

if len(argv) != 2:
    print("Please pass the run id, e.g. python 1_train_hmm.py 1")
    exit()

run = int(argv[1])

print("Importing packages")

from osl_dynamics.data import Data
from osl_dynamics.analysis import modes, power
from osl_dynamics.models.hmm import Config, Model

# Build model
config = Config(
    n_states=10,
    n_channels=120,
    sequence_length=4000,
    learn_means=False,
    learn_covariances=True,
    learn_trans_prob=True,
    batch_size=16,
    learning_rate=0.001,
    n_epochs=20,
)
model = Model(config)
model.summary()

# Load training data
base_dir = "/well/woolrich/users/wlo995/Gohil2024_HealthyAgeingRSNs"
training_data = Data(
    f"{base_dir}/1_preproc_and_source_recon/data/prepared",
    store_dir=f"tmp_{run}",
    use_tfrecord=True,
    n_jobs=8,
)

# Initialization
model.random_state_time_course_initialization(training_data, n_init=5, n_epochs=2)

print("Training model")
history = model.fit(training_data)

# Save the trained model
model.save(f"models/run{run}")

# Get free energy
free_energy = model.free_energy(training_data)
history["free_energy"] = free_energy

# Save training history
with open(f"models/run{run}/history.pkl", "wb") as file:
    pickle.dump(history, file)

with open(f"models/run{run}/loss.dat", "w") as file:
    file.write(f"ll_loss = {history['loss'][-1]}\n")
    file.write(f"free_energy = {free_energy}\n")

# Get inferred covariances
covs = model.get_covariances()

# Plot inferred power maps
raw_covs = modes.raw_covariances(
    covs,
    n_embeddings=training_data.n_embeddings,
    pca_components=training_data.pca_components,
)
power.save(
    power_map=raw_covs,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    subtract_mean=True,
    plot_kwargs={"cmap": "RdBu_r", "bg_on_data": 1, "darkness": 0.4, "alpha": 1},
    filename=f"models/run{run}/covs_.png",
)

# Delete temporary directory
training_data.delete_dir()
