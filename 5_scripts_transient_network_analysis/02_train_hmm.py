"""Train an HMM on prepared data.

"""

print("Importing packages")
from osl_dynamics.data import load_tfrecord_dataset
from osl_dynamics.models.hmm import Config, Model

# Build model
#
# Note: the original paper used sequence_length=4000, batch_size=16.
# However, we recommend the following hyperparameters instead when
# using the latest version of osl-dynamics.
config = Config(
    n_states=n_states,
    n_channels=120,
    sequence_length=400,
    learn_means=False,
    learn_covariances=True,
    batch_size=64,
    learning_rate=0.001,
    n_init=5,
    n_init_epochs=2,
    init_take=0.4,
    n_epochs=15,
    best_of=5, # can reduce to 1 to avoid training multiple models
)
model = Model(config)
model.summary()

# Load dataset and train model
dataset = load_tfrecord_dataset(
    "data/prepared",
    config.batch_size,
    buffer_size=1000,
)
model.train(dataset)
model.save("results/model")
