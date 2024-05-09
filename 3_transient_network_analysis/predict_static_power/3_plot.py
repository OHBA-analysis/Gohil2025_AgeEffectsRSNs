"""Plot results.

"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from osl_dynamics.analysis import power
from osl_dynamics.utils import plotting

plotting.set_style({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

colors = [(1, 0, 0, 1), (1, 1, 1, 0)]
cmap = LinearSegmentedColormap.from_list("cmap", colors, N=256)

copes = np.load("data/glm_copes.npy")
pvalues = np.load("data/glm_pvalues.npy")

power.save(
    copes,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
    filename="plots/pow_.png",
)
power.save(
    pvalues,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={
        "views": ["lateral"],
        "cmap": cmap,
        "vmin": 0,
        "vmax": 0.1,
    },
    filename="plots/pow_pvalues_.png",
)
