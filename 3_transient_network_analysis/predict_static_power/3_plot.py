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

# Plot individual states
for i in range(copes.shape[0]):
    power.save(
        copes[i],
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
        filename=f"plots/copes_{i}_.png",
    )
    power.save(
        pvalues[i],
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "cmap": cmap,
            "vmin": 0,
            "vmax": 0.1,
        },
        filename=f"plots/pvalues_{i}_.png",
    )

# Plot metastates
metastate1 = [0, 1, 4, 5, 6, 7, 9]
metastate2 = [3, 8]

p1 = np.sum(copes[metastate1], axis=0)
p2 = np.sum(copes[metastate2], axis=0)

power.save(
    p1,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
    filename="plots/combined_copes_0_.png",
)
power.save(
    -p2,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
    filename="plots/combined_copes_1_.png",
)
power.save(
    p1-p2,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
    filename="plots/combined_copes_2_.png",
)
