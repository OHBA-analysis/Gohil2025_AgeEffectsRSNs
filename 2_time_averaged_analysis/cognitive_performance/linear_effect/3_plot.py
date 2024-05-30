"""Plot results.

"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plot_pow = True
plot_mean_coh = True

colors = [(1, 0, 0, 1), (1, 1, 1, 0)]
cmap = LinearSegmentedColormap.from_list("cmap", colors, N=256)

if plot_pow:
    from osl_dynamics.analysis import power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    copes = np.load("data/glm_pow.npy").T * 1e2
    pvalues = np.load("data/glm_pow_pvalues.npy").T

    power.save(
        copes,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
        filename=f"plots/pow_.png",
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

if plot_mean_coh:
    from osl_dynamics.analysis import power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    copes = np.load("data/glm_mean_coh.npy").T * 1e2
    pvalues = np.load("data/glm_mean_coh_pvalues.npy").T

    power.save(
        copes,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
        filename=f"plots/mean_coh_.png",
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
        filename="plots/mean_coh_pvalues_.png",
    )
