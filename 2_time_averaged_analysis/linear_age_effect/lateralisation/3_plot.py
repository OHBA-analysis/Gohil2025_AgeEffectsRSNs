"""Plot GLM analysis.

"""

import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

os.makedirs("plots", exist_ok=True)

plot_pow_map = False
plot_coh_map = True
plot_aec_map = True

colors = [(1, 0, 0, 1), (1, 1, 1, 0)]
cmap = LinearSegmentedColormap.from_list("cmap", colors, N=256)

age = np.load("data/age.npy")

if plot_pow_map:
    from osl_dynamics.analysis import power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    beta = np.load("data/glm_pow_diff_age.npy")
    pvalues = np.load("data/glm_pow_diff_age_pvalues.npy")
    beta_0 = np.load("data/glm_pow_diff_mean.npy")

    # We fit y = beta*X + beta_0
    #
    # beta is the is the change in y per standardised X
    #
    # Get the value of y for the smallest value of standardised X
    # - This corresponds to the target value for someone aged 18
    X = (age - age.mean()) / age.std()
    y = beta * X.min() + beta_0

    # Use the mean as the reference
    p = 100 * beta / (age.std() * beta_0)

    power.save(
        p.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
            #"vmax": p.max(),
        },
        filename="plots/glm_pow_diff_age_.png",
    )
    power.save(
        pvalues.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "cmap": cmap,
            "vmin": 0,
            "vmax": 0.1,
        },
        filename="plots/glm_pow_diff_age_pvalues_.png",
    )

if plot_coh_map:
    from osl_dynamics.analysis import power

    beta = np.load("data/glm_mean_coh_diff_age.npy").T
    beta_0 = np.load("data/glm_mean_coh_diff_mean.npy").T
    pvalues = np.load("data/glm_mean_coh_diff_age_pvalues.npy").T

    X = (age - age.mean()) / age.std()
    y = beta * X.min() + beta_0
    p = 100 * beta / (age.std() * beta_0)

    power.save(
        p.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
        filename="plots/glm_mean_coh_diff_age_.png",
    )
    power.save(
        pvalues.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "cmap": cmap,
            "vmin": 0,
            "vmax": 0.1,
        },
        filename="plots/glm_mean_coh_diff_age_pvalues_.png",
    )

if plot_aec_map:
    from osl_dynamics.analysis import power

    beta = np.load("data/glm_mean_aec_diff_age.npy").T
    beta_0 = np.load("data/glm_mean_aec_diff_mean.npy").T
    pvalues = np.load("data/glm_mean_aec_diff_age_pvalues.npy").T

    X = (age - age.mean()) / age.std()
    y = beta * X.min() + beta_0
    p = 100 * beta / (age.std() * beta_0)

    power.save(
        p.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
        filename="plots/glm_mean_aec_diff_age_.png",
    )
    power.save(
        pvalues.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "cmap": cmap,
            "vmin": 0,
            "vmax": 0.1,
        },
        filename="plots/glm_mean_aec_diff_age_pvalues_.png",
    )
