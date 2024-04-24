"""Plot GLM analysis.

"""

import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

os.makedirs("plots", exist_ok=True)

plot_pow_map = True
plot_coh_net = True
plot_coh_map = True
plot_aec_net = True
plot_aec_map = True

def vec_to_mat(x):
    i, j = np.triu_indices(52, k=1)
    x_ = np.ones([52, 52] + list(x.shape[1:]))
    x_[i, j] = x
    x_[j, i] = x
    return x_

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

    beta = np.load("data/glm_pow_age.npy")
    pvalues = np.load("data/glm_pow_age_pvalues.npy")
    beta_0 = np.load("data/glm_pow_mean.npy")

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
        filename="plots/glm_pow_age_.png",
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
        filename="plots/glm_pow_age_pvalues_.png",
    )

if plot_coh_net:
    from osl_dynamics.analysis import connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    beta = np.load("data/glm_coh_age.npy")
    beta_0 = np.load("data/glm_coh_mean.npy")
    pvalues = np.load("data/glm_coh_age_pvalues.npy")

    X = (age - age.mean()) / age.std()
    y = beta * X.min() + beta_0
    beta = 100 * beta / (age.std() * beta_0)

    pvalues = vec_to_mat(pvalues).T

    c = vec_to_mat(beta).T
    c[pvalues > 0.05] = 0

    connectivity.save(
        c,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "annotate": False,
            "display_mode": "xz",
        },
        filename="plots/glm_coh_age_thres_.png",
    )

if plot_aec_net:
    from osl_dynamics.analysis import connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    beta = np.load("data/glm_aec_age.npy")
    beta_0 = np.load("data/glm_aec_mean.npy")
    pvalues = np.load("data/glm_aec_age_pvalues.npy")

    X = (age - age.mean()) / age.std()
    y = beta * X.min() + beta_0
    beta = 100 * beta / (age.std() * beta_0)

    pvalues = vec_to_mat(pvalues).T

    c = vec_to_mat(beta).T
    c[pvalues > 0.05] = 0

    connectivity.save(
        c,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "annotate": False,
            "display_mode": "xz",
        },
        filename="plots/glm_aec_age_thres_.png",
    )

if plot_coh_map:
    from osl_dynamics.analysis import power

    beta = np.load("data/glm_mean_coh_age.npy")
    beta_0 = np.load("data/glm_mean_coh_mean.npy")
    pvalues = np.load("data/glm_mean_coh_age_pvalues.npy")

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
        filename="plots/glm_mean_coh_age_.png",
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
        filename="plots/glm_mean_coh_age_pvalues_.png",
    )

if plot_aec_map:
    from osl_dynamics.analysis import power

    beta = np.load("data/glm_mean_aec_age.npy")
    beta_0 = np.load("data/glm_mean_aec_mean.npy")
    pvalues = np.load("data/glm_mean_aec_age_pvalues.npy")

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
        filename="plots/glm_mean_aec_age_.png",
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
        filename="plots/glm_mean_aec_age_pvalues_.png",
    )
