"""Plot GLM analysis.

"""

import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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

os.makedirs("plots", exist_ok=True)

colors = [(1, 0, 0, 1), (1, 1, 1, 0)]
cmap = LinearSegmentedColormap.from_list("cmap", colors, N=256)

if plot_pow_map:
    from osl_dynamics.analysis import power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    beta = np.load("data/glm_pow_copes.npy")[1]
    pvalues = np.load("data/glm_pow_pvalues.npy")[1]

    power.save(
        beta.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        filename="plots/glm_pow_cog_.png",
    )
    power.save(
        pvalues.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"cmap": cmap, "vmin": 0, "vmax": 0.1},
        filename="plots/glm_pow_cog_pvalues_.png",
    )

if plot_coh_net:
    from osl_dynamics.analysis import connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    beta = np.load("data/glm_coh_copes.npy")[1]
    pvalues = np.load("data/glm_coh_pvalues.npy")[1]

    pvalues = vec_to_mat(pvalues).T

    c = vec_to_mat(beta).T
    c[pvalues > 0.05] = 0

    connectivity.save(
        c,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"annotate": False, "display_mode": "xz"},
        filename="plots/glm_coh_cog_thres_.png",
    )

if plot_aec_net:
    from osl_dynamics.analysis import connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    beta = np.load("data/glm_aec_copes.npy")[1]
    pvalues = np.load("data/glm_aec_pvalues.npy")[1]

    pvalues = vec_to_mat(pvalues).T

    c = vec_to_mat(beta).T
    c[pvalues > 0.05] = 0

    connectivity.save(
        c,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"annotate": False, "display_mode": "xz"},
        filename="plots/glm_aec_cog_thres_.png",
    )

if plot_coh_map:
    from osl_dynamics.analysis import power

    beta = np.load("data/glm_mean_coh_copes.npy")[1]
    pvalues = np.load("data/glm_mean_coh_pvalues.npy")[1]

    power.save(
        beta.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        filename="plots/glm_mean_coh_cog_.png",
    )
    power.save(
        pvalues.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"cmap": cmap, "vmin": 0, "vmax": 0.1},
        filename="plots/glm_mean_coh_cog_pvalues_.png",
    )

if plot_aec_map:
    from osl_dynamics.analysis import power

    beta = np.load("data/glm_mean_aec_copes.npy")[1]
    pvalues = np.load("data/glm_mean_aec_pvalues.npy")[1]

    power.save(
        beta.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        filename="plots/glm_mean_aec_cog_.png",
    )
    power.save(
        pvalues.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"cmap": cmap, "vmin": 0, "vmax": 0.1},
        filename="plots/glm_mean_aec_cog_pvalues_.png",
    )
