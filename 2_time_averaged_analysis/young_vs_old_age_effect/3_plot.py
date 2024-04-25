"""Plot results.

"""

import mne
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plot_psd = False
plot_pow = False
plot_coh = False
plot_mean_coh = False
plot_aec = False
plot_mean_aec = False
plot_region_psds = True

def vec_to_mat(x):
    i, j = np.triu_indices(52, k=1)
    x_ = np.ones([52, 52] + list(x.shape[1:]))
    x_[i, j] = x
    x_[j, i] = x
    return x_

colors = [(1, 0, 0, 1), (1, 1, 1, 0)]
cmap = LinearSegmentedColormap.from_list("cmap", colors, N=256)

if plot_psd:
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    f = np.load("../data/f.npy")
    psd = np.load("data/psd.npy")
    category_list = np.load("data/category_list.npy")

    p1 = np.mean(psd[category_list == 2], axis=1)  # young
    p2 = np.mean(psd[category_list == 1], axis=1)  # old

    [F_obs, clusters, cluster_pv, H0] = mne.stats.permutation_cluster_test(
        [p1, p2],
        n_permutations=1000,
        n_jobs=16,
    )
    bar_y = []
    bar_x = []
    for i in range(len(clusters)):
        if cluster_pv[i] < 0.05:
            bar_x.append(f[clusters[i]])
            bar_y.append(np.ones_like(f[clusters[i]]))

    p1 = np.mean(p1, axis=0)
    p2 = np.mean(p2, axis=0)

    fig, ax = plotting.create_figure(figsize=(8,5))

    if len(bar_y) > 0:
        for x, y in zip(bar_x, bar_y):
            ax.plot(x, y * 1.1 * np.max([p1, p2]), color="red", lw=3)

    ax.plot(f, p1, label="Young", color="black", linestyle="dashed", linewidth=3)
    ax.plot(f, p2, label="Old", color="grey", linewidth=3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_xlim(f[0], f[-1])
    ax.legend()

    plotting.save(fig, filename="plots/psd.png", tight_layout=True)
    
if plot_pow:
    from osl_dynamics.analysis import power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    p0 = np.load("data/glm_pow_young.npy")
    p = np.load("data/glm_pow_old_minus_young.npy")
    pvalues = np.load("data/glm_pow_pvalues.npy")

    # Express diff as percentage of power for the young group
    p *= 100 / p0

    power.save(
        p.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
        filename="plots/pow_.png",
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
        filename="plots/pow_pvalues_.png",
    )

if plot_coh:
    from osl_dynamics.analysis import connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    c0 = np.load("data/glm_coh_young.npy")
    c = np.load("data/glm_coh_old_minus_young.npy")
    pvalues = np.load("data/glm_coh_pvalues.npy")

    # Express diff as percentage of coherence for young group
    c *= 100 / c0

    c = vec_to_mat(c).T
    pvalues = vec_to_mat(pvalues).T
    c[pvalues > 0.05] = 0
    connectivity.save(
        c,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "annotate": False,
            "display_mode": "xz",
        },
        filename="plots/glm_age_coh_.png",
    )

if plot_aec:
    from osl_dynamics.analysis import connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    c0 = np.load("data/glm_aec_young.npy")
    c = np.load("data/glm_aec_old_minus_young.npy")
    pvalues = np.load("data/glm_aec_pvalues.npy")

    # Express diff as percentage of AEC for young group
    c *= 100 / c0

    c = vec_to_mat(c).T
    pvalues = vec_to_mat(pvalues).T
    c[pvalues > 0.05] = 0
    connectivity.save(
        c,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "annotate": False,
            "display_mode": "xz",
        },
        filename="plots/glm_age_aec_.png",
    )

if plot_mean_coh:
    from osl_dynamics.analysis import power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    p0 = np.load("data/glm_mean_coh_young.npy")
    p = np.load("data/glm_mean_coh_old_minus_young.npy")
    pvalues = np.load("data/glm_mean_coh_pvalues.npy")

    # Express diff as percentage of power for the young group
    p *= 100 / p0

    power.save(
        p.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
        filename="plots/mean_coh_glm_age_.png",
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
        filename="plots/mean_coh_glm_age_pvalues_.png",
    )

if plot_mean_aec:
    from osl_dynamics.analysis import power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    p0 = np.load("data/glm_mean_aec_young.npy")
    p = np.load("data/glm_mean_aec_old_minus_young.npy")
    pvalues = np.load("data/glm_mean_aec_pvalues.npy")

    # Express diff as percentage of power for the young group
    p *= 100 / p0

    power.save(
        p.T,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
        filename="plots/mean_aec_glm_age_.png",
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
        filename="plots/mean_aec_glm_age_pvalues_.png",
    )

if plot_region_psds:
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    f = np.load("../data/f.npy")
    psd = np.load("data/psd.npy")
    category_list = np.load("data/category_list.npy")

    young = category_list == 2
    old = category_list == 1

    def plot(parcel, filename):
        p1 = psd[young, parcel]
        p2 = psd[old, parcel]

        [F_obs, clusters, cluster_pv, H0] = mne.stats.permutation_cluster_test(
            [p1, p2],
            n_permutations=1000,
            n_jobs=16,
        )
        bar_y = []
        bar_x = []
        for i in range(len(clusters)):
            if cluster_pv[i] < 0.05:
                bar_x.append(f[clusters[i]])
                bar_y.append(np.ones_like(f[clusters[i]]))

        p1 = np.mean(p1, axis=0)
        p2 = np.mean(p2, axis=0)

        fig, ax = plotting.create_figure(figsize=(8,5))

        if len(bar_y) > 0:
            for x, y in zip(bar_x, bar_y):
                ax.plot(x, y * 1.1 * np.max([p1, p2]), color="red", lw=3)

        ax.plot(f, p1, label="Young", color="black", linestyle="dashed", linewidth=3)
        ax.plot(f, p2, label="Old", color="grey", linewidth=3)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (a.u.)")
        ax.set_xlim(f[0], f[-1])
        ax.legend(loc=1)

        plotting.save(fig, filename=filename, tight_layout=True)

    plot(0, "plots/psd_occipital.png")
    plot(4, "plots/psd_motor.png")
    plot(11, "plots/psd_temporal.png")
    plot(22, "plots/psd_frontal.png")
