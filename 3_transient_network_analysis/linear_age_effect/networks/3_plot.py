"""Plot results.

"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import optimize

plot_pow_map = False
plot_coh_net = False
plot_coh_map = False
plot_trans_prob = False
plot_sum_stats = True

def vec_to_mat(x):
    i, j = np.triu_indices(52, k=1)
    x_ = np.ones([x.shape[0], 52, 52])
    x_[:, i, j] = x
    x_[:, j, i] = x
    return x_

def line(x, m, c):
    return m*x + c

os.makedirs("plots", exist_ok=True)

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
        p,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
            #"vmax": p.max(),
        },
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

    pvalues = vec_to_mat(pvalues)

    c = vec_to_mat(beta)
    c[pvalues > 0.05] = 0

    connectivity.save(
        c,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "annotate": False,
            "display_mode": "xz",
        },
        filename="plots/coh_thres_.png",
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
        p,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
        filename="plots/mean_coh_.png",
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

if plot_trans_prob:
    p = np.load("data/glm_trans_prob_age.npy")
    pvalues = np.load("data/glm_trans_prob_age_pvalues.npy")

    fig, ax = plt.subplots()

    im = ax.matshow(p)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    cax.tick_params(labelsize=15)
    cax.set_ylabel("Difference", fontsize=16)

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if pvalues[i, j] < 0.05:
                ax.text(j, i, "*", ha="center", va="center", color="red", fontsize=16)

    ax.set_xticklabels([""] + ["1", "3", "5", "7", "9"])
    ax.set_yticklabels([""] + ["1", "3", "5", "7", "9"])
    ax.tick_params(labelsize=15)
    ax.set_xlabel("State: To", fontsize=16)
    ax.set_ylabel("State: From", fontsize=16)
    ax.xaxis.set_label_position("top")

    plt.savefig("plots/tp.png")
    plt.close()

if plot_sum_stats:
    copes = np.load("data/glm_sum_stats_age.npy")
    pvalues = np.load("data/glm_sum_stats_age_pvalues.npy")

    age = np.load("data/age.npy")
    sum_stats = np.load("data/sum_stats.npy")
    sr = np.sum(sum_stats[:, -1], axis=-1)

    copes /= age.std()
    copes[1] *= 1e3

    tab10_cmap = plt.get_cmap("tab10")
    color = [tab10_cmap(i) for i in range(10)]
    titles = [
        "Fractional Occupancy",
        "Mean Lifetime (ms)",
        "Mean Interval (s)",
        "Switching Rate (Hz)",
    ]

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(13,3))

    # FO, LT, INTV
    for i in range(len(copes)):
        rects = ax[i].bar(range(1, 11), copes[i], color=color)
        ax[i].set_title(titles[i], fontsize=16)
        ax[i].set_xlabel("State", fontsize=16)
        ax[i].tick_params(labelsize=15)
        labels = []
        for p in pvalues[i]:
            if p < 0.05:
                labels.append("*")
            else:
                labels.append("")
        ax[i].bar_label(rects, padding=3, labels=labels)
        bottom, top = ax[i].get_ylim()
        ax[i].set_ylim(1.2 * bottom, 1.4 * top)
    ax[0].set_ylabel("Change Per Year", fontsize=16)

    # SR
    popt, pcov = optimize.curve_fit(line, age, sr)
    m, c = popt
    y = line(age, m, c)

    ax[-1].scatter(age, sr, s=10, c="black", label="Data")
    ax[-1].plot(age, y, c="red", lw=2, label=f"y = {m:.2g}x + {c:.3g}")
    ax[-1].set_title(titles[-1], fontsize=16)
    ax[-1].set_xlabel("Age (years)", fontsize=16)
    ax[-1].set_ylim(5, None)
    ax[-1].tick_params(labelsize=15)
    ax[-1].legend(fontsize=10)

    plt.tight_layout()
    filename = f"plots/sum_stats.png"
    print("Saving", filename)
    plt.savefig(filename)
