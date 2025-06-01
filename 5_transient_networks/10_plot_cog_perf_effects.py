"""Plot results.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

plot_pow_map = True
plot_coh_net = True
plot_coh_map = True
plot_trans_prob = True
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

if plot_pow_map:
    from osl_dynamics.analysis import power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    beta = np.load("data/glm/pow_copes.npy")[1]
    pvalues = np.load("data/glm/pow_pvalues.npy")[1]

    power.save(
        beta,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        filename="plots/glm_pow_cog_.png",
    )
    power.save(
        pvalues,
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

    beta = np.load("data/glm/coh_copes.npy")[1]
    pvalues = np.load("data/glm/coh_pvalues.npy")[1]

    pvalues = vec_to_mat(pvalues)

    c = vec_to_mat(beta)
    c[pvalues > 0.05] = 0

    connectivity.save(
        c,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"annotate": False, "display_mode": "xz"},
        filename="plots/glm_coh_thres_cog_.png",
    )

if plot_coh_map:
    from osl_dynamics.analysis import power

    beta = np.load("data/glm/mean_coh_copes.npy")[1]
    pvalues = np.load("data/glm/mean_coh_pvalues.npy")[1]

    power.save(
        beta,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        filename="plots/glm_mean_coh_cog_.png",
    )
    power.save(
        pvalues,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"cmap": cmap, "vmin": 0, "vmax": 0.1},
        filename="plots/glm_mean_coh_cog_pvalues_.png",
    )

if plot_trans_prob:
    p = np.load("data/glm/tp_copes.npy")[1]
    pvalues = np.load("data/glm/tp_pvalues.npy")[1]

    fig, ax = plt.subplots()

    im = ax.matshow(p)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    cax.tick_params(labelsize=15)
    cax.set_ylabel("COPE", fontsize=16)

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

    plt.savefig("plots/glm_tp_cog.png")
    plt.close()

if plot_sum_stats:
    copes = np.load("data/glm/sum_stats_copes.npy")[1]
    pvalues = np.load("data/glm/sum_stats_pvalues.npy")[1]

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
    ax[0].set_ylabel("COPE", fontsize=16)

    plt.tight_layout()
    filename = "plots/glm_sum_stats_cog.png"
    print("Saving", filename)
    plt.savefig(filename)
