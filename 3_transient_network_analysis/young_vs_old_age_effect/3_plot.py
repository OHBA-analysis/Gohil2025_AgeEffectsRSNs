"""Plot results.

"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

plot_psd = True
plot_pow = True
plot_trans_prob = True
plot_sum_stats = True

os.makedirs("plots", exist_ok=True)

model_dir = "../models/run2"

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

    f = np.load(f"{model_dir}/f.npy")
    psd = np.load("data/psd.npy")
    category_list = np.load("data/category_list.npy")

    p1 = np.mean(psd[category_list == 2], axis=2)  # young
    p2 = np.mean(psd[category_list == 1], axis=2)  # old

    n_states = p1.shape[1]

    for k in range(n_states):
        [F_obs, clusters, cluster_pv, H0] = mne.stats.permutation_cluster_test(
            [p1[:, k], p2[:, k]],
            n_permutations=1000,
            n_jobs=16,
        )
        bar_y = []
        bar_x = []
        for i in range(len(clusters)):
            if cluster_pv[i] < 0.05:
                bar_x.append(f[clusters[i]])
                bar_y.append(np.ones_like(f[clusters[i]]))

        p1_ = np.mean(p1[:, k], axis=0)
        p2_ = np.mean(p2[:, k], axis=0)

        fig, ax = plotting.create_figure(figsize=(8,5))

        if len(bar_y) > 0:
            for x, y in zip(bar_x, bar_y):
                ax.plot(x, y * 1.1 * np.max([p1_, p2_]), color="red", lw=3)

        ax.plot(f, p1_, label="Young", color="black", linestyle="dashed", linewidth=3)
        ax.plot(f, p2_, label="Old", color="grey", linewidth=3)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (a.u.)")
        ax.set_xlim(f[0], f[-1])
        ax.legend(loc=1)

        plotting.save(fig, filename=f"plots/psd_{k:02d}.png", tight_layout=True)

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
        p,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "views": ["lateral"],
            "symmetric_cbar": True,
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

if plot_trans_prob:
    p = np.load("data/glm_trans_prob_old_minus_young.npy")
    pvalues = np.load("data/glm_trans_prob_pvalues.npy")

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
    copes = np.load("data/glm_sum_stats_old_minus_young.npy")
    pvalues = np.load("data/glm_sum_stats_pvalues.npy")

    tab10_cmap = plt.get_cmap("tab10")
    color = [tab10_cmap(i) for i in range(10)]
    ylabels = [
        "Fractional Occupancy",
        "Mean Lifetime (ms)",
        "Mean Interval (s)",
        "Switching Rate (Hz)",
    ]

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,3))
    for i in range(len(copes)):
        rects = ax[i].bar(range(1, 11), copes[i], color=color)
        ax[i].set_ylabel(ylabels[i], fontsize=16)
        ax[i].set_xlabel("State", fontsize=16)
        ax[i].tick_params(labelsize=15)

        labels = []
        for p in pvalues[i]:
            if p < 0.05:
                labels.append("*")
            else:
                labels.append("")

        ax[i].bar_label(rects, padding=3, labels=labels)

    plt.tight_layout()

    filename = f"plots/sum_stats.png"
    print("Saving", filename)
    plt.savefig(filename)
