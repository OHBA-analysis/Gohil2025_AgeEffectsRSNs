"""Plot results.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import optimize

plot_pow = False
plot_mean_coh = False
plot_trans_prob = False
plot_sum_stats = True

def line(x, m, c):
    return m*x + c

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

    copes = np.load("data/glm_pow.npy") * 1e2
    pvalues = np.load("data/glm_pow_pvalues.npy")

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

    copes = np.load("data/glm_mean_coh.npy") * 1e2
    pvalues = np.load("data/glm_mean_coh_pvalues.npy")

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

if plot_trans_prob:
    p = np.load("data/glm_tp.npy")
    pvalues = np.load("data/glm_tp_pvalues.npy")

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

    plt.savefig("plots/tp.png")
    plt.close()

if plot_sum_stats:
    copes = np.load("data/glm_sum_stats.npy")
    pvalues = np.load("data/glm_sum_stats_pvalues.npy")

    cog = np.load("data/cog.npy")
    sum_stats = np.load("data/sum_stats.npy")
    sr = np.sum(sum_stats[:, -1], axis=-1)

    copes /= cog.std()
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
    ax[0].set_ylabel("COPE", fontsize=16)

    # SR
    popt, pcov = optimize.curve_fit(line, cog, sr)
    m, c = popt
    y = line(cog, m, c)

    ax[-1].scatter(cog, sr, s=10, c="black", label="Data")
    ax[-1].plot(cog, y, c="red", lw=2, label=f"y = {m:.2g}x + {c:.3g}")
    ax[-1].set_title(titles[-1], fontsize=16)
    ax[-1].set_xlabel("Cognitive Score", fontsize=16)
    ax[-1].set_ylim(5, None)
    ax[-1].tick_params(labelsize=15)
    ax[-1].legend(fontsize=10)

    plt.tight_layout()
    filename = f"plots/sum_stats.png"
    print("Saving", filename)
    plt.savefig(filename)
