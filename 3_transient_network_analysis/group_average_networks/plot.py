"""Plot group-average HMM networks.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plot_psds = False
plot_pow_maps = False
plot_coh_nets = False
plot_coh_maps = True
plot_pow_vs_coh = False
plot_trans_prob = False
plot_sum_stats = False

os.makedirs("plots", exist_ok=True)

model_dir = "../models/run2"

psd = np.load(f"{model_dir}/psd.npy")
p = np.mean(psd, axis=(0,2,3))
order = np.argsort(p)[::-1]

cmap = plt.get_cmap("tab10")

if plot_psds:
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 28,
        "lines.linewidth": 4,
    })

    f = np.load(f"{model_dir}/f.npy")
    psd = np.load(f"{model_dir}/psd.npy")[:, order]
    w = np.load(f"{model_dir}/w.npy")
    fo = np.load(f"{model_dir}/fo.npy")[:, order]

    # Group average PSD for each state
    gpsd = np.average(psd, axis=0, weights=w)
    gfo = np.average(fo, axis=0, weights=w)
    mgpsd = np.average(gpsd, axis=0, weights=gfo)
    p = np.mean(gpsd, axis=1)
    mp = np.mean(mgpsd, axis=0)

    for i in range(p.shape[0]):
        fig, ax = plotting.plot_line(
            [f],
            [mp],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
            x_range=[f[0], f[-1]],
            y_range=[0, 0.105],
            plot_kwargs={"color": "black", "linestyle": "--"},
        )
        ax.plot(f, p[i], color=cmap(i))
        plotting.save(fig, f"plots/psd_{i:02d}.png", tight_layout=True)

if plot_pow_maps:
    from osl_dynamics.analysis import power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    f = np.load(f"{model_dir}/f.npy")
    psd = np.load(f"{model_dir}/psd.npy")[:, order]
    w = np.load(f"{model_dir}/w.npy")
    fo = np.load(f"{model_dir}/fo.npy")[:, order]

    gpsd = np.average(psd, axis=0, weights=w)
    p = power.variance_from_spectra(f, gpsd)
    gfo = np.average(fo, axis=0, weights=w)

    power.save(
        p,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
        subtract_mean=True,
        mean_weights=gfo,
        filename="plots/pow_.png",
    )

if plot_coh_nets:
    from osl_dynamics.analysis import connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    f = np.load(f"{model_dir}/f.npy")
    coh = np.load(f"{model_dir}/coh.npy")[:, order]
    w = np.load(f"{model_dir}/w.npy")
    fo = np.load(f"{model_dir}/fo.npy")[:, order]

    gcoh = np.average(coh, axis=0, weights=w)
    c = connectivity.mean_coherence_from_spectra(f, gcoh)

    gfo = np.average(fo, axis=0, weights=w)
    c -= np.average(c, axis=0, weights=gfo)
    c = connectivity.threshold(c, percentile=97, absolute_value=True)

    connectivity.save(
        c,
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "display_mode": "xz",
            "annotate": False,
        },
        filename="plots/coh_.png",
    )

if plot_coh_maps:
    from osl_dynamics.analysis import connectivity, power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    f = np.load(f"{model_dir}/f.npy")
    coh = np.load(f"{model_dir}/coh.npy")[:, order]
    w = np.load(f"{model_dir}/w.npy")
    fo = np.load(f"{model_dir}/fo.npy")[:, order]

    gcoh = np.average(coh, axis=0, weights=w)
    c = connectivity.mean_coherence_from_spectra(f, gcoh)
    c = connectivity.mean_connections(c)

    gfo = np.average(fo, axis=0, weights=w)

    power.save(
        c,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"views": ["lateral"], "symmetric_cbar": True},
        subtract_mean=True,
        mean_weights=gfo,
        filename="plots/mean_coh_.png",
    )

if plot_pow_vs_coh:
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from osl_dynamics.analysis import power, connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
    })

    f = np.load(f"{model_dir}/f.npy")
    psd = np.load(f"{model_dir}/psd.npy")[:, order]
    coh = np.load(f"{model_dir}/coh.npy")[:, order]
    w = np.load(f"{model_dir}/w.npy")

    p = power.variance_from_spectra(f, psd)
    c = connectivity.mean_coherence_from_spectra(f, coh)
    c = connectivity.mean_connections(c)

    p_ = np.mean(p, axis=0)
    c_ = np.mean(c, axis=0)

    plotting.plot_scatter(
        p_,
        c_,
        labels=[f"State {i}" for i in range(1, 11)],
        x_label="Power",
        y_label="Coherence",
        filename="plots/pow_vs_coh_parcels.png",
    )

    p_ = np.mean(p, axis=-1).T
    c_ = np.mean(c, axis=-1).T

    plotting.plot_scatter(
        p_,
        c_,
        labels=[f"State {i}" for i in range(1, 11)],
        x_label="Power (a.u.)",
        y_label="Mean Coherence",
        filename="plots/pow_vs_coh_subjects.png",
    )

if plot_trans_prob:
    tp = np.load(f"{model_dir}/trans_prob.npy")
    tp = tp[np.ix_(order, order)]

    diag = np.diag(tp.copy())
    np.fill_diagonal(tp, 0)

    fig, ax = plt.subplots()

    im = ax.matshow(tp)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    cax.tick_params(labelsize=15)

    ax.set_xticklabels([""] + ["1", "3", "5", "7", "9"])
    ax.set_yticklabels([""] + ["1", "3", "5", "7", "9"])
    ax.tick_params(labelsize=15)
    ax.set_xlabel("State: To", fontsize=16)
    ax.set_ylabel("State: From", fontsize=16)
    ax.xaxis.set_label_position("top")

    plt.savefig("plots/trans_prob.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(4,8))

    im = ax.matshow(diag[:, np.newaxis])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="50%", pad=0.25)
    fig.colorbar(im, cax=cax, orientation="vertical")
    cax.tick_params(labelsize=15)

    ax.set_xticklabels([""])
    ax.set_yticklabels([""] + ["1", "3", "5", "7", "9"])
    ax.tick_params(labelsize=15)
    ax.set_ylabel("State", fontsize=16)

    plt.savefig("plots/trans_prob_diag.png")
    plt.close()

if plot_sum_stats:
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fo = np.load(f"{model_dir}/fo.npy")[:, order]
    lt = np.load(f"{model_dir}/lt.npy")[:, order]
    intv = np.load(f"{model_dir}/intv.npy")[:, order]
    sr = np.load(f"{model_dir}/sr.npy")[:, order]

    plotting.plot_hmm_summary_stats(fo, lt, intv, sr, filename="plots/sum_stats.png")
