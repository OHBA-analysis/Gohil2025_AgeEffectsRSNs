"""Plot group-average PSD and networks.

"""

import os

os.makedirs("plots", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

plot_psd = True
plot_coh = True
plot_pow_maps = True
plot_coh_nets = True
plot_aec_nets = True
plot_mean_coh_maps = True
plot_mean_aec_maps = True

freq_bands = [[1, 4], [4, 8], [8, 13], [13, 24], [30, 45]]
cmap = plt.get_cmap("tab10")

if plot_psd:
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 28,
    })

    f = np.load("../data/f.npy")
    psd = np.load("../data/psd.npy")
    w = np.load("../data/w.npy")

    gpsd = np.average(psd, axis=0, weights=w)
    p = np.mean(gpsd, axis=0)
    e = np.std(gpsd, axis=0)

    fig, ax = plotting.plot_line(
        [f],
        [p],
        x_label="Frequency (Hz)",
        y_label="PSD (a.u.)",
        x_range=[f[0], f[-1]],
        y_range=[None, 0.1],
        plot_kwargs={"linewidth": 3, "color": "black"},
    )
    ax.fill_between(f, p - e, p + e, color="black", alpha=0.3)
    for i, band in enumerate(freq_bands):
        ax.axvspan(band[0], band[1], color=cmap(i), alpha=0.3)
    plotting.save(fig, filename="plots/psd.png", tight_layout=True)

if plot_coh:
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 28,
    })

    f = np.load("../data/f.npy")
    coh = np.load("../data/coh.npy")
    w = np.load("../data/w.npy")

    i, j = np.triu_indices(coh.shape[-2], k=1)
    gcoh = np.average(coh, axis=0, weights=w)
    gcoh = gcoh[i, j]
    c = np.mean(gcoh, axis=0)
    e = np.std(gcoh, axis=0)

    fig, ax = plotting.plot_line(
        [f],
        [c],
        x_label="Frequency (Hz)",
        y_label="Coherence",
        x_range=[f[0], f[-1]],
        y_range=[0.025, 0.14],
        plot_kwargs={"linewidth": 3, "color": "black"},
    )
    ax.set_xlabel("Frequency (Hz)", fontsize=20)
    ax.set_ylabel("Coherence", fontsize=20)
    ax.fill_between(f, c - e, c + e, color="black", alpha=0.3)
    for i, band in enumerate(freq_bands):
        ax.axvspan(band[0], band[1], color=cmap(i), alpha=0.3)
    plotting.save(fig, filename="plots/coh.png", tight_layout=True)

if plot_pow_maps:
    from osl_dynamics.analysis import power
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    f = np.load("../data/f.npy")
    psd = np.load("../data/psd.npy")
    w = np.load("../data/w.npy")

    # Group average
    psd = np.average(psd, axis=0, weights=w)

    # Power in each band
    p = []
    for i, band in enumerate(freq_bands):
        p.append(power.variance_from_spectra(f, psd, frequency_range=band))
    p = np.array(p)

    # Weighted average power across frequency bands
    ref = np.mean(p, axis=0)

    # Apply reference
    p -= ref[np.newaxis, ...]
    p /= ref[np.newaxis, ...]

    # Save
    power.save(
        p,
        filename=f"plots/pow_band_.png",
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "alpha": 1,
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
    )

if plot_coh_nets:
    from osl_dynamics.analysis import connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    f = np.load("../data/f.npy")
    coh = np.load("../data/coh.npy")
    w = np.load("../data/w.npy")

    # Group average
    coh = np.average(coh, axis=0, weights=w)

    # Mean coherence in each band
    c = []
    for i, band in enumerate(freq_bands):
        c.append(connectivity.mean_coherence_from_spectra(f, coh, frequency_range=band))
    c = np.array(c)

    # Apply reference
    c -= np.mean(c, axis=0, keepdims=True)

    # Threshold
    c = connectivity.threshold(c, percentile=95, absolute_value=True)

    # Save
    connectivity.save(
        c,
        filename="plots/coh_band_.png",
        plot_kwargs={
            "annotate": False,
            "display_mode": "xz",
        },
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    )

if plot_aec_nets:
    from osl_dynamics.analysis import connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # Load
    aec = np.load("../data/aec.npy")
    w = np.load("../data/w.npy")

    # Group average
    c = np.average(aec, axis=0, weights=w)
    c = np.moveaxis(c, -1, 0)

    # Apply reference
    c -= np.mean(c, axis=0, keepdims=True)

    # Threshold
    c = connectivity.threshold(c, percentile=95, absolute_value=True)

    # Save
    connectivity.save(
        c,
        filename="plots/aec_band_.png",
        plot_kwargs={
            "annotate": False,
            "display_mode": "xz",
        },
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
    )

if plot_mean_coh_maps:
    from osl_dynamics.analysis import power, connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    f = np.load("../data/f.npy")
    coh = np.load("../data/coh.npy")
    w = np.load("../data/w.npy")

    # Group average
    coh = np.average(coh, axis=0, weights=w)

    # Mean coherence in each band
    c = []
    for i, band in enumerate(freq_bands):
        c.append(connectivity.mean_coherence_from_spectra(f, coh, frequency_range=band))
    c = np.array(c)

    # Average over edges
    c = connectivity.mean_connections(c)

    # Apply reference
    c -= np.mean(c, axis=0, keepdims=True)

    # Save
    power.save(
        c,
        filename=f"plots/mean_coh_band_.png",
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "alpha": 1,
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
    )

if plot_mean_aec_maps:
    from osl_dynamics.analysis import power, connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    # Load
    aec = np.load("../data/aec.npy")
    w = np.load("../data/w.npy")

    # Group average
    c = np.average(aec, axis=0, weights=w)
    c = np.moveaxis(c, -1, 0)

    # Average over edges
    c = connectivity.mean_connections(c)

    # Apply reference
    c -= np.mean(c, axis=0, keepdims=True)

    # Save
    power.save(
        c,
        filename=f"plots/mean_aec_band_.png",
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "alpha": 1,
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
    )
