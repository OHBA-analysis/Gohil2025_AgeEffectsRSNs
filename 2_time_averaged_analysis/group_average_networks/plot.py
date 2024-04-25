"""Plot group-average PSD and networks.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

plot_psd = False
plot_coh = False
plot_pow_maps = False
plot_coh_nets = False
plot_aec_nets = False
plot_mean_coh_maps = False
plot_mean_aec_maps = False
plot_mean_coh_vs_pow_vs_age = True

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

if plot_mean_coh_vs_pow_vs_age:
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from osl_dynamics.analysis import power, connectivity
    from osl_dynamics.utils import plotting

    plotting.set_style({
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # Load static PSDs
    f = np.load("../data/f.npy")
    psd = np.load("../data/psd.npy")
    coh = np.load("../data/coh.npy")

    # Age cohorts
    width = 10
    age = np.load("../data/age.npy").astype(int)
    n_groups = (age.max() - age.min()) // width + 1
    start_ages = [width * i + age.min() for i in range(n_groups)]
    start_ages[-1] += 1  # include participants aged 88 into the last group
    print("Groups:", start_ages)
    print()

    P_bands = []
    C_bands = []
    eP_bands = []
    eC_bands = []
    for i, band in enumerate(freq_bands):
        # Power integrated over frequency for each component
        p = power.variance_from_spectra(f, psd, frequency_range=[band[0], band[1]])

        # Average coherence across all frequencies in each component
        c = connectivity.mean_coherence_from_spectra(f, coh, frequency_range=[band[0], band[1]])

        # Get the average coherence for each parcel
        c = connectivity.mean_connections(c)
        
        # Coherence vs power for each subject, averaged over parcels
        p_ = np.mean(p, axis=-1)
        c_ = np.mean(c, axis=-1)
        
        # Calculate the average power/coherence for each group
        P = []
        C = []
        eP = []
        eC = []
        groups = []
        for j in range(len(start_ages) - 1):
            groups.append((start_ages[j] + start_ages[j + 1]) / 2)
            keep = np.logical_and(age >= start_ages[j], age < start_ages[j + 1])
            P.append(np.mean(p_[keep]))
            C.append(np.mean(c_[keep]))
            eP.append(np.std(p_[keep]) / np.sqrt(p_[keep].shape[0]))
            eC.append(np.std(c_[keep]) / np.sqrt(c_[keep].shape[0]))
        P = np.array(P)
        C = np.array(C)
        eP = np.array(eP)
        eC = np.array(eC)
        groups = np.array(groups)

        P_bands.append(P)
        eP_bands.append(eP)
        C_bands.append(C)
        eC_bands.append(eC)

    markers = ["o", "v", "s", "X", "x"]
    labels = [
        r"$\delta$ (1-4 Hz)",
        r"$\theta$ (4-8 Hz)",
        r"$\alpha$ (8-13 Hz)",
        r"$\beta$ (13-24 Hz)",
        r"$\gamma$ (30-45 Hz)",
    ]

    # Plot with age
    cmap = plt.get_cmap()
    fig, ax = plt.subplots(figsize=(8,5))
    for P, C, eP, eC in zip(P_bands, C_bands, eP_bands, eC_bands):
        for p, c, ep, ec, g in zip(P, C, eP, eC, groups):
            ax.errorbar(p, c, xerr=ep, yerr=ec, c=cmap(g / groups.max()))
    for P, C, m, l in zip(P_bands, C_bands, markers, labels):
        pc = ax.scatter(P, C, c=groups, marker=m, label=l)
    ax.set_xlabel("Power (a.u.)", fontsize=18)
    ax.set_ylabel("Mean Coherence", fontsize=18)
    #ax.set_ylim(None, 0.095)
    ax.legend(loc=2, fontsize=18)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pc, cax=cax, orientation="vertical")
    cax.set_ylabel("Age (years)", fontsize=18)
    plt.tight_layout()
    filename = f"plots/pow_vs_coh_vs_age.png"
    print(f"Saving {filename}")
    plt.savefig(filename)
    plt.close()

