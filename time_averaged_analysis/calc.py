"""Calculate static (time-averaged) quantities.

"""

import os
import numpy as np
from glob import glob

os.makedirs("data", exist_ok=True)

calc_spectra = True
calc_pow = True
calc_coh = True
calc_aec = True

freq_bands = [[1, 4], [4, 8], [8, 13], [13, 24], [30, 45]]

files = sorted(glob("/well/woolrich/projects/camcan/spring23/src/*/sflip_parc-raw.fif"))

if calc_spectra:
    from osl_dynamics.analysis import static
    from osl_dynamics.data import Data

    data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=16)
    data = data.trim_time_series(n_embeddings=15, sequence_length=4000)

    f, psd, coh, w = static.welch_spectra(
        data=data,
        window_length=500,
        sampling_frequency=250,
        frequency_range=[1, 45],
        return_weights=True,
        standardize=True,
        calc_coh=True,
        n_jobs=16,
    )

    print("Saving spectra")
    np.save("data/f.npy", f)
    np.save("data/psd.npy", psd)
    np.save("data/coh.npy", coh)
    np.save("data/w.npy", w)

if calc_pow:
    from osl_dynamics.analysis import power

    f = np.load("data/f.npy")
    psd = np.load("data/psd.npy")

    pow_ = []
    for band in freq_bands:
        p = power.variance_from_spectra(f, psd, frequency_range=band)
        pow_.append(p)
    pow_ = np.swapaxes(pow_, 0, 1)

    np.save("data/pow.npy", pow_)

if calc_coh:
    from osl_dynamics.analysis import connectivity

    f = np.load("data/f.npy")
    coh = np.load("data/coh.npy")

    mean_coh = []
    for band in freq_bands:
        c = connectivity.mean_coherence_from_spectra(f, coh, frequency_range=band)
        mean_coh.append(c)
    mean_coh = np.swapaxes(mean_coh, 0, 1)

    np.save("data/mean_coh.npy", mean_coh)

if calc_aec:
    from osl_dynamics.analysis import static
    from osl_dynamics.data import Data

    data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=16)
    data.standardize()
    x = data.time_series()

    data = Data(x, sampling_frequency=250, load_memmaps=False, n_jobs=16)

    aec = []
    for band in freq_bands:
        data.filter(low_freq=band[0], high_freq=band[1], use_raw=True)
        data.amplitude_envelope()
        x = data.time_series()
        aec.append(static.functional_connectivity(x))
    aec = np.moveaxis(aec, 0, -1)

    np.save("data/aec.npy", aec)
