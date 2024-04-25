"""Gather data needed to fit a GLM.

"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, optimize
from glob import glob

from osl_dynamics.analysis import power, connectivity

def get_device_fids(fif_file):
    raw = mne.io.read_raw_fif(fif_file, verbose=False)
    head_fids = mne.viz._3d._fiducial_coords(raw.info["dig"])
    head_fids = np.vstack(([0, 0, 0], head_fids))
    fid_space = raw.info["dig"][0]["coord_frame"]
    dev2head = raw.info["dev_head_t"]
    head2dev = mne.transforms.invert_transform(dev2head)
    return mne.transforms.apply_trans(head2dev, head_fids)

def get_headsize_and_pos(fif_file):
    dfids = get_device_fids(fif_file)
    hs = np.abs(dfids[1, 0] - dfids[3, 0])
    x = dfids[0, 0]
    y = dfids[0, 1]
    z = dfids[0, 2]
    return hs, x, y, z

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def get_precise_freq(x, y):
    try:
        popt, pcov = optimize.curve_fit(gauss, x, y, p0=[y[1], x[1], x[2]-x[0]])
        return popt[1]
    except:
        return

base_dir = "/well/woolrich/users/wlo995/Gohil2024_HealthyAgeingRSNs"

# Load target data
f = np.load(f"{base_dir}/2_time_averaged_analysis/data/f.npy")
psd = np.load(f"{base_dir}/2_time_averaged_analysis/data/psd.npy")

# Only consider the theta/alpha band
psd[:, :, f < 4] = 0
psd[:, :, f > 13] = 0

# Only keep occipital parcels
parcels = [0, 1, 2, 3, 26, 27, 28, 29]
psd = psd[:, parcels]

# Source data file and subjects IDs
files = sorted(glob(f"{base_dir}/1_preproc_and_source_recon/data/src/*/sflip_parc-raw.fif"))
ids = np.array([file.split("/")[-2].split("-")[1] for file in files])

def get_targets(id):
    i = np.squeeze(np.argwhere(ids == id))
    p = np.mean(psd[i], axis=0)
    indices, _ = signal.find_peaks(p)
    peaks = f[indices]
    peaks = peaks[peaks != 4.39453125]
    if len(peaks) == 0:
        return
    elif len(peaks) > 1:
        i = np.argmax([p[f == peak] for peak in peaks])
        index = np.argwhere(f == peaks[i])[0,0]
        x = f[index-1:index+2]
        y = p[index-1:index+2]
        P = get_precise_freq(x, y)
    else:
        index = np.argwhere(f == peaks[0])[0,0]
        x = f[index-1:index+2]
        y = p[index-1:index+2]
        P = get_precise_freq(x, y)
    return P

# Lists to hold data
peak_freq_ = []
age_ = []
sex_ = []
brain_vol_ = []
gm_vol_ = []
hip_vol_ = []
headsize_ = []
x_ = []
y_ = []
z_ = []

# Load participant confound data
csv = pd.read_csv(f"{base_dir}/all_collated_camcan.csv")

for _, row in csv.iterrows():
    id = row["ID"]
    preproc_file = (
        f"{base_dir}/1_preproc_and_source_recon/data/preproc"
        f"/mf2pt2_sub-{id}_ses-rest_task-rest_meg"
        f"/mf2pt2_sub-{id}_ses-rest_task-rest_meg_preproc_raw.fif"
    )
    if Path(preproc_file).exists():
        print(f"sub-{id}")
        if id == "CC221585":
            continue

        # Get data
        p = get_targets(id)
        if p is None:
            print(f"could not find peak freq for sub-{id}")
            continue
        hs, x, y, z = get_headsize_and_pos(preproc_file)

        # Add to lists
        peak_freq_.append(p)
        age_.append(row["Fixed_Age"])
        sex_.append(row["Sex (1=female, 2=male)"])
        brain_vol_.append(row["Brain_Vol"])
        gm_vol_.append(row["GM_Vol_Norm"])
        hip_vol_.append(row["Hippo_Vol_Norm"])
        headsize_.append(hs)
        x_.append(x)
        y_.append(y)
        z_.append(z)

    else:
        print(f"sub-{id} not found")

# Save data
np.save("data/peak_freq.npy", peak_freq_)
np.save("data/age.npy", age_)
np.save("data/sex.npy", sex_)
np.save("data/brain_vol.npy", brain_vol_)
np.save("data/gm_vol.npy", gm_vol_)
np.save("data/hip_vol.npy", hip_vol_)
np.save("data/headsize.npy", headsize_)
np.save("data/x.npy", x_)
np.save("data/y.npy", y_)
np.save("data/z.npy", z_)
