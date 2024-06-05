"""Gather data needed to fit a GLM.

"""

import os
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

from osl_dynamics.analysis import power

os.makedirs("data", exist_ok=True)

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

base_dir = "/well/woolrich/users/wlo995/Gohil2024_HealthyAgeingRSNs"
static_dir = f"{base_dir}/2_time_averaged_analysis"
model_dir = f"{base_dir}/3_transient_network_analysis/models/run2"

bands = [[1, 4], [4, 8], [8, 13], [13, 24], [30, 45]]

# Load static power
static_pow = np.load(f"{static_dir}/data/pow.npy")

# Load summary stats
fo = np.load(f"{model_dir}/fo.npy")

# Load state PSD and calculate power
f = np.load(f"{model_dir}/f.npy")
state_psd = np.load(f"{model_dir}/psd.npy")
state_pow = np.array([power.variance_from_spectra(f, state_psd, frequency_range=b) for b in bands])
state_pow = np.rollaxis(state_pow, 0, -1)

# Reorder the states
p = np.mean(state_psd, axis=(0,2,3))
order = np.argsort(p)[::-1]
fo = fo[:, order]
state_pow = state_pow[:, order]

# Source data file and subjects IDs
files = sorted(glob(f"{base_dir}/1_preproc_and_source_recon/data/src/*/sflip_parc-raw.fif"))
ids = np.array([file.split("/")[-2].split("-")[1] for file in files])

def get_targets(id):
    i = np.squeeze(np.argwhere(ids == id))
    return static_pow[i], state_pow[i], fo[i]

# Lists to hold target data
static_pow_ = []
state_pow_ = []
fo_ = []

# Lists to hold regressor data
age_ = []
sex_ = []
brain_vol_ = []
gm_vol_ = []
wm_vol_ = []
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
        static_p_i, state_p_i, fo_i = get_targets(id)
        if len(static_p_i) == 0:
            print(f"sub-{id} has no psd")
            continue
        hs, x, y, z = get_headsize_and_pos(preproc_file)

        # Add data to lists
        static_pow_.append(static_p_i)
        state_pow_.append(state_p_i)
        fo_.append(fo_i)
        age_.append(row["Fixed_Age"])
        sex_.append(row["Sex (1=female, 2=male)"])
        brain_vol_.append(row["Brain_Vol"])
        gm_vol_.append(row["GM_Vol_Norm"])
        wm_vol_.append(row["WM_Vol_Norm"])
        hip_vol_.append(row["Hippo_Vol_Norm"])
        headsize_.append(hs)
        x_.append(x)
        y_.append(y)
        z_.append(z)

    else:
        print(f"sub-{id} not found")

# Save data
np.save("data/static_pow.npy", static_pow_)
np.save("data/state_pow.npy", state_pow_)
np.save("data/fo.npy", fo_)
np.save("data/age.npy", age_)
np.save("data/sex.npy", sex_)
np.save("data/brain_vol.npy", brain_vol_)
np.save("data/gm_vol.npy", gm_vol_)
np.save("data/wm_vol.npy", wm_vol_)
np.save("data/hip_vol.npy", hip_vol_)
np.save("data/headsize.npy", headsize_)
np.save("data/x.npy", x_)
np.save("data/y.npy", y_)
np.save("data/z.npy", z_)
