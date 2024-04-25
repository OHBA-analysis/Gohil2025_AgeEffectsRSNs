"""Gather data needed to fit a GLM.

"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
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

base_dir = "/well/woolrich/users/wlo995/Gohil2024_HealthyAgeingRSNs"

# Load target data
f = np.load(f"{base_dir}/2_time_averaged_analysis/data/f.npy")
psd = np.load(f"{base_dir}/2_time_averaged_analysis/data/psd.npy")
coh = np.load(f"{base_dir}/2_time_averaged_analysis/data/coh.npy")
aec = np.load(f"{base_dir}/2_time_averaged_analysis/data/aec.npy")

freq_bands = [[1, 4], [4, 8], [8, 13], [13, 24], [30, 45]]
m, n = np.triu_indices(coh.shape[-2], k=1)

# Source data file and subjects IDs
files = sorted(glob(f"{base_dir}/1_preproc_and_source_recon/data/src/*/sflip_parc-raw.fif"))
ids = np.array([file.split("/")[-2].split("-")[1] for file in files])

def get_targets(id):
    i = np.squeeze(np.argwhere(ids == id))
    p = psd[i]
    c = coh[i]
    a = aec[i]
    p = np.array([power.variance_from_spectra(f, p, frequency_range=b) for b in freq_bands]).T
    c = np.array([connectivity.mean_coherence_from_spectra(f, c, frequency_range=b) for b in freq_bands])
    mc = connectivity.mean_connections(c).T
    c = c.T[m, n]
    ma = connectivity.mean_connections(a.T).T
    a = a[m, n]
    return p, c, mc, a, ma

# Lists to hold target data
pow_ = []
coh_ = []
mean_coh_ = []
aec_ = []
mean_aec_ = []

# Lists to hold regressor data
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
        p, c, mc, a, ma = get_targets(id)
        if len(p) == 0:
            print(f"sub-{id} has no psd")
            continue
        hs, x, y, z = get_headsize_and_pos(preproc_file)

        # Add to target data lists
        pow_.append(p)
        coh_.append(c)
        mean_coh_.append(mc)
        aec_.append(a)
        mean_aec_.append(ma)

        # Add to regressor lists
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
np.save("data/pow.npy", pow_)
np.save("data/coh.npy", coh_)
np.save("data/mean_coh.npy", mean_coh_)
np.save("data/aec.npy", aec_)
np.save("data/mean_aec.npy", mean_aec_)
np.save("data/age.npy", age_)
np.save("data/sex.npy", sex_)
np.save("data/brain_vol.npy", brain_vol_)
np.save("data/gm_vol.npy", gm_vol_)
np.save("data/hip_vol.npy", hip_vol_)
np.save("data/headsize.npy", headsize_)
np.save("data/x.npy", x_)
np.save("data/y.npy", y_)
np.save("data/z.npy", z_)
