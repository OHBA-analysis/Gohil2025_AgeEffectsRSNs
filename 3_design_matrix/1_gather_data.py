"""Gather data needed to fit a GLM.

"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path

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

# Load participant confound data and cognitive score
part_csv = pd.read_csv("../3_design_matrix/data/all_collated_camcan.csv")
cog_csv = pd.read_csv("../2_cognitive_score/data/cognitive_metrics_pca.csv")

cog_ids = cog_csv["ID"].values
cog_scores = cog_csv["Component 0"].values

# Lists to hold regressor data
id_ = []
age_ = []
cog_ = []
sex_ = []
brain_vol_ = []
gm_vol_ = []
wm_vol_ = []
headsize_ = []
x_ = []
y_ = []
z_ = []

for _, row in part_csv.iterrows():
    id = row["ID"]
    preproc_file = (
        f"../1_preproc_and_source_recon/output/sub-{id}/sub-{id}_preproc-raw.fif"
    )
    if Path(preproc_file).exists():
        print(f"sub-{id}")
        if id not in cog_ids:
            continue

        # Get data
        hs, x, y, z = get_headsize_and_pos(preproc_file)
        cog = cog_scores[cog_ids == id][0]
        
        # Add to regressor lists
        id_.append(id)
        age_.append(row["Fixed_Age"])
        cog_.append(cog)
        sex_.append(row["Sex (1=female, 2=male)"])
        brain_vol_.append(row["Brain_Vol"])
        gm_vol_.append(row["GM_Vol_Norm"])
        wm_vol_.append(row["WM_Vol_Norm"])
        headsize_.append(hs)
        x_.append(x)
        y_.append(y)
        z_.append(z)

    else:
        print(f"sub-{id} not found")

# Save data
np.save("data/id.npy", id_)
np.save("data/age.npy", age_)
np.save("data/cog.npy", cog_)
np.save("data/sex.npy", sex_)
np.save("data/brain_vol.npy", brain_vol_)
np.save("data/gm_vol.npy", gm_vol_)
np.save("data/wm_vol.npy", wm_vol_)
np.save("data/headsize.npy", headsize_)
np.save("data/x.npy", x_)
np.save("data/y.npy", y_)
np.save("data/z.npy", z_)
