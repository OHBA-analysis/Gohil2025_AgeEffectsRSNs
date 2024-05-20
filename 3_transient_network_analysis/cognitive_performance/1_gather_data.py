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
model_dir = f"{base_dir}/3_transient_network_analysis/models/run2"

# Load multitaper
f = np.load(f"{model_dir}/f.npy")
psd = np.load(f"{model_dir}/psd.npy")
coh = np.load(f"{model_dir}/coh.npy")

m, n = np.triu_indices(coh.shape[-2], k=1)

# Load summary stats
fo = np.load(f"{model_dir}/fo.npy")
lt = np.load(f"{model_dir}/lt.npy")
intv = np.load(f"{model_dir}/intv.npy")
sr = np.load(f"{model_dir}/sr.npy")
tp = np.load(f"{model_dir}/tp.npy")

# Reorder the states
p = np.mean(psd, axis=(0,2,3))
order = np.argsort(p)[::-1]

psd = psd[:, order]
coh = coh[:, order]
fo = fo[:, order]
lt = lt[:, order]
intv = intv[:, order]
sr = sr[:, order]
tp = [tp_[np.ix_(order, order)] for tp_ in tp]

# Source data file and subjects IDs
files = sorted(glob(f"{base_dir}/1_preproc_and_source_recon/data/src/*/sflip_parc-raw.fif"))
ids = np.array([file.split("/")[-2].split("-")[1] for file in files])

def get_targets(id):
    i = np.squeeze(np.argwhere(ids == id))
    P = psd[i]
    p = power.variance_from_spectra(f, P)
    c = connectivity.mean_coherence_from_spectra(f, coh[i])
    mc = np.mean(c, axis=-1)
    c = c[:, m, n]
    return p, c, mc, tp[i], fo[i], lt[i], intv[i], sr[i]

# Lists to hold target data
pow_ = []
coh_ = []
mean_coh_ = []
tp_ = []
fo_ = []
lt_ = []
intv_ = []
sr_ = []

# Lists to hold regressor data
category_list_ = []
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

# Load participant confound data and cognitive score
part_csv = pd.read_csv(f"{base_dir}/all_collated_camcan.csv")
cog_csv = pd.read_csv("../../2_time_averaged_analysis/cognitive_performance/do_pca/data/cognitive_metrics_pca.csv")

cog_ids = cog_csv["ID"].values
cog_scores = cog_csv["Component 0"].values
bottom_thres, top_thres = np.quantile(cog_scores, [0.25, 0.75])

for _, row in part_csv.iterrows():
    id = row["ID"]
    if id not in cog_ids:
        continue
    cog = cog_scores[cog_ids == id][0]

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
        p, c, mc, tp_i, fo_i, lt_i, intv_i, sr_i = get_targets(id)
        if len(p) == 0:
            print(f"sub-{id} has no psd")
            continue
        hs, x, y, z = get_headsize_and_pos(preproc_file)

        if cog > top_thres:
            category_list_.append(1)
        elif cog < bottom_thres:
            category_list_.append(2)
        else:
            continue
        
        # Add to target data lists
        pow_.append(p)
        coh_.append(c)
        mean_coh_.append(mc)
        tp_.append(tp_i)
        fo_.append(fo_i)
        lt_.append(lt_i)
        intv_.append(intv_i)
        sr_.append(sr_i)

        # Add to regressor lists
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

# Combine summary stats into one array
sum_stats_ = np.swapaxes([fo_, lt_, intv_, sr_], 0, 1)

# Save data
np.save("data/pow.npy", pow_)
np.save("data/coh.npy", coh_)
np.save("data/mean_coh.npy", mean_coh_)
np.save("data/tp.npy", tp_)
np.save("data/sum_stats.npy", sum_stats_)
np.save("data/category_list.npy", category_list_)
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
