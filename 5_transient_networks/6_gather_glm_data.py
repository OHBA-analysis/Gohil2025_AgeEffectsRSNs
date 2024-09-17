"""Gather data needed to fit a GLM.

"""

import numpy as np
from glob import glob

from osl_dynamics.analysis import power, connectivity

base_dir = "/well/woolrich/users/wlo995/Gohil2024_AgeCognitionEffectsRSNs"
model_dir = f"{base_dir}/5_transient_networks/models/run2"

# Source data file and subjects IDs
files = sorted(glob(f"{base_dir}/1_preproc_and_source_recon/data/src/*/sflip_parc-raw.fif"))
ids = np.array([file.split("/")[-2].split("-")[1] for file in files])

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

def get_targets(id):
    i = np.squeeze(np.argwhere(ids == id))
    P = psd[i]
    p = power.variance_from_spectra(f, P)
    c = connectivity.mean_coherence_from_spectra(f, coh[i])
    mc = np.mean(c, axis=-1)
    c = c[:, m, n]
    return p, c, mc, tp[i], fo[i], lt[i], intv[i], sr[i]

# Load confound data
confound_ids = np.load(f"{base_dir}/3_design_matrix/data/id.npy")
age = np.load(f"{base_dir}/3_design_matrix/data/age.npy")
cog = np.load(f"{base_dir}/3_design_matrix/data/cog.npy")
sex = np.load(f"{base_dir}/3_design_matrix/data/sex.npy")
brain_vol = np.load(f"{base_dir}/3_design_matrix/data/brain_vol.npy")
gm_vol = np.load(f"{base_dir}/3_design_matrix/data/gm_vol.npy")
wm_vol = np.load(f"{base_dir}/3_design_matrix/data/wm_vol.npy")
headsize = np.load(f"{base_dir}/3_design_matrix/data/headsize.npy")
x = np.load(f"{base_dir}/3_design_matrix/data/x.npy")
y = np.load(f"{base_dir}/3_design_matrix/data/y.npy")
z = np.load(f"{base_dir}/3_design_matrix/data/z.npy")

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

for id in ids:
    if id not in confound_ids:
        print(f"sub-{id} not found")
        continue
    print(f"sub-{id}")
    i = confound_ids == id

    # Get data
    p, c, mc, tp_i, fo_i, lt_i, intv_i, sr_i = get_targets(id)
    
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
    age_.append(age[i][0])
    cog_.append(cog[i][0])
    sex_.append(sex[i][0])
    brain_vol_.append(brain_vol[i][0])
    gm_vol_.append(gm_vol[i][0])
    wm_vol_.append(wm_vol[i][0])
    headsize_.append(headsize[i][0])
    x_.append(x[i][0])
    y_.append(y[i][0])
    z_.append(z[i][0])

# Combine summary stats into one array
sum_stats_ = np.swapaxes([fo_, lt_, intv_, sr_], 0, 1)

# Save data
np.save("data/glm_pow.npy", pow_)
np.save("data/glm_coh.npy", coh_)
np.save("data/glm_mean_coh.npy", mean_coh_)
np.save("data/glm_tp.npy", tp_)
np.save("data/glm_sum_stats.npy", sum_stats_)
np.save("data/glm_age.npy", age_)
np.save("data/glm_cog.npy", cog_)
np.save("data/glm_sex.npy", sex_)
np.save("data/glm_brain_vol.npy", brain_vol_)
np.save("data/glm_gm_vol.npy", gm_vol_)
np.save("data/glm_wm_vol.npy", wm_vol_)
np.save("data/glm_headsize.npy", headsize_)
np.save("data/glm_x.npy", x_)
np.save("data/glm_y.npy", y_)
np.save("data/glm_z.npy", z_)
