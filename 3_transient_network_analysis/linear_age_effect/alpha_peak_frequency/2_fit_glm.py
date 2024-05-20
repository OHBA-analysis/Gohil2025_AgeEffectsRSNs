"""Fit a GLM and do stats.

"""

import numpy as np
import glmtools as glm
from scipy import stats

def do_stats(
    design,
    data,
    model,
    contrast_idx,
    nperms=1000,
    metric="copes",
    tail=0,
    pooled_dims=(),
    nprocesses=16,
):
    perm = glm.permutations.MaxStatPermutation(
        design=design,
        data=data,
        contrast_idx=contrast_idx,
        nperms=nperms,
        metric=metric,
        tail=tail,
        pooled_dims=pooled_dims,
        nprocesses=nprocesses,
    )
    nulls = np.squeeze(perm.nulls)
    if metric == "tstats":
        tstats = abs(model.tstats[contrast_idx])
        percentiles = stats.percentileofscore(nulls, tstats)
    elif metric == "copes":
        copes = abs(model.copes[contrast_idx])
        percentiles = stats.percentileofscore(nulls, copes)
    return 1 - percentiles / 100

def fit_glm_and_do_stats(target):
    remove = np.isnan(target)

    age = np.load("data/age.npy")
    sex = np.load("data/sex.npy")
    brain_vol = np.load("data/brain_vol.npy")
    gm_vol = np.load("data/gm_vol.npy")
    wm_vol = np.load("data/wm_vol.npy")
    hip_vol = np.load("data/hip_vol.npy")
    headsize = np.load("data/headsize.npy")
    x = np.load("data/x.npy")
    y = np.load("data/y.npy")
    z = np.load("data/z.npy")

    data = glm.data.TrialGLMData(
        data=target[~remove],
        age=age[~remove],
        sex=sex[~remove],
        brain_vol=brain_vol[~remove],
        gm_vol=gm_vol[~remove],
        wm_vol=wm_vol[~remove],
        hip_vol=hip_vol[~remove],
        headsize=headsize[~remove],
        x=x[~remove],
        y=y[~remove],
        z=z[~remove],
    )

    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Age", rtype="Parametric", datainfo="age", preproc="z")
    DC.add_regressor(name="Sex", rtype="Parametric", datainfo="sex", preproc="z")
    DC.add_regressor(name="Brain Vol.", rtype="Parametric", datainfo="brain_vol", preproc="z")
    DC.add_regressor(name="GM Vol.", rtype="Parametric", datainfo="gm_vol", preproc="z")
    DC.add_regressor(name="WM Vol.", rtype="Parametric", datainfo="wm_vol", preproc="z")
    DC.add_regressor(name="Hippo. Vol.", rtype="Parametric", datainfo="hip_vol", preproc="z")
    DC.add_regressor(name="Head Size", rtype="Parametric", datainfo="headsize", preproc="z")
    DC.add_regressor(name="x", rtype="Parametric", datainfo="x", preproc="z")
    DC.add_regressor(name="y", rtype="Parametric", datainfo="y", preproc="z")
    DC.add_regressor(name="z", rtype="Parametric", datainfo="z", preproc="z")
    DC.add_regressor(name="Mean", rtype="Constant")

    DC.add_contrast(name="", values=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    design = DC.design_from_datainfo(data.info)
    design.plot_summary(savepath="plots/glm_design.png", show=False)
    design.plot_leverage(savepath="plots/glm_leverage.png", show=False)
    design.plot_efficiency(savepath="plots/glm_efficiency.png", show=False)

    model = glm.fit.OLSModel(design, data)

    mean = model.betas[-1]
    age = model.copes[0]
    pvalues = do_stats(design, data, model, contrast_idx=0)
    return mean, age, pvalues

target = np.load("data/peak_freq.npy")
n_states = target.shape[-1]
age = []
pvalues = []
for i in range(n_states):
    _, a, p = fit_glm_and_do_stats(target[:, i])
    age.append(a[0])
    pvalues.append(p[0])
np.save("data/glm_age.npy", age)
np.save("data/glm_age_pvalues.npy", pvalues)
