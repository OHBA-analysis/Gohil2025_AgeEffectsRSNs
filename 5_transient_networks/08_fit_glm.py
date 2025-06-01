"""Fit a GLM and do stats.

"""

import numpy as np
import glmtools as glm
from scipy import stats

do_pow = True
do_coh = True
do_mean_coh = True
do_trans_prob = True
do_sum_stats = True

def do_stats(
    design,
    data,
    model,
    contrast_idx,
    nperms=1000,
    metric="copes",
    tail=0,
    pooled_dims=(1,2),
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

def fit_glm_and_do_stats(target, metric="copes"):
    data = glm.data.TrialGLMData(
        data=target,
        age=np.load("data/glm/age.npy"),
        cog=np.load("data/glm/cog.npy"),
        sex=np.load("data/glm/sex.npy"),
        brain_vol=np.load("data/glm/brain_vol.npy"),
        gm_vol=np.load("data/glm/gm_vol.npy"),
        wm_vol=np.load("data/glm/wm_vol.npy"),
        headsize=np.load("data/glm/headsize.npy"),
        x=np.load("data/glm/x.npy"),
        y=np.load("data/glm/y.npy"),
        z=np.load("data/glm/z.npy"),
    )

    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Age", rtype="Parametric", datainfo="age", preproc="z")
    DC.add_regressor(name="Cog. Perf.", rtype="Parametric", datainfo="cog", preproc="z")
    DC.add_regressor(name="Sex", rtype="Parametric", datainfo="sex", preproc="z")
    DC.add_regressor(name="Brain Vol.", rtype="Parametric", datainfo="brain_vol", preproc="z")
    DC.add_regressor(name="GM Vol.", rtype="Parametric", datainfo="gm_vol", preproc="z")
    DC.add_regressor(name="WM Vol.", rtype="Parametric", datainfo="wm_vol", preproc="z")
    DC.add_regressor(name="Head Size", rtype="Parametric", datainfo="headsize", preproc="z")
    DC.add_regressor(name="x", rtype="Parametric", datainfo="x", preproc="z")
    DC.add_regressor(name="y", rtype="Parametric", datainfo="y", preproc="z")
    DC.add_regressor(name="z", rtype="Parametric", datainfo="z", preproc="z")
    DC.add_regressor(name="Mean", rtype="Constant")

    DC.add_contrast(name="", values=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    DC.add_contrast(name="", values=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    design = DC.design_from_datainfo(data.info)
    design.plot_summary(savepath="plots/glm_design.png", show=False)
    design.plot_leverage(savepath="plots/glm_leverage.png", show=False)
    design.plot_efficiency(savepath="plots/glm_efficiency.png", show=False)

    model = glm.fit.OLSModel(design, data)

    copes = model.copes
    pvalues = [
        do_stats(design, data, model, contrast_idx=0, metric=metric),
        do_stats(design, data, model, contrast_idx=1),
    ]
    mean = model.betas[-1]

    return copes, pvalues, mean

if do_pow:
    target = np.load("data/glm/pow.npy")
    copes, pvalues, mean = fit_glm_and_do_stats(target)
    np.save("data/glm/pow_copes.npy", copes)
    np.save("data/glm/pow_pvalues.npy", pvalues)
    np.save("data/glm/pow_mean.npy", mean)

if do_coh:
    target = np.load("data/glm/coh.npy")
    copes, pvalues, mean = fit_glm_and_do_stats(target)
    np.save("data/glm/coh_copes.npy", copes)
    np.save("data/glm/coh_pvalues.npy", pvalues)
    np.save("data/glm/coh_mean.npy", mean)

if do_mean_coh:
    target = np.load("data/glm/mean_coh.npy")
    copes, pvalues, mean = fit_glm_and_do_stats(target)
    np.save("data/glm/mean_coh_copes.npy", copes)
    np.save("data/glm/mean_coh_pvalues.npy", pvalues)
    np.save("data/glm/mean_coh_mean.npy", mean)

if do_trans_prob:
    target = np.load("data/glm/tp.npy")
    copes, pvalues, mean = fit_glm_and_do_stats(target)
    np.save("data/glm/tp_copes.npy", copes)
    np.save("data/glm/tp_pvalues.npy", pvalues)
    np.save("data/glm/tp_mean.npy", mean)

if do_sum_stats:
    target = np.load("data/glm/sum_stats.npy")
    copes, pvalues, mean = fit_glm_and_do_stats(target, metric="tstats")
    np.save("data/glm/sum_stats_copes.npy", copes)
    np.save("data/glm/sum_stats_pvalues.npy", pvalues)
    np.save("data/glm/sum_stats_mean.npy", mean)
