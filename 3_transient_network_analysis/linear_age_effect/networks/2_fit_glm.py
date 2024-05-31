"""Fit a GLM and do stats.

"""

import numpy as np
import glmtools as glm
from scipy import stats

do_pow = False
do_coh = False
do_mean_coh = False
do_trans_prob = False
do_sum_stats = False
do_sr = True

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

def fit_glm_and_do_stats(target, metric="copes", pooled_dims=(1,2)):
    data = glm.data.TrialGLMData(
        data=target,
        age=np.load("data/age.npy"),
        sex=np.load("data/sex.npy"),
        brain_vol=np.load("data/brain_vol.npy"),
        gm_vol=np.load("data/gm_vol.npy"),
        wm_vol=np.load("data/wm_vol.npy"),
        hip_vol=np.load("data/hip_vol.npy"),
        headsize=np.load("data/headsize.npy"),
        x=np.load("data/x.npy"),
        y=np.load("data/y.npy"),
        z=np.load("data/z.npy"),
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
    pvalues = do_stats(
        design,
        data,
        model,
        contrast_idx=0,
        metric=metric,
        pooled_dims=pooled_dims,
    )
    return mean, age, pvalues

if do_pow:
    target = np.load("data/pow.npy")
    mean, age, pvalues = fit_glm_and_do_stats(target)
    np.save("data/glm_pow_mean.npy", mean)
    np.save("data/glm_pow_age.npy", age)
    np.save("data/glm_pow_age_pvalues.npy", pvalues)

if do_coh:
    target = np.load("data/coh.npy")
    mean, age, pvalues = fit_glm_and_do_stats(target)
    np.save("data/glm_coh_mean.npy", mean)
    np.save("data/glm_coh_age.npy", age)
    np.save("data/glm_coh_age_pvalues.npy", pvalues)

if do_mean_coh:
    target = np.load("data/mean_coh.npy")
    mean, age, pvalues = fit_glm_and_do_stats(target)
    np.save("data/glm_mean_coh_mean.npy", mean)
    np.save("data/glm_mean_coh_age.npy", age)
    np.save("data/glm_mean_coh_age_pvalues.npy", pvalues)

if do_trans_prob:
    target = np.load("data/trans_prob.npy")
    mean, age, pvalues = fit_glm_and_do_stats(target)
    np.save("data/glm_trans_prob_mean.npy", mean)
    np.save("data/glm_trans_prob_age.npy", age)
    np.save("data/glm_trans_prob_age_pvalues.npy", pvalues)

if do_sum_stats:
    target = np.load("data/sum_stats.npy")
    target = target[:, :3]  # keep fo, lt, intv
    mean, age, pvalues = fit_glm_and_do_stats(target, metric="tstats")
    np.save("data/glm_sum_stats_mean.npy", mean)
    np.save("data/glm_sum_stats_age.npy", age)
    np.save("data/glm_sum_stats_age_pvalues.npy", pvalues)

if do_sr:
    target = np.load("data/sum_stats.npy")
    target = np.sum(target[:, -1], axis=-1)
    mean, age, pvalues = fit_glm_and_do_stats(target, metric="tstats", pooled_dims=())
    np.save("data/glm_sr_mean.npy", mean)
    np.save("data/glm_sr_age.npy", age)
    np.save("data/glm_sr_age_pvalues.npy", pvalues)
