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
        category_list=np.load("data/category_list.npy"),
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
    DC.add_regressor(name="Old", rtype="Categorical", codes=1)
    DC.add_regressor(name="Young", rtype="Categorical", codes=2)
    DC.add_regressor(name="Sex", rtype="Parametric", datainfo="sex", preproc="z")
    DC.add_regressor(name="Brain Vol.", rtype="Parametric", datainfo="brain_vol", preproc="z")
    DC.add_regressor(name="GM Vol.", rtype="Parametric", datainfo="gm_vol", preproc="z")
    DC.add_regressor(name="WM Vol.", rtype="Parametric", datainfo="wm_vol", preproc="z")
    DC.add_regressor(name="Hippo. Vol.", rtype="Parametric", datainfo="hip_vol", preproc="z")
    DC.add_regressor(name="Head Size", rtype="Parametric", datainfo="headsize", preproc="z")
    DC.add_regressor(name="x", rtype="Parametric", datainfo="x", preproc="z")
    DC.add_regressor(name="y", rtype="Parametric", datainfo="y", preproc="z")
    DC.add_regressor(name="z", rtype="Parametric", datainfo="z", preproc="z")

    DC.add_contrast(name="", values=[1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    DC.add_contrast(name="", values=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    design = DC.design_from_datainfo(data.info)
    design.plot_summary(savepath="plots/glm_design.png", show=False)
    design.plot_leverage(savepath="plots/glm_leverage.png", show=False)
    design.plot_efficiency(savepath="plots/glm_efficiency.png", show=False)

    model = glm.fit.OLSModel(design, data)

    young = model.copes[1]
    old_minus_young = model.copes[0]
    pvalues = do_stats(design, data, model, contrast_idx=0, metric=metric)
    return young, old_minus_young, pvalues

if do_pow:
    target = np.load("data/pow.npy")
    young, old_minus_young, pvalues = fit_glm_and_do_stats(target)
    np.save("data/glm_pow_young.npy", young)
    np.save("data/glm_pow_old_minus_young.npy", old_minus_young)
    np.save("data/glm_pow_pvalues.npy", pvalues)

if do_coh:
    target = np.load("data/coh.npy")
    young, old_minus_young, pvalues = fit_glm_and_do_stats(target)
    np.save("data/glm_coh_young.npy", young)
    np.save("data/glm_coh_old_minus_young.npy", old_minus_young)
    np.save("data/glm_coh_pvalues.npy", pvalues)

if do_mean_coh:
    target = np.load("data/mean_coh.npy")
    young, old_minus_young, pvalues = fit_glm_and_do_stats(target)
    np.save("data/glm_mean_coh_young.npy", young)
    np.save("data/glm_mean_coh_old_minus_young.npy", old_minus_young)
    np.save("data/glm_mean_coh_pvalues.npy", pvalues)

if do_trans_prob:
    target = np.load("data/trans_prob.npy")
    young, old_minus_young, pvalues = fit_glm_and_do_stats(target)
    np.save("data/glm_trans_prob_young.npy", young)
    np.save("data/glm_trans_prob_old_minus_young.npy", old_minus_young)
    np.save("data/glm_trans_prob_pvalues.npy", pvalues)

if do_sum_stats:
    target = np.load("data/sum_stats.npy")
    young, old_minus_young, pvalues = fit_glm_and_do_stats(target, metric="tstats")
    np.save("data/glm_sum_stats_young.npy", young)
    np.save("data/glm_sum_stats_old_minus_young.npy", old_minus_young)
    np.save("data/glm_sum_stats_pvalues.npy", pvalues)
