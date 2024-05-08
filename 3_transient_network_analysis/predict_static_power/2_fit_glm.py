"""Fit a GLM and do stats.

"""

import numpy as np
import glmtools as glm
from scipy import stats

def do_stats(design, data, model, contrast_idx, metric="copes"):
    perm = glm.permutations.MaxStatPermutation(
        design=design,
        data=data,
        contrast_idx=contrast_idx,
        nperms=1000,
        metric=metric,
        tail=0,  # two-tailed t-test
        pooled_dims=(1,2),  # pool over channels and frequencies
        nprocesses=16,
    )
    if metric == "tstats":
        tstats = abs(model.tstats[contrast_idx])
        percentiles = stats.percentileofscore(perm.nulls, tstats)
    elif metric == "copes":
        copes = abs(model.copes[contrast_idx])
        percentiles = stats.percentileofscore(perm.nulls, copes)
    return 1 - percentiles / 100

def fit_glm_and_do_stats(target, state, metric="copes"):
    fo = np.load("data/fo.npy")[:, state]
    lt = np.load("data/lt.npy")[:, state]
    intv = np.load("data/intv.npy")[:, state]
    sr = np.load("data/sr.npy")[:, state]

    data = glm.data.TrialGLMData(
        data=target,
        fo=fo,
        lt=lt,
        intv=intv,
        sr=sr,
    )

    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Mean", rtype="Constant")
    DC.add_regressor(name="FO", rtype="Parametric", datainfo="fo", preproc="z")
    DC.add_regressor(name="LT", rtype="Parametric", datainfo="lt", preproc="z")
    DC.add_regressor(name="INTV", rtype="Parametric", datainfo="intv", preproc="z")
    DC.add_regressor(name="SR", rtype="Parametric", datainfo="sr", preproc="z")

    DC.add_simple_contrasts()

    design = DC.design_from_datainfo(data.info)
    design.plot_summary(savepath="plots/glm_design.png", show=False)
    design.plot_leverage(savepath="plots/glm_leverage.png", show=False)
    design.plot_efficiency(savepath="plots/glm_efficiency.png", show=False)

    exit()

    model = glm.fit.OLSModel(design, data)

    copes = model.copes
    #pvalues = [do_stats(design, data, model, contrast_idx=0, metric=metric) for i in range(len(copes))]
    return copes#, pvalues

target = np.load("data/pow.npy")
fit_glm_and_do_stats(target, state=0)
