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
    pooled_dims=1,
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
    fo = np.load("data/fo.npy")
    n_states = fo.shape[1]

    regressors = {f"fo{i}": fo[:, i] for i in range(n_states)}

    data = glm.data.TrialGLMData(data=target, **regressors)

    DC = glm.design.DesignConfig()
    for i in range(n_states):
        DC.add_regressor(name=f"FO{i + 1}", rtype="Parametric", datainfo=f"fo{i}", preproc="z")
    DC.add_regressor(name="Mean", rtype="Constant")

    DC.add_simple_contrasts()

    design = DC.design_from_datainfo(data.info)
    design.plot_summary(savepath="plots/glm_design.png", show=False)
    design.plot_leverage(savepath="plots/glm_leverage.png", show=False)
    design.plot_efficiency(savepath="plots/glm_efficiency.png", show=False)

    model = glm.fit.OLSModel(design, data)

    copes = model.copes[:-1]
    pvalues = np.array([
        do_stats(design, data, model, contrast_idx=i, metric=metric)
        for i in range(n_states)
    ])

    return copes, pvalues

target = np.load("data/static_pow.npy")
copes, pvalues = fit_glm_and_do_stats(target)
np.save("data/glm_copes.npy", copes)
np.save("data/glm_pvalues.npy", pvalues)
