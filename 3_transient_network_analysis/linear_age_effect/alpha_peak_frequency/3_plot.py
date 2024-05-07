"""Plot GLM analysis.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def line(x, m, c):
    return m*x + c

freq_shift = np.load("data/glm_age.npy")
pvalues = np.load("data/glm_age_pvalues.npy")
for fs, p in zip(freq_shift, pvalues):
    print(f"frequency shift = {fs}, p-value = {p}")

age = np.load("data/age.npy")
peak_freq = np.load("data/peak_freq.npy")

for i, pf in enumerate(peak_freq.T):
    remove = np.isnan(pf)
    a = age[~remove]
    pf = pf[~remove]

    popt, pcov = optimize.curve_fit(line, a, pf)
    m, c = popt
    y = line(a, m, c)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(a, pf, s=8, label="Data")
    ax.plot(a, y, c="tab:red", lw=3, label=f"y = {m:.2g}x + {c:.3g}")
    ax.set_xlabel("Age (years)", fontsize=16)
    ax.set_ylabel(r"$\alpha$ peak (Hz)", fontsize=16)
    ax.legend(fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()

    filename = f"plots/freq_vs_age_{i:02d}.png"
    print("Saving", filename)
    plt.savefig(filename)
    plt.close()
