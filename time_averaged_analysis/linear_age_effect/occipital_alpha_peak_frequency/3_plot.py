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

popt, pcov = optimize.curve_fit(line, age, peak_freq)
m, c = popt
y = line(age, m, c)

fig, ax = plt.subplots()
ax.scatter(age, peak_freq, s=8, label="Data")
ax.plot(age, y, c="tab:red", lw=3, label=f"y = {m:.2g}x + {c:.3g}")
ax.set_xlabel("Age (years)", fontsize=16)
ax.set_ylabel(r"Occipital $\alpha$ peak (Hz)", fontsize=16)
ax.legend(fontsize=16)
ax.tick_params(axis="both", labelsize=14)
plt.tight_layout()

filename = "plots/freq_vs_age.png"
print("Saving", filename)
plt.savefig(filename)
plt.close()
