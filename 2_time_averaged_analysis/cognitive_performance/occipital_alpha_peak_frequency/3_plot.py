"""Plot results.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import optimize

def line(x, m, c):
    return m*x + c

# Load data
cog = np.load("data/cog.npy")
peak_freq = np.load("data/peak_freq.npy")
age = np.load("data/age.npy")

# Check the GLM stats
cope = np.load("data/glm_copes.npy")
pvalue = np.load("data/glm_pvalues.npy")
print(f"cope={cope}, pvalue={pvalue}")

# Remove outliers
mu = np.mean(peak_freq)
sigma = np.std(peak_freq)
keep = np.logical_and(mu - 3 * sigma < peak_freq, peak_freq < mu + 3 * sigma)
cog = cog[keep]
peak_freq = peak_freq[keep]
age = age[keep]

# Fit straight line
popt, pcov = optimize.curve_fit(line, peak_freq, cog)
m, c = popt
y = line(peak_freq, m, c)

# Plot
cmap = plt.get_cmap()

fig, ax = plt.subplots(figsize=(5,3))

s = ax.scatter(peak_freq, cog, s=8, c=age)
ax.plot(peak_freq, y, c="tab:red", lw=3, label=f"y = {m:.2g}x + {c:.3g}")

ax.set_ylim(-10, None)
ax.set_xlabel(r"Occipital $\alpha$ peak (Hz)", fontsize=16)
ax.set_ylabel("First PCA Component", fontsize=16)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=16)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(s, cax=cax, orientation="vertical")
cax.set_ylabel("Age (years)", fontsize=13)
cax.tick_params(axis="both", labelsize=12)

plt.tight_layout()
plt.savefig("plots/cog_vs_peak_freq.png")

