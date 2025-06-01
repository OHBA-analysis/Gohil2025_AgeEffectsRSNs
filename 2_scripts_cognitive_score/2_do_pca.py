"""Do PCA to reduce cognitive scores to a single metric.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
data = pd.read_csv("data/cognitive_metrics.csv", index_col=0)
ids = data["ID"]
data = data.drop(columns=["ID", "Sex (1=male, 2=female)"])
order = [0, 1, 8, 5, 10, 2, 3, 7, 11, 13, 4, 9, 6, 12]
data = data[data.columns[order]]

# For MltTs and MRSp lower values indicate a better score so they are
# positively correlated with age.
# 
# In https://www.nature.com/articles/ncomms6658, MltTs is multiplied by
# -1, we do the same here
data["MltTs"] *= -1
# Similar, the motor response speed MRSp is really a response time,
# so we'll multiply it by -1 to obtain a negative correlation with age
data["MRSp"] *= -1

# Remove very non-gaussian metrics
reduced_data = data.drop(columns=["ProV", "FacReg", "StW"])

# Remove outliers
data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
reduced_data = reduced_data[(np.abs(stats.zscore(reduced_data)) < 3).all(axis=1)]

# Original data
d = reduced_data.drop(columns="Age")
X = d.values
print("X.shape =", X.shape)

# Standardise
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=1)
X = pca.fit_transform(X)
W = pca.components_

print("X.shape =", X.shape)
print("W.shape =", W.shape)
print("explained variance:", pca.explained_variance_ratio_)

# Flip the first PC to indicate cognitive decline with age
X[:, 0] *= -1

# Plot correlation with age
age = reduced_data["Age"].values
fig, ax = plt.subplots()
ax.scatter(age, X[:, 0])
ax.set_xlabel("Age (years)", fontsize=16)
ax.set_ylabel("Cognitive Score\n(First PCA component)", fontsize=16)
ax.tick_params(axis="both", labelsize=15)
plt.tight_layout()
plt.savefig("plots/pca_vs_age.png")
plt.close()

fig, ax = plt.subplots()
ax.bar(list(d.keys()), -W[0])
ax.set_xlabel("Cognitive Task", fontsize=16)
ax.set_ylabel("Loading", fontsize=16)
ax.tick_params(axis="both", labelsize=15)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("plots/pca_loadings.png")
plt.close()

# Save PCA'ed cognitive metrics
d = pd.DataFrame({f"Component {i}": X[:, i] for i in range(X.shape[-1])})
d["Age"] = reduced_data["Age"]
d["ID"] = ids
d.to_csv("data/cognitive_metrics_pca.csv")
