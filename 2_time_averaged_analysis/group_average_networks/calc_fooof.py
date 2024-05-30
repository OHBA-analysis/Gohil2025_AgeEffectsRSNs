"""Fit FOOOF to PSD.

"""

import numpy as np
from fooof import FOOOF

f = np.load("../data/f.npy")
psd = np.load("../data/psd.npy").mean(axis=(0,1))

fm = FOOOF()
fm.report(f, psd, [1,45])
