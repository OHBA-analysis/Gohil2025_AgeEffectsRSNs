"""Gather cognitive scores.

"""

import numpy as np
import pandas as pd
from scipy import io

# Load cognitive data
cognitive_data = io.loadmat("data/CogDatAll.mat", simplify_cells=True)["CogDatAll"]

# Columns:
#  1) Sex (1=male, 2=female)   NOTE: this is different to participants.tsv (which is 1=female, 2=male) !!
#  2) Age
#  3) FldIn   : fluid intelligence           f   ex
#  4) FacReg  : face recognition             f   emo
#  5) EmoRec  : emotional recognition        f   emo
#  6) MltTs   : multitask                    f   ex     (negatively correlated with others - indicates reaction times)
#  7) PicName : picture naming               f   lang
#  8) ProV    : proverb comprehension        c   lang
#  9) MRSp    : motor speed                  f   mot    (negatively correlated with others - indicates reaction times)
# 10) MRCv    : motor speed variance         f   mot
# 11) SntRec  : sentence comprehension       c/f lang
# 12) VSTM    : visual short-term memory     f   mem
# 13) StrRec  : story recall                 f   mem
# 14) StW     : spot the word                c   lang
# 15) VrbFl   : verbal fluency               f   lang
# 16) ID

# Create a dataframe
cognitive_data = pd.DataFrame({
    "ID": cognitive_data[:, 15].astype(int),
    "Sex (1=male, 2=female)": cognitive_data[:, 0],
    "Age": cognitive_data[:, 1],
    "FldIn": cognitive_data[:, 2],
    "FacReg": cognitive_data[:, 3],
    "EmoRec": cognitive_data[:, 4],
    "MltTs": cognitive_data[:, 5],
    "PicName": cognitive_data[:, 6],
    "ProV": cognitive_data[:, 7],
    "MRSp": cognitive_data[:, 8],
    "MRCv": cognitive_data[:, 9],
    "SntRec": cognitive_data[:, 10],
    "VSTM": cognitive_data[:, 11],
    "StrRec": cognitive_data[:, 12],
    "StW": cognitive_data[:, 13],
    "VrbFl": cognitive_data[:, 14],
})

# Prepend "CC" to the IDs
cognitive_data["ID"] = "CC" + cognitive_data["ID"].astype(str)

# Save
cognitive_data.to_csv("data/cognitive_metrics.csv")
