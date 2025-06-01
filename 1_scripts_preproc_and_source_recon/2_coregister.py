"""Coregisteration.

"""

import numpy as np
from glob import glob
from pathlib import Path
from dask.distributed import Client
from osl_ephys import source_recon, utils

def fix_headshape_points(src_dir, subject):
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])
    nas[2] -= 40
    distances = np.sqrt((nas[0] - hs[0]) ** 2 + (nas[1] - hs[1]) ** 2 + (nas[2] - hs[2]) ** 2)
    keep = distances > 70
    hs = hs[:, keep]
    keep = hs[0] < rpa[0]
    hs = hs[:, keep]
    keep = hs[0] > lpa[0]
    hs = hs[:, keep]
    remove = hs[2] < min(lpa[2], rpa[2]) - 4
    hs = hs[:, ~remove]
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    config = """
        source_recon:
        - extract_fiducials_from_fif: {}
        - fix_headshape_points: {}
        - compute_surfaces:
            include_nose: False
        - coregister:
            use_nose: False
            use_headshape: True
    """

    smri_dir = "cc700/mri/pipeline/release004/BIDS_20190411/anat"
    outdir = "output"

    subjects = []
    smri_files = []
    for directory in sorted(glob(f"{outdir}/*")):
        subject = directory.split("/")[-1].split("_")[1]
        smri_file = f"{smri_dir}/{subject}/anat/{subject}_T1w.nii.gz"
        if Path(smri_file).exists():
            subjects.append(subject)
            smri_files.append(smri_file)

    source_recon.run_src_batch(
        config,
        outdir=outdir,
        subjects=subjects,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_points],
        dask_client=True,
    )
