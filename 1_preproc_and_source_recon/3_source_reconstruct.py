"""Source reconstruction: forward modelling, beamforming and parcellation.

"""

from glob import glob
from dask.distributed import Client
from osl_ephys import source_recon, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    config = """
        source_recon:
        - forward_model:
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 80]
            chantypes: [mag, grad]
            rank: {meg: 60}
            parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    outdir = "output"

    subjects = []
    for directory in sorted(glob(f"{outdir}/*")):
        subject = directory.split("/")[-1].split("_")[1]

    source_recon.run_src_batch(
        config,
        outdir=outdir,
        subjects=subjects,
        dask_client=True,
    )
