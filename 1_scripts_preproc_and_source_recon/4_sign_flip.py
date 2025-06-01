"""Dipole sign flipping.

"""

from glob import glob
from dask.distributed import Client
from osl_ephys import source_recon, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    outdir = "output"

    subjects = []
    for path in sorted(glob(f"{outdir}/*/parc/lcmv-parc-raw.fif")):
        subject = path.split("/")[-3]
        subjects.append(subject)

    template = source_recon.find_template_subject(
        outdir,
        subjects,
        n_embeddings=15,
        standardize=True,
    )

    config = f"""
        source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: True
            n_init: 5
            n_iter: 5000
            max_flips: 20
    """

    source_recon.run_src_batch(
        config,
        outdir=outdir,
        subjects=subjects,
        dask_client=True,
    )
