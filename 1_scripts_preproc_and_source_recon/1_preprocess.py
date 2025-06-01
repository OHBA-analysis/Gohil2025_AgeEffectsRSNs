"""Sensor-level preprocessing.

"""

import pathlib
from glob import glob
from dask.distributed import Client
from osl_ephys import preprocessing, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    config = """
        preproc:
        - crop: {tmin: 30}
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 88 100, notch_widths: 2}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: grad, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: grad, mode: diff, significance_level: 0.1}
        - bad_channels: {picks: mag, significance_level: 0.1}
        - bad_channels: {picks: grad, significance_level: 0.1}
        - ica_raw: {picks: meg, n_components: 64}
        - ica_autoreject: {picks: meg, ecgmethod: correlation, eogthreshold: auto}
        - interpolate_bads: {}
    """

    rawdir = "cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002"
    outdir = "output"

    subjects = []
    inputs = []
    for subject in sorted(glob(f"{rawdir}/sub-*")):
        subject = pathlib.Path(subject).stem
        subjects.append(subject)
        inputs.append(f"{rawdir}/{subject}/mf2pt2_{subject}_ses-rest_task-rest_meg.fif")

    preprocessing.run_proc_batch(
        config,
        inputs,
        outdir=outdir,
        overwrite=True,
        dask_client=True,
    )
