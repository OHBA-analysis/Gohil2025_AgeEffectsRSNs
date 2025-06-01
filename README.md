# Effects of age on resting-state cortical networks

Scripts for reproducing the results in [Gohil et al. (2024)](https://www.biorxiv.org/content/10.1101/2024.09.23.614004v1).

The network data is provided in:

- `0_data_time_averaged_network`: the `target_*.npy` files contain the network features in (subjects, parcels, frequencies) format.
- `0_data_transient_network`: the `target_*.npy` files contain the network features in (subjects, states, parcels) format.

## Prerequisites

To run these scripts you need to install [osl-ephys](https://github.com/OHBA-analysis/osl-ephys) and [osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics). You will also need to install glmtools:
```
pip install glmtools
```

## Getting help

Please open an issue on this repository if you run into errors, need help or spot any typos. Alternatively, you can email chetan.gohil@psych.ox.ac.uk.
