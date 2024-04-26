# Transient Network Analysis

1. Train an HMM a few times and select the best run to analyse:

    - `1_train_hmm.py`: train an HMM.
    - `2_get_inf_params.py`: get the inferred parameters (state probabilities and covariances) for the best model.
    - `3_calc_multitaper.py`: calculate subject and state-specific multitaper spectra.
    - `4_calc_summary_stats.py`: calculate subject and state-specific summary statistics.

2. Plot networks and summary stats (`group_average_networks`).

3. Do stats to study linear ageing effects (`linear_age_effect`).

4. Do stats to study young vs old groups (`young_vs_old_age_effect`).
