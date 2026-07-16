target += normal_lpdf(ac_scale_raw | 0, 1);

// Non-centred random walk: the increments are standardised (unit scale); the
// process scale enters via ac_noise = ac_scale[proc] * ac_noise_raw (see the
// transformed-parameters block). This avoids the funnel geometry of the centred
// form, which produced divergences / E-BFMI warnings.
target += std_normal_lpdf(ac_noise_raw);

target += normal_lpdf(obs_ac_scale_raw | 0, 1);

target += std_normal_lpdf(obs_ac_noise_raw);
