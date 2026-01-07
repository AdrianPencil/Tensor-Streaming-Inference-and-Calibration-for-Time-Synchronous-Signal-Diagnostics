# Noise spectroscopy and system identification

SSP includes a minimal noise spectroscopy toolkit:
- PSD estimation (Welch)
- parametric PSD models (white, 1/f, Lorentzian/RTN-like)
- robust parameter fitting in log-space
- empirical transfer function / coherence estimation

## Why include this

Noise spectroscopy provides:
- diagnostics for sensor quality
- parameterized summaries for drift and monitoring
- additional contract signals (band constraints, model residuals)

Implementation lives in `ssp.noise_spectroscopy`.
