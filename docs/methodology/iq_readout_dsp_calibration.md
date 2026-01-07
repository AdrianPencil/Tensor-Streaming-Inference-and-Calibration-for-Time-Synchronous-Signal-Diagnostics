# IQ readout DSP and calibration

This module family handles complex baseband IQ streams:
- demodulation / mixing
- DC removal
- imbalance correction (gain/phase)
- decimation and matched filtering
- feature extraction for scoring

## Why include an IQ stack

It provides a realistic signal-processing measurement chain that:
- produces features used by P1 scoring
- can be validated with invariants and synthetic signals
- naturally integrates with replay and reproducibility

Implementation lives in `ssp.iq_readout`.
