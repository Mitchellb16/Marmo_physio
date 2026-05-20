# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:46:07 2026

@author: mitch
"""

import pandas as pd
import numpy as np
import neurokit2 as nk
from neurokit2.signal import signal_sanitize, signal_rate

def resp_clean(resp_signal, sampling_rate=2000, lowcut=0.5, highcut=3.0):
    clean = nk.signal_filter(
        signal=resp_signal, sampling_rate=sampling_rate,
        lowcut=lowcut, highcut=highcut, method="butterworth", order=3 
    )
    clean = nk.signal_detrend(clean, order=1)
    return clean

def resp_process(resp_signal, sampling_rate=2000, method="khodadad2018", amp_min=0.05):
    print("    -> [RESP] Sanitizing input...")
    resp_signal = signal_sanitize(resp_signal)
    
    print("    -> [RESP] Cleaning signal (Bandpass & Detrend)...")
    rsp_cleaned = resp_clean(resp_signal, sampling_rate=sampling_rate)
    
    print(f"    -> [RESP] Extracting peaks using '{method}' method...")
    peak_signal, info = nk.rsp_peaks(
        rsp_cleaned, sampling_rate=sampling_rate, method=method, amplitude_min=amp_min
    )
    
    info["sampling_rate"] = sampling_rate  

    print("    -> [RESP] Calculating respiratory phase & amplitude...")
    phase = nk.rsp_phase(peak_signal, desired_length=len(resp_signal))
    amplitude = nk.rsp_amplitude(rsp_cleaned, peak_signal)
    symmetry = nk.rsp_symmetry(rsp_cleaned, peak_signal)
# =============================================================================
#     rvt = nk.rsp_rvt(
#         rsp_cleaned,
#         sampling_rate=sampling_rate,
#         silent=True,
#     )
# =============================================================================
    
    print("    -> [RESP] Calculating continuous breathing rate...")
    raw_rate = signal_rate(info["RSP_Troughs"], sampling_rate=sampling_rate, desired_length=len(resp_signal))
    
    # --- ADVANCED OUTLIER CENSORING (Non-Destructive) ---
    print("    -> [RESP] Censoring non-physiological rate jumps...")
    rate_series = pd.Series(raw_rate)
    
    # 1. Build the logical masks
    mask_bounds = (rate_series < 10) | (rate_series > 200)
    
    rate_diff = rate_series.diff().abs()
    is_impossible_jump = rate_diff > 5.0
    
    buffer_samples = int(0.5 * sampling_rate)
    if buffer_samples > 0:
        bad_zones = is_impossible_jump.rolling(window=buffer_samples * 2, center=True, min_periods=1).max() > 0
    else:
        bad_zones = is_impossible_jump
        
    combined_artifact_mask = mask_bounds | bad_zones
    
    # 2. Interpolate for Rate smoothness, but keep Artifacts in an explicit column
    rate_series.loc[combined_artifact_mask] = np.nan
    rate_series = rate_series.interpolate(method='linear', limit_direction='both')
    smooth_rate = rate_series.rolling(window=int(2.0 * sampling_rate), center=True, min_periods=1).median().values
    # ----------------------------------

    print("    -> [RESP] Compiling final DataFrame...")
    signals = pd.DataFrame(
        {
            "RSP_Raw": resp_signal,
            "RSP_Clean": rsp_cleaned,
            "RSP_Amplitude": amplitude,
            "RSP_Rate": smooth_rate, 
            "RSP_Artifact": combined_artifact_mask.astype(int) # Non-destructive tracker
        }
    )
    signals = pd.concat([signals, phase, symmetry, peak_signal], axis=1)

    print("    -> [RESP] Pipeline Complete!")
    return signals, info