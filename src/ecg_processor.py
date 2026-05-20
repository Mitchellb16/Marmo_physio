# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:12:47 2026

@author: mitch
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.ndimage import binary_dilation
from neurokit2.ecg.ecg_delineate import ecg_delineate
from neurokit2.ecg.ecg_phase import ecg_phase
from neurokit2.ecg.ecg_peaks import ecg_peaks
from neurokit2.ecg.ecg_quality import ecg_quality
from neurokit2.signal import signal_sanitize, signal_rate

def ecg_clean(ecg_signal, sampling_rate=2000, lowcut=1.0, highcut=450.0):
    clean = nk.signal_filter(
        signal=ecg_signal, sampling_rate=sampling_rate,
        lowcut=lowcut, highcut=highcut, method="butterworth", order=3
    )
    clean = nk.signal_filter(signal=clean, sampling_rate=sampling_rate, method="powerline")
    return clean
    
def ecg_process(ecg_signal, sampling_rate=2000, **kwargs):
    print("    -> [ECG] Sanitizing and checking lead polarity...")
    ecg_signal = signal_sanitize(ecg_signal) 
    
    # Standard inversion running on the cropped signal
    ecg_inverted, _ = nk.ecg_invert(ecg_signal, sampling_rate=sampling_rate) 
    
    nyquist = sampling_rate / 2
    safe_highcut = min(450.0, nyquist - 1) 
    
    print("    -> [ECG] Filtering signal (Bandpass & Powerline)...")
    ecg_cleaned = ecg_clean(ecg_inverted, sampling_rate=sampling_rate, lowcut=2.5, highcut=safe_highcut) 
    
    print("    -> [ECG] Detecting R-peaks (Method: Tuned NeuroKit)...")
    instant_peaks, info = ecg_peaks(
        ecg_cleaned=ecg_cleaned, 
        sampling_rate=sampling_rate, 
        method='neurokit',
        smoothwindow=0.03,
        avgwindow=0.250,
        gradthreshweight=1.25,
        minlenweight=0.4,
        mindelay=0.02
    )
    
    print("    -> [ECG] Calculating continuous heart rate...")
    raw_rate = signal_rate(info, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned))

    print("    -> [ECG] Assessing signal quality (Template Match)...")
    quality = ecg_quality(ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate, method='templatematch')

    # --- ADVANCED OUTLIER CENSORING ---
    print("    -> [ECG] Censoring artifacts (Quality & Derivative Gating)...")
    rate_series = pd.Series(raw_rate)
    
    # 1. Build the logical masks
    quality_threshold = 0.6
    mask_quality = (quality < quality_threshold)
    mask_bounds = (rate_series < 150) | (rate_series > 700)
    
    # Max change of 100 BPM per 1 second = 0.05 BPM per 1/2000th second
    max_derivative = 100 / sampling_rate
    mask_derivative = rate_series.diff().abs() > max_derivative
    
    # 2. Combine and Dilate
    combined_artifact_mask = mask_quality | mask_bounds | mask_derivative
    
    # Dilate by 0.2 seconds in both directions to swallow the "tent" edges
    dilation_iterations = int(0.2 * sampling_rate)
    padded_mask = binary_dilation(combined_artifact_mask, iterations=dilation_iterations)
    
    # 3. Apply mask, interpolate, and smooth
    rate_series.loc[padded_mask] = np.nan  # Swapped pd.NA for np.nan for safe numeric interpolation
    rate_series = rate_series.interpolate(method='linear', limit_direction='both')
    smooth_rate = rate_series.rolling(window=int(1.0 * sampling_rate), center=True, min_periods=1).median().values
    # ----------------------------------

    signals = pd.DataFrame({
        "ECG_Raw": ecg_signal, 
        "ECG_Clean": ecg_cleaned,
        "ECG_Rate": smooth_rate, 
        "ECG_Quality": quality,
        "ECG_Artifact": padded_mask.astype(int)  # Explicit non-destructive artifact tracker
    })

    print("    -> [ECG] Delineating QRS complex...")
    delineate_signal, delineate_info = ecg_delineate(
        ecg_cleaned=ecg_cleaned, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )
    info.update(delineate_info)  

    print("    -> [ECG] Determining cardiac phases...")
    cardiac_phase = ecg_phase(
        ecg_cleaned=ecg_cleaned, rpeaks=info["ECG_R_Peaks"], delineate_info=delineate_info
    )

    print("    -> [ECG] Compiling final DataFrame...")
    signals = pd.concat([signals, instant_peaks, delineate_signal, cardiac_phase], axis=1)

    return signals, info