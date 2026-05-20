# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:20:31 2026

@author: mitch
"""

# Written with help from Gemini AI (Gemini 3.1 Pro)
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
import neurokit2 as nk

def pupil_process(raw_pupil, sampling_rate, mad_multiplier=4.0, buffer_ms=100):
    """
    Refined Pupil Processor using MAD.
    Outputs 'honest' data (NaNs for blinks) plus Session-Normalized Metrics (Z-Score & Robust %Max).
    """
    pupil_series = pd.Series(raw_pupil).copy()
    
    # low pass filter to remove artifacts
    pupil_series = pd.Series(nk.signal_filter(
        signal=pupil_series, sampling_rate=sampling_rate,highcut=10, method="butterworth", order=3
    ))
    
    # 1. Calculate Dynamic MAD Threshold
    median_val = pupil_series.median()
    mad_val = np.median(np.abs(pupil_series - median_val))
    threshold = median_val - (mad_multiplier * mad_val)
    
    # 2. Detect and Pad Blinks
    is_blink = (pupil_series < threshold) | (pupil_series <= 0)
    n_iterations = int((buffer_ms / 1000) * sampling_rate)
    padded_mask = binary_dilation(is_blink, iterations=n_iterations)
    
    # 3. Temporary Interpolation (For Math Purposes Only)
    temp_continuous = pupil_series.copy()
    temp_continuous[padded_mask] = np.nan
    temp_continuous = temp_continuous.interpolate(method='linear', limit_direction='both')
    
    # 4. Smoothing (100ms rolling median)
    smooth_window = int(0.1 * sampling_rate)
    if smooth_window % 2 == 0: smooth_window += 1 
    smooth_pupil = temp_continuous.rolling(window=smooth_window, center=True).median().bfill().ffill()
    
    # 5. RE-CENSOR THE DATA (The "Honest Data" Step)
    final_clean_pupil = smooth_pupil.copy()
    final_clean_pupil[padded_mask] = np.nan
    
    # 6. Build Pipeline-Standard DataFrame
    signals = pd.DataFrame({
        'Pupil_Raw': pupil_series,
        'Pupil_Clean': final_clean_pupil, 
        'Pupil_Artifact': (padded_mask).astype(int) 
    })
    
    # --- 7. SESSION NORMALIZATIONS ---
    
    # A. Z-Scoring (For ML / Cross-Correlation)
    # (x - mean) / standard_deviation
    signals['Pupil_Z'] = (signals['Pupil_Clean'] - signals['Pupil_Clean'].mean()) / signals['Pupil_Clean'].std()
    
    # B. Robust %Max (For Interpretable Feature Extraction)
    # (x / 99th_percentile) * 100
    pupil_99th = signals['Pupil_Clean'].quantile(0.99)
    signals['Pupil_PctMax'] = (signals['Pupil_Clean'] / pupil_99th) * 100
    
    # Info Dictionary for QC Logging
    info = {
        'mad_threshold_used': threshold,
        'percent_lost': (padded_mask.sum() / len(padded_mask)) * 100,
        'session_99th_percentile': pupil_99th
    }
    
    return signals, info