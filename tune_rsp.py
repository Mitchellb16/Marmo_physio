#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:43:40 2026

@author: mitchell
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
import neurokit2 as nk

# Setup paths to use your existing functions
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from src.ingestion import load_physio_acq
from src.resp_processor import resp_clean
from neurokit2.signal import signal_sanitize

def compute_rsp_peaks(cleaned_signal, fs, method, amp_min=0.3, dist=0.8, prom=0.5):
    """Wraps NeuroKit's peak detector to catch errors if parameters get too aggressive."""
    try:
        if method == "khodadad":
            # Khodadad uses amplitude_min
            _, info = nk.rsp_peaks(cleaned_signal, sampling_rate=fs, method=method, amplitude_min=amp_min)
        elif method == "scipy":
            # SciPy uses peak_distance and peak_prominence
            _, info = nk.rsp_peaks(cleaned_signal, sampling_rate=fs, method=method, peak_distance=dist, peak_prominence=prom)
        
        peaks = info.get("RSP_Peaks", [])
        troughs = info.get("RSP_Troughs", [])
        return peaks, troughs
    except Exception:
        # If the math breaks (e.g., slider set too extreme), return nothing
        return [], []

if __name__ == "__main__":
    print("Loading Poppy's Respiration data...")
    TEST_FILE = PROJECT_ROOT / 'raw_data' / 'test' / '20260416_Freddy_silence(ecg_resp).acq'
    
    # Grab 20 seconds of data to see several full breath cycles
    df, fs = load_physio_acq(str(TEST_FILE))
    start_idx = int(fs * 900) # 5 minutes in
    chunk_time = 100
    end_idx = start_idx + int(fs * chunk_time) # 20 seconds long
    
    raw_snippet = df['Thermistor'].iloc[start_idx:end_idx].values
    
    sanitized = signal_sanitize(raw_snippet)
    # Use your custom bandpass to prep the signal exactly how the pipeline will
    cleaned_signal = resp_clean(sanitized, sampling_rate=fs)
    time_axis = np.linspace(0, chunk_time, len(cleaned_signal))

    # --- Setup the Matplotlib UI ---
    fig, (ax_kho, ax_sci) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    plt.subplots_adjust(bottom=0.3) 

    # 1. Khodadad Plot (Top)
    ax_kho.plot(time_axis, cleaned_signal, lw=1.5, color='black', alpha=0.7)
    kho_peak_scatter = ax_kho.scatter([], [], color='red', marker='^', s=60, label='Peaks (Inhale)', zorder=5)
    kho_trough_scatter = ax_kho.scatter([], [], color='blue', marker='v', s=60, label='Troughs (Exhale)', zorder=5)
    ax_kho.set_title("Method 1: Khodadad (Amplitude Thresholding)", fontweight='bold')
    ax_kho.legend(loc='upper right')
    ax_kho.grid(True, linestyle='--', alpha=0.5)

    # 2. SciPy Plot (Bottom)
    ax_sci.plot(time_axis, cleaned_signal, lw=1.5, color='black', alpha=0.7)
    sci_peak_scatter = ax_sci.scatter([], [], color='red', marker='^', s=60, zorder=5)
    sci_trough_scatter = ax_sci.scatter([], [], color='blue', marker='v', s=60, zorder=5)
    ax_sci.set_title("Method 2: SciPy (Distance & Prominence)", fontweight='bold')
    ax_sci.grid(True, linestyle='--', alpha=0.5)
    ax_sci.set_xlabel('Time (seconds)')

    # --- Sliders ---
    axcolor = 'lightgoldenrodyellow'
    ax_amp = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_dist = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_prom = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

    # Default NeuroKit values
    sl_amp = Slider(ax_amp, 'Kho: Amp Min', 0.01, 1.5, valinit=0.3)
    sl_dist = Slider(ax_dist, 'Sci: Peak Dist (s)', 0.1, 2.0, valinit=0.8)
    sl_prom = Slider(ax_prom, 'Sci: Prominence', 0.05, 1.5, valinit=0.5)

    def update(val):
        amp = sl_amp.val
        dist = sl_dist.val
        prom = sl_prom.val
        
        # Calculate Khodadad
        k_peaks, k_troughs = compute_rsp_peaks(cleaned_signal, fs, "khodadad", amp_min=amp)
        if len(k_peaks) > 0:
            kho_peak_scatter.set_offsets(np.c_[time_axis[k_peaks], cleaned_signal[k_peaks]])
        else:
            kho_peak_scatter.set_offsets(np.empty((0, 2)))
            
        if len(k_troughs) > 0:
            kho_trough_scatter.set_offsets(np.c_[time_axis[k_troughs], cleaned_signal[k_troughs]])
        else:
            kho_trough_scatter.set_offsets(np.empty((0, 2)))

        # Calculate SciPy
        s_peaks, s_troughs = compute_rsp_peaks(cleaned_signal, fs, "scipy", dist=dist, prom=prom)
        if len(s_peaks) > 0:
            sci_peak_scatter.set_offsets(np.c_[time_axis[s_peaks], cleaned_signal[s_peaks]])
        else:
            sci_peak_scatter.set_offsets(np.empty((0, 2)))
            
        if len(s_troughs) > 0:
            sci_trough_scatter.set_offsets(np.c_[time_axis[s_troughs], cleaned_signal[s_troughs]])
        else:
            sci_trough_scatter.set_offsets(np.empty((0, 2)))
            
        fig.canvas.draw_idle()

    # Link sliders
    sl_amp.on_changed(update)
    sl_dist.on_changed(update)
    sl_prom.on_changed(update)

    # Init
    update(0)
    plt.show()