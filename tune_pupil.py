#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:27:04 2026

@author: mitchell
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
from scipy.ndimage import binary_dilation

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from src.ingestion import load_pupil_edf

def process_pupil_chunk(raw_pupil, fs, threshold, pad_ms, smooth_ms):
    """
    Applies the Detect -> Pad -> Interpolate -> Smooth logic using fast binary dilation.
    """
    # THE FIX: Add .copy() to prevent permanently overwriting the raw data in memory
    pupil_series = pd.Series(raw_pupil).copy()
    
    # 1. Detect Blinks
    is_blink = (pupil_series < threshold).values 
    
    # 2. Pad the blink mask
    pad_samples = int((pad_ms / 1000.0) * fs)
    
    if pad_samples > 0:
        structure = np.ones(pad_samples * 2 + 1, dtype=bool)
        padded_mask = binary_dilation(is_blink, structure=structure)
    else:
        padded_mask = is_blink
        
    # Apply the mask (turn bad data to NaN)
    pupil_series.loc[padded_mask] = np.nan
    
    # 3. Interpolate (Cubic spline is best for biological curves)
    try:
        interpolated = pupil_series.interpolate(method='cubicspline', limit_direction='both')
    except ValueError:
        interpolated = pupil_series.interpolate(method='linear', limit_direction='both')
        
    # 4. Smooth
    smooth_samples = max(1, int((smooth_ms / 1000.0) * fs))
    smoothed = interpolated.rolling(window=smooth_samples, center=True, min_periods=1).mean()
    
    return smoothed.values, padded_mask

if __name__ == "__main__":
    print("Loading Poppy's EyeLink data...")
    EDF_FILE = PROJECT_ROOT / 'Task' / 'results' / '2026_04_17_Poppy' / '2026_04_17_Poppy.EDF' 
    
    if not EDF_FILE.exists():
        RESULTS_DIR = PROJECT_ROOT / 'Task' / 'results'
        edf_files = list(RESULTS_DIR.rglob('*.EDF')) + list(RESULTS_DIR.rglob('*.edf'))
        if edf_files:
            EDF_FILE = edf_files[0]
            print(f"Using found EDF: {EDF_FILE.name}")
        else:
            print("No EDF files found. Please check paths.")
            sys.exit()

    df, fs = load_pupil_edf(str(EDF_FILE))
    
    # Grab a 15-second chunk starting 4 minutes in
    chunk_len = 30.0
    start_idx = int(fs * 240) 
    end_idx = start_idx + int(fs * chunk_len) 
    
    raw_snippet = df['Pupil_Eye_0'].iloc[start_idx:end_idx].values
    time_axis = np.linspace(0, chunk_len, len(raw_snippet))

    # --- Setup the Matplotlib UI ---
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.35) 

    # Plot Raw Data
    ax.plot(time_axis, raw_snippet, color='lightgray', lw=2, label='Raw EyeLink Data')
    
    # Plot Cleaned Data
    clean_line, = ax.plot(time_axis, raw_snippet, color='dodgerblue', lw=2, label='Cleaned & Interpolated')
    
    # THE FIX: Create a standard line object for the blink indicator
    # We will position it at 110% of the max raw value so it floats above the data
    indicator_height = np.nanmax(raw_snippet) * 1.1 
    blink_line, = ax.plot(time_axis, np.full_like(time_axis, np.nan), color='red', lw=4, label='Blink Mask')

    ax.set_title("Pupillometry Artifact Rejection", fontweight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pupil Area/Diameter (a.u.)")
    ax.set_ylim(np.nanmin(raw_snippet)*0.5, np.nanmax(raw_snippet)*1.2)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- Sliders ---
    axcolor = 'lightgoldenrodyellow'
    ax_thresh = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_pad = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_smooth = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)

    max_val = np.nanmax(raw_snippet)
    sl_thresh = Slider(ax_thresh, 'Blink Threshold', 0, max_val, valinit=max_val*0.2)
    sl_pad = Slider(ax_pad, 'Pad Window (ms)', 0, 300, valinit=50)
    sl_smooth = Slider(ax_smooth, 'Smooth Window (ms)', 1, 500, valinit=50)

    def update(val):
        thresh = sl_thresh.val
        pad = sl_pad.val
        smooth = sl_smooth.val
        
        cleaned, mask = process_pupil_chunk(raw_snippet, fs, thresh, pad, smooth)
        
        # Update the cleaned blue line
        clean_line.set_ydata(cleaned)
        
        # THE FIX: Update the red indicator line by setting non-blink areas to NaN
        # This is incredibly fast and doesn't require 'global'
        indicator_y = np.where(mask, indicator_height, np.nan)
        blink_line.set_ydata(indicator_y)
        
        fig.canvas.draw_idle()

    sl_thresh.on_changed(update)
    sl_pad.on_changed(update)
    sl_smooth.on_changed(update)

    update(0)
    plt.show()