# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:21:01 2026

@author: mitch
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).



import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Dynamically determine paths regardless of OS or machine
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from src.pupil_processor import pupil_process
from src.ingestion import load_pupil_edf

def test_pupil_module(edf_file_path, start_time_sec=30, sample_seconds=60):
    """
    Loads an EDF file, processes a chunk of it, and visualizes the blink interpolation
    using the pipeline-standard MAD-based pupil processor.
    """
    print(f"Loading {edf_file_path}...")
    
    # 1. Load data using the ingestion function
    data, fs = load_pupil_edf(edf_file_path)
    
    if data is None:
        return None
        
    print(f"Detected Sampling Rate: {fs} Hz")
    
    # 2. Slice a chunk for testing
    start_idx = int(fs * start_time_sec)
    end_idx = int(start_idx + (fs * sample_seconds))
    end_idx = min(end_idx, len(data)) # Ensure we don't slice past the end
    
    raw_pupil_chunk = data.iloc[start_idx:end_idx].copy()
    time_chunk = raw_pupil_chunk['Time'].values

    # 3. Identify how many eyes we are tracking
    eye_cols = [col for col in raw_pupil_chunk.columns if 'Pupil_Eye_' in col]
    num_eyes = len(eye_cols)
    print(f"Eyes Tracked: {num_eyes}")

    # --- VISUALIZATION SETUP ---
    print("Processing signals and generating plot...")
    
    # Create 2 subplots per eye (Raw on top, Processed on bottom)
    fig, axes = plt.subplots(num_eyes * 2, 1, figsize=(15, 5 * (num_eyes * 2)), sharex=True)
    
    # Force axes to be a 1D iterable array
    if type(axes) is not np.ndarray:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
        
    # 4. Process and Plot Each Eye
    for i, col in enumerate(eye_cols):
        raw_signal = raw_pupil_chunk[col]
        
        # Run our standardized processor
        signals, info = pupil_process(raw_signal, sampling_rate=fs, mad_multiplier=4.0)
        
        # Print QC metrics
        print(f"{col} - MAD Threshold Used: {info['mad_threshold_used']:.2f}")
        print(f"{col} - Percent Unusable/Blinks: {info['percent_lost']:.2f}%")
        
        # Assign the two subplots for this specific eye
        ax_raw = axes[i * 2]
        ax_clean = axes[i * 2 + 1]
        
        # --- Plot 1: Raw Signal ---
        ax_raw.plot(time_chunk, signals['Pupil_Raw'], color='dimgrey', label='Raw Pupil', linewidth=1.5)
        ax_raw.set_title(f'{col} - Raw Signal', fontsize=14, fontweight='bold')
        ax_raw.set_ylabel('Pupil Size (a.u.)', fontsize=12)
        ax_raw.legend(loc='lower right')
        ax_raw.grid(True, linestyle='--', alpha=0.6)
        
        # --- Plot 2: Cleaned & Interpolated Signal ---
        ax_clean.plot(time_chunk, signals['Pupil_Clean'], color='dodgerblue', label='Cleaned & Smoothed', linewidth=2)
        
        # Dynamic Y-axis limits tailored to the cleaned data
        y_min = np.nanmin(signals['Pupil_Clean']) * 0.8
        y_max = np.nanmax(signals['Pupil_Clean']) * 1.2
        ax_clean.set_ylim(y_min, y_max)
        
        # Highlight the blink regions (Where Quality == 0)
        blink_mask = signals['Pupil_Quality'] == 0
        ax_clean.fill_between(time_chunk, y_min, y_max, where=blink_mask, 
                         color='red', alpha=0.15, label='Detected Blinks (MAD Padded)')

        ax_clean.set_title(f'{col} - Processed (MAD Blink Interpolation)', fontsize=14, fontweight='bold')
        ax_clean.set_ylabel('Pupil Size (a.u.)', fontsize=12)
        ax_clean.legend(loc='lower right')
        ax_clean.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return raw_pupil_chunk

if __name__ == "__main__":
    # --- EASY SWITCH ---
    # Just change this single string to test a different session!
    SESSION_NAME = 'freddy3_2026_04_13_14_20'
    
    # Dynamically build the path based on your project's folder structure
    TEST_EDF = PROJECT_ROOT / 'Task' / 'results' / SESSION_NAME / f'{SESSION_NAME}.EDF'
    
    # Quick safety check to prevent cryptic Pandas errors if the file doesn't exist
    if not TEST_EDF.exists():
        print(f"❌ ERROR: File not found at {TEST_EDF}")
        print("Please check the SESSION_NAME and ensure the folder exists.")
    else:
        print(f"✅ Found EDF for {SESSION_NAME}. Beginning processing...")
        
        # Starting 60 seconds into the file, looking at a 100-second window
        # (Adjust these values if you want to look at different parts of the recording)
        test_pupil_module(TEST_EDF, start_time_sec=170, sample_seconds=500)