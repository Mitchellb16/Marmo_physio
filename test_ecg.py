# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:28:55 2026

@author: mitch
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

import sys
import neurokit2 as nk
import matplotlib.pyplot as plt
from pathlib import Path

# Dynamically determine paths regardless of OS or machine
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Ensure the physio_pipeline folder is accessible to import your modules
sys.path.append(str(CURRENT_DIR))

from src.ecg_processor import ecg_process
from src.ingestion import load_physio_acq

def test_ecg_module(acq_file_path, sample_seconds=10):
    """
    Loads an .acq file, crops it at the first TTL pulse, 
    and visualizes the ECG processing steps.
    """
    print(f"Loading {acq_file_path}...")
    data, fs = load_physio_acq(str(acq_file_path))
    
    if data is None:
        return

    print(f"Detected Sampling Rate: {fs} Hz")
    
    if 'ECG' not in data.columns:
        print("Error: 'ECG' column not found.")
        return
        
    # --- CROPPING LOGIC ---
    start_idx = 0
    if 'block_sync' in data.columns:
        # Find rising edges in the TTL channel
        threshold = 3.0
        trigger = (data['block_sync'] >= threshold) & (data['block_sync'].shift(1) < threshold)
        pulse_indices = data.index[trigger].tolist()
        
        if pulse_indices:
            start_idx = pulse_indices[0]
            print(f"Success: First TTL pulse found at {start_idx / fs:.2f} seconds. Cropping from here.")
        else:
            print("Warning: No TTL pulses found > 3.0V. Defaulting to start of file.")
    else:
        print("Warning: 'block_sync' channel not found. Defaulting to start of file.")

    # Define the exact bounds for the data slice
    end_idx = int(start_idx + (fs * sample_seconds))
    
    # Slice the pandas series
    raw_ecg_signal = data['ECG'].iloc[start_idx:end_idx]

    print("Processing ECG signal...")
    
    # Pass to your processor
    signals, info = ecg_process(raw_ecg_signal, sampling_rate=fs)
    
    print(f"Found {len(info['ECG_R_Peaks'])} R-peaks.")
    print(f"Average Heart Rate: {signals['ECG_Rate'].mean():.1f} BPM")
    
    # Visualize using NeuroKit2's built-in plotting
    print("Generating plot...")
    nk.ecg_plot(signals, info)
    
    # Adjust plot for the high density of marmoset beats
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.show()

if __name__ == "__main__":
    # Construct the relative path to the exact test file
    TEST_FILE = PROJECT_ROOT / 'raw_data' / 'test' / '20260407_Freddy_tsik(ecg_resp).acq'

    test_ecg_module(TEST_FILE, sample_seconds=115)