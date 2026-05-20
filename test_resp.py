# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:49:08 2026

@author: mitch
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

import sys
import neurokit2 as nk
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from src.resp_processor import resp_process
from src.ingestion import load_physio_acq

def test_resp_module(acq_file_path, sample_seconds=20):
    """
    Loads an .acq file, crops it starting 10s before the first TTL pulse, 
    visualizes both Belt and Thermistor processing steps, and returns the dataframes.
    """
    print(f"Loading {acq_file_path}...")
    data, fs = load_physio_acq(str(acq_file_path))
    
    if data is None:
        return None

    print(f"Detected Sampling Rate: {fs} Hz")
    
    if 'Respiration' not in data.columns:
        print("Error: 'Respiration' column not found.")
        return None
        
    start_idx = 0
    if 'block_sync' in data.columns:
        threshold = 3.0
        trigger = (data['block_sync'] >= threshold) & (data['block_sync'].shift(1) < threshold)
        pulse_indices = data.index[trigger].tolist()
        
        if pulse_indices:
            # Include 10 seconds before the onset. 
            # Using max(0, ...) to prevent negative indexing if the TTL is very early.
            start_idx = max(0, pulse_indices[0] - int(fs * 10))
            print(f"Success: First TTL pulse found. Starting window 10s prior at {start_idx / fs:.2f} seconds.")
    
    end_idx = int(start_idx + (fs * sample_seconds))
    
    # ==========================================
    # 1. PROCESS MECHANOSENSORY BELT
    # ==========================================
    print("\n" + "="*40)
    print("PROCESSING MECHANOSENSORY BELT ('Respiration')")
    print("="*40)
    raw_resp_belt = data['Respiration'].iloc[start_idx:end_idx]
    
    # We will stick to the default khodadad2018 method
    belt_signals, belt_info = resp_process(raw_resp_belt, sampling_rate=fs, method="khodadad2018")
    
    print(f"Belt - Found {len(belt_info['RSP_Peaks'])} Inhalations.")
    print(f"Belt - Avg Breathing Rate: {belt_signals['RSP_Rate'].mean():.1f} breaths/min")
    
    print("Generating Belt plot...")
    belt_signals.RSP_Raw = belt_signals.RSP_Raw - belt_signals.RSP_Raw.mean() 
    nk.rsp_plot(belt_signals, belt_info)
    fig1 = plt.gcf()
    fig1.suptitle("Mechanosensory Belt (Method: khodadad2018)", fontsize=16, fontweight='bold')
    fig1.set_size_inches(15, 10)

    # ==========================================
    # 2. PROCESS THERMISTOR
    # ==========================================
    therm_signals, therm_info = None, None
    if 'Thermistor' in data.columns:
        print("\n" + "="*40)
        print("PROCESSING THERMISTOR ('Thermistor')")
        print("="*40)
        raw_resp_therm = data['Thermistor'].iloc[start_idx:end_idx]
        
        therm_signals, therm_info = resp_process(raw_resp_therm, sampling_rate=fs, method="khodadad2018")
        
        print(f"Thermistor - Found {len(therm_info['RSP_Peaks'])} Inhalations.")
        print(f"Thermistor - Avg Breathing Rate: {therm_signals['RSP_Rate'].mean():.1f} breaths/min")
        
        print("Generating Thermistor plot...")
        # subtract mean out of raw signal so it is easier to see in our plot
        therm_signals['RSP_Raw'] = therm_signals['RSP_Raw'] - therm_signals['RSP_Raw'].mean()
        nk.rsp_plot(therm_signals, therm_info)
        fig2 = plt.gcf()
        fig2.suptitle("Thermistor (Method: khodadad2018)", fontsize=16, fontweight='bold')
        fig2.set_size_inches(15, 10)
    else:
        print("\nWarning: 'Thermistor' column not found in this file.")

    plt.show()

    # Return a dictionary so you can easily access both sets of data in the console
    return {
        'belt': (belt_signals, belt_info),
        'thermistor': (therm_signals, therm_info)
    }

if __name__ == "__main__":
    TEST_FILE = PROJECT_ROOT / 'raw_data' / 'test' / '20260511_Plusle_silence(ecg_resp).acq'
    
    # Store the outputs in variables if running interactively
    results = test_resp_module(TEST_FILE, sample_seconds=60)