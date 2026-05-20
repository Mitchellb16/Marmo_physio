# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:10:04 2026

@author: mitch
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from src.ingestion import load_physio_acq
from src.ecg_processor import ecg_process

def test_full_session_ecg(acq_file_path):
    print("--- Isolating Full-File ECG Processing ---")
    print(f"Target File: {Path(acq_file_path).name}")
    
    start_time = time.time()
    
    print("\n[Step 1] Ingesting Full ACQ File...")
    data, fs = load_physio_acq(str(acq_file_path))
    
    if data is None:
        print("Failed to load data.")
        return
        
    duration_mins = len(data) / fs / 60
    print(f"Successfully loaded. Size: {len(data)} rows. Duration: {duration_mins:.2f} minutes.")
    
    print("\n[Step 2] Passing Full Signal to ecg_process()...")
    
    # We pass the full series to trigger the hang
    try:
        signals, info = ecg_process(data['ECG'], sampling_rate=fs)
        print("\n[SUCCESS] ECG Processing Completed!")
        print(f"Found {len(info['ECG_R_Peaks'])} heartbeats.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline crashed: {e}")
        
    end_time = time.time()
    print(f"\nTotal script execution time: {(end_time - start_time):.2f} seconds.")

if __name__ == "__main__":
    TEST_FILE = PROJECT_ROOT / 'raw_data' / 'test' / '20260406_Poppy_trill(ecg_resp).acq'
    test_full_session_ecg(TEST_FILE)