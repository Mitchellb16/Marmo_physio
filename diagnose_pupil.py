#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:11:43 2026

@author: mitchell

Diagnostic script to visualize missing pupil data in the -60s baseline window.
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Path Setup (Assuming script is in physio_pipeline/)
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
PICKLE_FILE = PROJECT_ROOT / 'preprocessed_data' / 'master_session_dict.pkl'

if not PICKLE_FILE.exists():
    print(f"Error: Could not find {PICKLE_FILE}")
    exit()

print(f"Loading data from {PICKLE_FILE}...")
with open(PICKLE_FILE, 'rb') as f:
    master_dict = pickle.load(f)

for session_name, session_data in master_dict.items():
    if 'epoched_pupil' not in session_data:
        continue
        
    pupil_df = session_data['epoched_pupil']
    
    # Find the pupil column dynamically
    p_col = 'Pupil_Clean'
    
    # Isolate the 60-second baseline window
    baseline_df = pupil_df[(pupil_df['Time'] >= -60.0) & (pupil_df['Time'] < 0.0)]
    
    # --- 1. CONSOLE REPORT ---
    print(f"\n{'='*40}")
    print(f"SESSION: {session_name}")
    print(f"{'='*40}")
    
    for trial_id, trial_df in baseline_df.groupby('Trial'):
        nan_pct = trial_df[p_col].isna().mean() * 100
        status = "❌ REJECTED" if nan_pct > 30 else "✅ PASSED"
        print(f"  Trial {trial_id}: {nan_pct:.1f}% Missing Data -> {status}")
    
    # --- 2. DIAGNOSTIC PLOT ---
    plt.figure(figsize=(14, 6))
    sns.set_style("darkgrid")
    
    # Plot with markers so gaps (NaNs) are visually obvious
    sns.lineplot(
        data=baseline_df, 
        x='Time', 
        y=p_col, 
        hue='Trial', 
        palette='tab10', 
        linewidth=1,
        alpha=0.8
    )
    
    plt.title(f"Pupil Baseline Diagnostics (-60s to 0s) | {session_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Time relative to stimulus (s)", fontsize=12)
    plt.ylabel(f"{p_col} Value", fontsize=12)
    
    # Draw a vertical line to show where the stimulus occurs
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stimulus Onset')
    
    plt.legend(title="Trial", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()