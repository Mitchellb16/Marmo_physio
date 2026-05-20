# -*- coding: utf-8 -*-
"""
Generates a Methodology figure demonstrating raw vs. cleaned signals 
and peak/trough detection across a 10-second continuous window.
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 1. SETUP & LOAD DATA ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
PREPROCESSED_DIR = PROJECT_ROOT / 'preprocessed_data'
PICKLE_FILE = PREPROCESSED_DIR / 'master_session_dict.pkl'

if not PICKLE_FILE.exists():
    print(f"Error: Cannot find {PICKLE_FILE}")
    exit()

print("Loading data...")
with open(PICKLE_FILE, 'rb') as f:
    master_dict = pickle.load(f)

# Grab the first session available
session_name = list(master_dict.keys())[6]
session_data = master_dict[session_name]
print(f"Plotting 10s chunk from: {session_name}")

sampling_rate = 2000
pupil_sampling_rate = 500

acq_df = session_data['unepoched_acq']
pupil_df = session_data.get('unepoched_pupil', None)

# --- 2. DEFINE THE 10-SECOND SLICE ---
# Find the middle of the session
max_time = acq_df['time_acq_absolute'].max()
mid_time = max_time / 2

# You can manually change this START_TIME if you want to hunt for a "prettier" chunk!
START_TIME = mid_time 
END_TIME = START_TIME + 10.0

# Slice the dataframes
# =============================================================================
# acq_slice = acq_df[(acq_df['time_acq_absolute'] >= START_TIME) & (acq_df['time_acq_absolute'] <= END_TIME)]
# =============================================================================

acq_slice = acq_df.loc[250 * sampling_rate:255 * sampling_rate,:]

if pupil_df is not None:
    # Use whatever your continuous pupil time column is named (assuming time_pupil_absolute)
    time_col_pupil = 'time_pupil_absolute' if 'time_pupil_absolute' in pupil_df.columns else 'Time'
    pupil_slice = pupil_df.loc[150 * pupil_sampling_rate: 155 * pupil_sampling_rate]

# --- 3. PLOTTING ---
# Set up poster-style contexts
sns.set_context("poster", font_scale=1.0)
sns.set_style("whitegrid")

# Helper function to plot Raw vs Clean + Peaks
def plot_signal(df, time_col, raw_col, clean_col, peak_col=None, trough_col=None, title="", color="blue"):
    fig, ax = plt.subplots(figsize=(14,4))
    if raw_col in df.columns and clean_col in df.columns:
        # Plot Raw (Light Gray, slightly thicker)
        ax.plot(df[time_col], df[raw_col], color='lightgray', linewidth=3, label='Raw', alpha=0.7)
        # Plot Clean (Colored, thinner line)
        ax.plot(df[time_col], df[clean_col], color=color, linewidth=2, label='Cleaned')
        
        # Overlay Peaks if they exist
        if peak_col and peak_col in df.columns:
            peaks = df[df[peak_col] == 1]
            ax.scatter(peaks[time_col], peaks[clean_col], color='gold', edgecolor='black', s=100, zorder=5, label='Peaks')
            
        # Overlay Troughs if they exist
        if trough_col and trough_col in df.columns:
            troughs = df[df[trough_col] == 1]
            ax.scatter(troughs[time_col], troughs[clean_col], color='cyan', edgecolor='black', s=100, zorder=5, marker='v', label='Troughs')

        ax.set_title(title, fontweight='bold', loc='left')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, f"Columns missing for {title}", ha='center')

# 1. ECG
plot_signal(acq_slice, 'time_acq_absolute', 'ECG_Raw', 'ECG_Clean', peak_col='ECG_R_Peaks', title="ECG Processing & R-Peak Detection", color="#E63946")

# 2. Respiration (Belt)
plot_signal(acq_slice, 'time_acq_absolute', 'RSP_Raw', 'RSP_Clean', peak_col='RSP_Peaks', trough_col='RSP_Troughs', title="Pneumatic Belt & Phase Detection", color="#2A9D8F")

# 3. Thermistor
# =============================================================================
# plot_signal(axes[2], acq_slice, 'time_acq_absolute', 'Thermistor_Raw', 'Thermistor_Clean', peak_col='Thermistor_Peaks', trough_col='Thermistor_Troughs', title="Thermistor Airflow Processing", color="#E76F51")
# =============================================================================

# 4. Pupil
if pupil_df is not None:
    # Assuming your original raw pupil is called something like 'pupil_size' and your clean is 'Pupil_Clean'
    # Adjust these strings to match your pupil_processor.py outputs!
    raw_pupil_col = [col for col in pupil_slice.columns if 'pupil' in col.lower() and 'raw' not in col.lower() and 'clean' not in col.lower()][0] 
    plot_signal(pupil_slice, time_col_pupil, raw_col=raw_pupil_col, clean_col='Pupil_Clean', title="Pupillometry (Blink Interpolation & Smoothing)", color="#457B9D")


plt.tight_layout()
plt.show()