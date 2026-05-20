# -*- coding: utf-8 -*-
"""
Created on Mon May 18 2026
@author: mitch

Execution script for continuous epoching and feature correlation analysis.
This code was written with help from Gemini AI (Gemini 3.1 Pro).
"""

import sys
import pickle
import pandas as pd
from pathlib import Path

# Adjust these imports based on where your src folder is located relative to this script
from src.continuous_epoching import generate_epochs
from src.plotting import plot_feature_correlations, plot_feature_pairplot

# Path Setup
CURRENT_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = CURRENT_DIR.parent                  
sys.path.append(str(CURRENT_DIR)) 


# ---------------------------------------------------------
# 1. DEFINE DIRECTORIES
# ---------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent 
RESULTS_DIR = CURRENT_DIR / 'results'
PREPROCESSED_DIR = PROJECT_ROOT / 'preprocessed_data' 
PICKLE_FILE = PREPROCESSED_DIR / 'master_session_dict.pkl'

epoch_length = 10

print(f"Loading preprocessed data from {PICKLE_FILE}...")
with open(PICKLE_FILE, 'rb') as f:
    master_dict = pickle.load(f)
    
all_sessions_epochs = []

# ---------------------------------------------------------
# 2. EXTRACT LOOP (Continuous 30s Epochs)
# ---------------------------------------------------------
for session_name, session_data in master_dict.items():
    print(f"Segmenting into 30s continuous epochs for: {session_name}")
    
    # Parse Filename
    name_no_ext = session_name.replace('.acq', '') 
    parts = name_no_ext.split('_')                 
    subject = parts[1] if len(parts) > 1 else "Unknown"
    condition = parts[2].split('(')[0] if len(parts) > 2 else "Unknown" 
    
    # Extract variables
    unepoched_acq = session_data['unepoched_acq']
    unepoched_pupil = session_data['unepoched_pupil']
    
    # Pass the raw row indices directly 
    sync_indices = session_data['acq_sync_index']
    
    # Run the epoching function
    session_epochs_df = generate_epochs(
        unepoched_acq=unepoched_acq, 
        unepoched_pupil=unepoched_pupil, 
        session_condition=condition, 
        sync_indices=sync_indices,       # Pass raw indices here
        epoch_length=epoch_length,
        washout_periods=1
    )
    
    # Append Metadata
    session_epochs_df.insert(0, 'Session', session_name)
    session_epochs_df.insert(1, 'Subject', subject)
    session_epochs_df.insert(2, 'Epoch_len', epoch_length)
    
    all_sessions_epochs.append(session_epochs_df)

# ---------------------------------------------------------
# 3. AGGREGATE, SAVE, AND VISUALIZE
# ---------------------------------------------------------
global_epochs_df = pd.concat(all_sessions_epochs, ignore_index=True)

# Ensure the results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

output_path = RESULTS_DIR / f'global_{epoch_length}s_features.csv'
global_epochs_df.to_csv(output_path, index=False)
print(f"\nSaved global feature dataset ({len(global_epochs_df)} total epochs) to {output_path}\n")

print("Generating global cross-feature correlation matrix...")
plot_feature_correlations(global_epochs_df, "All subjects")

print('\nPlotting individual cross correlation matrices via groupby...')
# Group the dataset by subject name and loop through each individual's subset data
for subject_name, subject_df in global_epochs_df.groupby('Subject'):
    print(f" -> Processing correlation plot for subject: {subject_name} ({len(subject_df)} epochs)")
    
    # If your plotting function has a title parameter, you might want to adjust it, 
    # but passing the filtered dataframe directly works perfectly.
    plot_feature_correlations(subject_df, subject_name)

print("\nGenerating feature pairplots colored by epoch label...")
plot_feature_pairplot(global_epochs_df)

print("\nAnalysis complete.")
