# -*- coding: utf-8 -*-
"""
Execution script to generate 4 types of epochs and extract features.
"""

import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path
import sys

# Adjust imports based on your structure
from src.feature_extraction2 import extract_features, audit_quality

# Path Setup
CURRENT_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = CURRENT_DIR.parent                  
sys.path.append(str(CURRENT_DIR))  

# 1. DEFINE DIRECTORIES
PREPROCESSED_DIR = PROJECT_ROOT / 'preprocessed_data' 
PICKLE_FILE = PREPROCESSED_DIR / 'master_session_dict.pkl'

RESULTS_DIR = CURRENT_DIR / 'results' 
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. LOAD DATA
if not PICKLE_FILE.exists():
    print(f"Cannot find {PICKLE_FILE}. Please run main.py to preprocess first.")
    sys.exit()
    
print(f"Loading preprocessed data from {PICKLE_FILE}...")
with open(PICKLE_FILE, 'rb') as f:
    master_dict = pickle.load(f)

                        
epoch_len=30.0

all_results = {'interval': [], 'baseline': [], 'stimulus': [], 'peri': []}

for session, data in master_dict.items():
    print(f"Processing session: {session}")
    
    # Extract correct dict entries
    acq_df = data['unepoched_acq']
    pupil_df = data['unepoched_pupil']
    fs_acq = data['fs_acq']
    fs_pupil = data['fs_pupil']
    acq_events = data['acq_sync_index']
    pupil_events = data['pupil_sync_index']
    
    # Parse Filename
    name_no_ext = session.replace('.acq', '') 
    parts = name_no_ext.split('_')                 
    subject = parts[1] if len(parts) > 1 else "Unknown"
    condition = parts[2].split('(')[0] if len(parts) > 2 else "Unknown"
    
    # --- 1. GENERATE PARALLEL EPOCH DICTIONARIES ---
    
    # Continuous Interval Bins (requires generated events)
# =============================================================================
#     bin_samples_acq = int(epoch_len * fs_acq)
#     bin_samples_pupil = int(epoch_len * fs_pupil)
#     cont_events_acq = np.arange(0, len(acq_df), bin_samples_acq)
#     cont_events_pupil = np.arange(0, len(pupil_df), bin_samples_pupil)
# =============================================================================
    
    # Build Dictionary of ACQ Epochs
    acq_epoch_bundles = {
        'interval': nk.epochs_create(acq_df, events=None, sampling_rate=fs_acq, epochs_start=0, epochs_end=epoch_len),
        'baseline': nk.epochs_create(acq_df, events=acq_events, sampling_rate=fs_acq, epochs_start=-10, epochs_end=0),
        'stimulus': nk.epochs_create(acq_df, events=acq_events, sampling_rate=fs_acq, epochs_start=0, epochs_end=10),
        'peri': nk.epochs_create(acq_df, events=acq_events, sampling_rate=fs_acq, epochs_start=-10, epochs_end=10)
    }
    
    # Build Dictionary of Pupil Epochs
    pupil_epoch_bundles = {
        'interval': nk.epochs_create(pupil_df, events=None, sampling_rate=fs_pupil, epochs_start=0, epochs_end=epoch_len),
        'baseline': nk.epochs_create(pupil_df, events=pupil_events, sampling_rate=fs_pupil, epochs_start=-60, epochs_end=0),
        'stimulus': nk.epochs_create(pupil_df, events=pupil_events, sampling_rate=fs_pupil, epochs_start=0, epochs_end=10),
        'peri': nk.epochs_create(pupil_df, events=pupil_events, sampling_rate=fs_pupil, epochs_start=-10, epochs_end=10)
    }
    
    # --- 2. EXTRACT & AUDIT ---
    
    for name in ['interval', 'baseline', 'stimulus', 'peri']:
        if not acq_epoch_bundles[name]:
            continue
            
        # Determine logic type
        a_type = 'peri' if name == 'peri' else 'interval'
        
        # Pass BOTH dictionaries to extract features and audit
        features = extract_features(acq_epoch_bundles[name], pupil_epoch_bundles[name], 
                                    sampling_rate=fs_acq, analysis_type=a_type)
                                    
        features = audit_quality(acq_epoch_bundles[name], pupil_epoch_bundles[name], 
                                 features, threshold=0.5)
        
        # --- ADD MISSING METADATA AND TRIAL COLUMN ---
        features['Session'] = session
        features['Epoch_len'] = epoch_len
        
        # Parse Subject and Label directly from the filename string 
        # (e.g., "20260406_Poppy_trill(ecg_resp).acq")
        try:
            parts = session.split('_')
            features['Subject'] = parts[1]
            features['Label'] = parts[2].split('(')[0]
        except IndexError:
            features['Subject'] = "Unknown"
            features['Label'] = "Unknown"
            
        features['Epoch_Type'] = name
        
        # The index currently holds the dictionary keys from NeuroKit (the Trial numbers).
        # We rename the index to 'Trial' and pop it out into a standard column.
        features = features.rename_axis('Trial').reset_index()
        
        all_results[name].append(features)

# --- 3. CONCATENATE AND EXPORT ---
Path('results').mkdir(exist_ok=True)

for name, df_list in all_results.items():
    if df_list:
        final_df = pd.concat(df_list)
        # Reorder columns to put Session first for readability
        cols = ['Session', 'Epoch_Type'] + [c for c in final_df if c not in ['Session', 'Epoch_Type']]
        final_df = final_df[cols]
        
        out_file = f'results/{name}_features.csv'
        final_df.to_csv(out_file, index=False)
        print(f"Saved {len(final_df)} rows to {out_file}")
        

