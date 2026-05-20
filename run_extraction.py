#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:23:00 2026

@author: mitchell
"""
# This code was written with help from Gemini AI (Gemini 3.1 Pro).
# -*- coding: utf-8 -*-
"""
Execution script for extracting trial features from preprocessed pickle data.
"""
import pandas as pd
import pickle
import sys
from pathlib import Path
import scipy.stats

# Path Setup
CURRENT_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = CURRENT_DIR.parent                  
sys.path.append(str(CURRENT_DIR))                  

# Import our modular logic
from src.feature_extraction import extract_signal_features, extract_pupil_features
from src.plotting import plot_baseline_vs_response

if __name__ == "__main__":
    
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
        
    all_session_features = []
    prefixes_to_extract = ["ECG", "RSP", "Thermistor"]
    
    # 3. EXTRACT LOOP
    for session_name, session_data in master_dict.items():
        print(f"\nExtracting features for: {session_name}")
        
        # Parse Filename
        name_no_ext = session_name.replace('.acq', '') 
        parts = name_no_ext.split('_')                 
        subject = parts[1] if len(parts) > 1 else "Unknown"
        condition = parts[2].split('(')[0] if len(parts) > 2 else "Unknown" 
        
        session_df = None
        
        # --- A. Extract Continuous ACQ Signals ---
        for prefix in prefixes_to_extract:
            signal_df = extract_signal_features(session_data['epoched_acq'], prefix=prefix)
            
            if signal_df is not None:
                if session_df is None:
                    session_df = signal_df
                else:
                    session_df = pd.merge(session_df, signal_df, on='Trial', how='outer')
                    
        # --- B. Extract Native Pupil Data ---
        if 'epoched_pupil' in session_data:
            pupil_df = extract_pupil_features(session_data['epoched_pupil'])
            if pupil_df is not None:
                if session_df is None:
                    session_df = pupil_df
                else:
                    session_df = pd.merge(session_df, pupil_df, on='Trial', how='outer')
        
        if session_df is not None:
            session_df['Session'] = session_name
            session_df['Subject'] = subject
            session_df['Condition'] = condition
            all_session_features.append(session_df)
        
    final_features_df = pd.concat(all_session_features, ignore_index=True)
    
    # 4. Z-SCORE ACROSS SESSION
    print("\nZ-scoring metrics within sessions...")
    
    # Only Z-score the ACQ Delta variables. We explicitly ignore Pupil features.
    cols_to_zscore = [col for col in final_features_df.columns 
                      if col.endswith('_Delta') and not col.startswith('Pupil_')]
    
    for col in cols_to_zscore:
        final_features_df[f'{col}_Z'] = final_features_df.groupby('Session')[col].transform(
            lambda x: scipy.stats.zscore(x, nan_policy='omit')
        )
        
    # 5. SAVE OUTPUT
    OUT_FILE = RESULTS_DIR / 'extracted_features.csv'
    final_features_df.to_csv(OUT_FILE, index=False)
    print(f"\nFeature extraction complete! Saved to {OUT_FILE}")
    
    # 6. PLOT
    print("Generating summary plots...")
    for prefix in prefixes_to_extract:
        if f"{prefix}_Rate_Baseline" in final_features_df.columns:
            plot_baseline_vs_response(final_features_df, metric_prefix=f"{prefix}_Rate", group_col="Condition")
            plot_baseline_vs_response(final_features_df, metric_prefix=f"{prefix}_Var", group_col="Condition")
            
    # --- PUPIL PLOTS ---
    # 1) Averaged PctMax for baseline and response
    if "Pupil_Mean_Baseline" in final_features_df.columns:
        plot_baseline_vs_response(final_features_df, metric_prefix="Pupil_Mean", group_col="Condition")
        
    # 2) Maximum PctMax for baseline and response
    if "Pupil_Max_Baseline" in final_features_df.columns:
        plot_baseline_vs_response(final_features_df, metric_prefix="Pupil_Max", group_col="Condition")