#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:54:24 2026

@author: mitchell
"""
# run_pca.py

import sys
from pathlib import Path
import pandas as pd

# Add the current directory to sys.path so Python can find the 'src' folder
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

# Import the math/plotting function from our src folder
from src.dimensionality import run_and_plot_pca

if __name__ == '__main__':
    # 1. DEFINE DIRECTORIES
    # ---------------------------------------------------------
    # Assuming run_pca.py is in the folder directly above src/ and results/
    RESULTS_DIR = CURRENT_DIR / 'results' 
    CSV_FILE = RESULTS_DIR / 'global_10s_features.csv'
    
    # 2. LOAD DATA
    # ---------------------------------------------------------
    print(f"Loading epoched features from {CSV_FILE}...")
    
    if not CSV_FILE.exists():
        print(f"ERROR: Could not find {CSV_FILE}.")
        print("Please ensure your feature extraction script has run and saved the CSV.")
        sys.exit()
        
    epoch_df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(epoch_df)} epochs.")
    
    # 3. PRE-PCA CLEANING
    # ---------------------------------------------------------
    # PCA uses linear algebra that will instantly crash if there are any NaN (missing) values.
    # We drop any epochs that are missing data (e.g., if the eye-tracker lost the pupil).
    initial_len = len(epoch_df)
    epoch_df = epoch_df.dropna()
    dropped = initial_len - len(epoch_df)
    if dropped > 0:
        print(f"Dropped {dropped} epochs due to missing (NaN) values.")
        
    # 4. EXECUTE
    # ---------------------------------------------------------
    pca_model, pca_results = run_and_plot_pca(epoch_df)
