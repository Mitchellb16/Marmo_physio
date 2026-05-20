# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:46:23 2026

@author: mitch
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

import sys
import pickle
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from src.assembly import process_session
# Import the new plotting function
#from src.plotting import plot_full_session, plot_single_trial

if __name__ == "__main__":
    EYELINK_DIR = PROJECT_ROOT / 'Task' / 'results'
    TEST_DIR = PROJECT_ROOT / 'raw_data' / 'test'
    PREPROCESSED_DIR = PROJECT_ROOT / 'preprocessed_data' 
    # Define where to save the pickled data
    PICKLE_FILE = PREPROCESSED_DIR / 'master_session_dict.pkl'
    
    # =========================================================================
    # LOAD EXISTING DATA 
    # =========================================================================
# =============================================================================
#     if PICKLE_FILE.exists():
#          print(f"Loading preprocessed data from {PICKLE_FILE}...")
#          with open(PICKLE_FILE, 'rb') as f:
#              master_dict = pickle.load(f)
#          print("Load complete! Ready for Feature Extraction.")
#          sys.exit() # Or continue to feature extraction logic
# =============================================================================
    # =========================================================================

    # =========================================================================
    # OPTION B: RUN PREPROCESSING PIPELINE
    # =========================================================================
    master_dict = {}
    
    acq_files = list(TEST_DIR.glob('*.acq'))
    
    if not acq_files:
        print(f"No .acq files found in {TEST_DIR}")
        sys.exit()
        
    print(f"Found {len(acq_files)} files to process.\n" + "="*40)
    
    for acq_file in acq_files:
        print(f"\n>>> Starting Pipeline for: {acq_file.name}")
        
        session = process_session(acq_file, EYELINK_DIR)
        
        master_dict[Path(acq_file).name] = session
         
        print(f"<<< Finished: {acq_file.name}")
        print("-" * 40)
        
    print("\nAll files in test directory processed successfully!")
    
    # Save the master dictionary
    print(f"Saving master dictionary to {PICKLE_FILE}...")
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(master_dict, f)
    print("Save complete!") 
    