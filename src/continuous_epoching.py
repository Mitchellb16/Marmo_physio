# -*- coding: utf-8 -*-
"""
Created on Mon May 18 2026
@author: mitch

This code was written with help from Gemini AI (Gemini 3.1 Pro).
"""

import pandas as pd
import numpy as np

def generate_epochs(unepoched_acq, unepoched_pupil, session_condition, sync_indices, epoch_length=30.0, washout_periods=1):
    """
    Slices continuous acq and pupil data into 30s bins and extracts features
    using the explicit time_acq_absolute column.
    """
    
    acq_time_col = 'time_acq_absolute'
    pupil_time_col = 'time_pupil_absolute'
    
    # 1. Convert row indices to actual times using the explicit time column
    if len(sync_indices) > 0:
        stimulus_times = unepoched_acq[acq_time_col].iloc[sync_indices].tolist()
    else:
        stimulus_times = []
        
    # 2. Base our max time on the time columns
    max_time = min(unepoched_acq[acq_time_col].max(), unepoched_pupil[pupil_time_col].max())
    bins = np.arange(0, max_time, epoch_length)
    
    epoch_features = []
    washout_counter = 0
    
    for i in range(len(bins) - 1):
        start_time = bins[i]
        end_time = bins[i+1]
        
        # 3. Slice using the explicit time columns
        acq_slice = unepoched_acq[(unepoched_acq[acq_time_col] >= start_time) & (unepoched_acq[acq_time_col] < end_time)]
        pupil_slice = unepoched_pupil[(unepoched_pupil[pupil_time_col] >= start_time) & (unepoched_pupil[pupil_time_col] < end_time)]
        
        # Determine Label
        is_stimulus = any((start_time <= t < end_time) for t in stimulus_times)
        
        if is_stimulus:
            label = session_condition
            washout_counter = washout_periods  
        elif washout_counter > 0:
            label = 'washout'
            washout_counter -= 1
        else:
            label = 'silence'
        
        # Extract Features - ACQ Signals
        features = {
            'Epoch_Start': start_time,
            'Label': label,
            'Condition': session_condition,
            'ECG_Rate_Mean': acq_slice['ECG_Rate'].mean() if 'ECG_Rate' in acq_slice.columns else np.nan,
            'ECG_Rate_SD': acq_slice['ECG_Rate'].std() if 'ECG_Rate' in acq_slice.columns else np.nan,
            'RSP_Rate_Mean': acq_slice['RSP_Rate'].mean() if 'RSP_Rate' in acq_slice.columns else np.nan,
            'RSP_Amplitude_Mean': acq_slice['RSP_Amplitude'].mean() if 'RSP_Amplitude' in acq_slice.columns else np.nan,
            'RSP_Rate_SD': acq_slice['RSP_Rate'].std() if 'RSP_Rate' in acq_slice.columns else np.nan,
            
        }
        
        # Extract Features - Pupil
        if 'Pupil_PctMax' in pupil_slice.columns:
            features['Pupil_PctMax_Mean'] = pupil_slice['Pupil_PctMax'].mean()
# =============================================================================
#             features['Pupil_PctMax_Max'] = pupil_slice['Pupil_PctMax'].max()
#             features['Pupil_PctMax_SD'] = pupil_slice['Pupil_PctMax'].std()
# =============================================================================
            
        epoch_features.append(features)
        
    return pd.DataFrame(epoch_features)