# -*- coding: utf-8 -*-
"""
Created on [Current Date]

@author: mitch
"""
import pandas as pd
import numpy as np

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

# -*- coding: utf-8 -*-
"""
Created on [Current Date]
@author: mitch
"""
import pandas as pd
import numpy as np

def extract_signal_features(epoched_acq, prefix="ECG"):
    """
    Extracts Quality-gated Dual-Window features dynamically based on the signal prefix.
    Baseline = -60s to 0s | Response = 0s to +10s.
    """
    features = []
    rate_col = f"{prefix}_Rate"
    quality_col = f"{prefix}_Quality"
    
    if rate_col not in epoched_acq.columns:
        print(f"  [-] Skipping {prefix}: '{rate_col}' not found in DataFrame.")
        return None

    for trial_id, trial_df in epoched_acq.groupby('Trial'):
        trial_metrics = {'Trial': trial_id}
        
        # --- SLICE WINDOWS ---
        baseline_rate_df = trial_df[(trial_df['Time'] >= -10.0) & (trial_df['Time'] < 0.0)]
        response_rate_df = trial_df[(trial_df['Time'] >= 0.0) & (trial_df['Time'] <= 10.0)]
        
        baseline_var_df = trial_df[(trial_df['Time'] >= -10.0) & (trial_df['Time'] < 0.0)]
        response_var_df = response_rate_df 
        
        # --- QUALITY CHECK ---
        rate_nan_pct = baseline_rate_df[rate_col].isna().mean()
        valid_for_rate = rate_nan_pct <= 0.30 
        
        if quality_col in trial_df.columns:
            var_interp_pct = (baseline_var_df[quality_col] < 0.6).mean() 
            valid_for_var = var_interp_pct <= 0.15 
        else:
            valid_for_var = valid_for_rate
        
        if not valid_for_rate:
            print(f"  [!] Warning: Trial {trial_id} rejected for {prefix} Rate (>30% NaN).")
        if not valid_for_var:
            print(f"  [!] Warning: Trial {trial_id} rejected for {prefix} Variability (Poor Quality).")
            
        # --- EXTRACT METRICS ---
        if valid_for_rate and not baseline_rate_df.empty and not response_rate_df.empty:
            bl_rate = baseline_rate_df[rate_col].mean(skipna=True)
            resp_rate = response_rate_df[rate_col].mean(skipna=True)
            trial_metrics[f'{prefix}_Rate_Baseline'] = bl_rate
            trial_metrics[f'{prefix}_Rate_Response'] = resp_rate
            trial_metrics[f'{prefix}_Rate_Delta'] = resp_rate - bl_rate
        else:
            trial_metrics[f'{prefix}_Rate_Baseline'] = np.nan
            trial_metrics[f'{prefix}_Rate_Response'] = np.nan
            trial_metrics[f'{prefix}_Rate_Delta'] = np.nan
            
        if valid_for_var and not baseline_var_df.empty and not response_var_df.empty:
            bl_var = baseline_var_df[rate_col].std(skipna=True)
            resp_var = response_var_df[rate_col].std(skipna=True)
            trial_metrics[f'{prefix}_Var_Baseline'] = bl_var
            trial_metrics[f'{prefix}_Var_Response'] = resp_var
            trial_metrics[f'{prefix}_Var_Delta'] = resp_var - bl_var
        else:
            trial_metrics[f'{prefix}_Var_Baseline'] = np.nan
            trial_metrics[f'{prefix}_Var_Response'] = np.nan
            trial_metrics[f'{prefix}_Var_Delta'] = np.nan
            
        features.append(trial_metrics)
        
    return pd.DataFrame(features)


def extract_pupil_features(epoched_pupil):
    """
    Extracts pupil features directly from the pre-normalized Pupil_PctMax column.
    Baseline = -60s to 0s | Response = 0s to +30s.
    """
    features = []
    norm_col = 'Pupil_PctMax' 
    
    if norm_col not in epoched_pupil.columns:
        print(f"  [-] Skipping Pupil: '{norm_col}' not found in DataFrame.")
        return None

    for trial_id, trial_df in epoched_pupil.groupby('Trial'):
        trial_metrics = {'Trial': trial_id}
        
        # --- SLICE WINDOWS ---
        baseline_df = trial_df[(trial_df['Time'] >= -60.0) & (trial_df['Time'] < 0.0)]
        response_df = trial_df[(trial_df['Time'] >= 0.0) & (trial_df['Time'] <= 30.0)]
        
        # --- QUALITY CHECK ---
        nan_pct = baseline_df[norm_col].isna().mean()
        valid = nan_pct <= 0.30 
        
        if not valid:
            print(f"  [!] Warning: Trial {trial_id} rejected for Pupil (>30% missing in baseline).")
            
        # --- EXTRACT METRICS ---
        if valid and not baseline_df.empty and not response_df.empty:
            # 1. Averaged PctMax
            trial_metrics['Pupil_Mean_Baseline'] = baseline_df[norm_col].mean(skipna=True)
            trial_metrics['Pupil_Mean_Response'] = response_df[norm_col].mean(skipna=True)
            
            # 2. Maximum PctMax
            trial_metrics['Pupil_Max_Baseline'] = baseline_df[norm_col].max(skipna=True)
            trial_metrics['Pupil_Max_Response'] = response_df[norm_col].max(skipna=True)
        else:
            trial_metrics['Pupil_Mean_Baseline'] = np.nan
            trial_metrics['Pupil_Mean_Response'] = np.nan
            trial_metrics['Pupil_Max_Baseline'] = np.nan
            trial_metrics['Pupil_Max_Response'] = np.nan
            
        features.append(trial_metrics)
        
    return pd.DataFrame(features)