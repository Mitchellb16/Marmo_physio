# -*- coding: utf-8 -*-
"""
Created on Tue May 19 2026
@author: mitch

Streamlined feature extraction module.
Separates Feature Extraction from Quality Auditing to handle 
differentially sampled ACQ and Pupil data.
"""

import pandas as pd
import neurokit2 as nk
import numpy as np

def safe_extract(nk_function, epochs_dict):
    """
    Safely applies a NeuroKit extraction function to a dictionary of epochs one-by-one.
    If an epoch fails (e.g., too few peaks to compute HRV splines), it catches the error
    and returns an empty DataFrame row for that epoch, preventing full pipeline crashes.
    """
    results = []
    for key, df in epochs_dict.items():
        try:
            # NeuroKit expects a dictionary even for a single epoch
            res = nk_function({key: df})
            results.append(res)
        except Exception as e:
            # The math failed (usually due to lack of peaks). 
            # We append a blank DataFrame with the epoch's key as the index.
            # Pandas will automatically fill this row with NaNs when we concatenate.
            results.append(pd.DataFrame(index=[key]))
            
    # Concatenate all rows. Pandas handles the missing columns automatically.
    return pd.concat(results) if results else pd.DataFrame()


def extract_features(acq_epochs, pupil_epochs, sampling_rate=2000, analysis_type='interval'):
    """
    Extracts physiological features depending on the epoch style.
    Handles ACQ and Pupil data separately to avoid NaN padding from different Hz.
    """
    # 1. Dispatch ACQ to NeuroKit Engines Safely
    if analysis_type in ['interval', 'baseline', 'stimulus']:
        ecg_features = safe_extract(nk.ecg_intervalrelated, acq_epochs)
        rsp_features = safe_extract(nk.rsp_intervalrelated, acq_epochs)
        # add in amplitude
    elif analysis_type == 'peri':
        ecg_features = safe_extract(nk.ecg_eventrelated, acq_epochs)
        rsp_features = safe_extract(nk.rsp_eventrelated, acq_epochs)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")
        
    features_df = pd.concat([ecg_features, rsp_features], axis=1)
    features_df = features_df.loc[:, ~features_df.columns.duplicated()]
    
    # NeuroKit HRV outputs often nest values in arrays e.g., [[60.5]]
    def unpack_value(val):
        if isinstance(val, (list, np.ndarray)):
            try:
                # np.ravel flattens nested lists/arrays of any depth to 1D
                return float(np.ravel(val)[0])
            except (IndexError, TypeError, ValueError):
                return np.nan
        return val
        
    # Apply the unpacking function to every cell in the dataframe
    features_df = features_df.apply(lambda col: col.map(unpack_value))
    # ---------------------------------------------
    
    # 2. Extract Pupil features manually from the parallel pupil dict
    pupil_metrics = {}
    if pupil_epochs:
        for key, df in pupil_epochs.items():
            if 'Pupil_PctMax' in df.columns:
                pupil_metrics[key] = {
                    'Pupil_Mean': df['Pupil_PctMax'].mean(),
                    'Pupil_Max': df['Pupil_PctMax'].max(),
                    'Pupil_SD': df['Pupil_PctMax'].std()
                }
            else:
                pupil_metrics[key] = {'Pupil_Mean': None, 'Pupil_Max': None, 'Pupil_SD': None}
    
    pupil_df = pd.DataFrame.from_dict(pupil_metrics, orient='index')
    
    # Ensure indices match (e.g., Trial 1, Trial 2) before concatenating
    pupil_df.index = features_df.index 
    features_df = pd.concat([features_df, pupil_df], axis=1)
    
    return features_df


def audit_quality(acq_epochs, pupil_epochs, features_df, threshold=0.3):
    """
    Calculates artifact percentages and flags epochs for each signal independently.
    """
    audit_data = {}
    
    # Iterate through keys (assuming ACQ and Pupil epoch dictionaries have matched keys)
    for key in acq_epochs.keys():
        epoch_artifacts = {}
        
        # 1. Pull ACQ Artifacts
        acq_df = acq_epochs[key]
        for col in ['ECG_Artifact', 'RSP_Artifact']:
            if col in acq_df.columns:
                epoch_artifacts[col] = acq_df[col].mean()
                
        # 2. Pull Pupil Artifacts
        if pupil_epochs and key in pupil_epochs:
            pupil_df = pupil_epochs[key]
            if 'Pupil_Artifact' in pupil_df.columns:
                epoch_artifacts['Pupil_Artifact'] = pupil_df['Pupil_Artifact'].mean()
                
        audit_data[key] = epoch_artifacts
            
    audit_df = pd.DataFrame(audit_data).T
    audit_df = audit_df.add_suffix('_Pct')
    audit_df.index = features_df.index
    
    # Merge markers into the feature set
    features_df = pd.concat([features_df, audit_df], axis=1)
    
    # Mark inclusions for each signal individually
    for col in audit_df.columns:
        signal_name = col.replace('_Artifact_Pct', '')
        features_df[f'{signal_name}_Included'] = features_df[col] <= threshold
    
    return features_df