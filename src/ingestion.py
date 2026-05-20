# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:27:55 2026

@author: mitch
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

import neurokit2 as nk
import pandas as pd
import numpy as np
import re
from pathlib import Path
from eyelinkio import read_edf

def load_physio_acq(file_path):
    """Loads an .acq file using NeuroKit2 and standardizes column names."""
    try:
        data, sampling_rate = nk.read_acqknowledge(str(file_path))
        rename_map = {
            'TSD160B - Differential Pressure, 12.5 c': 'Respiration',
            'ECG2303000390': 'ECG',
            'block_sync': 'block_sync',
            'resp_thermistor - SKT100C': 'Thermistor'
        }
        data.rename(columns=rename_map, inplace=True)
        data['Time'] = data.index / sampling_rate
        return data, sampling_rate
    except Exception as e:
        print(f"Problem loading file {file_path}: {e}")
        return None, None

def match_acq_to_edf(acq_filepath, results_dir):
    """
    Finds the matching .EDF file in the Task/results directory based on the .acq filename.
    """
    acq_name = Path(acq_filepath).name
    
    # 1. Extract Date from .acq (YYYYMMDD) -> YYYY_MM_DD
    date_match = re.search(r'^(\d{8})_', acq_name)
    if not date_match:
        return None
        
    d = date_match.group(1)
    edf_date_format = f"{d[:4]}_{d[4:6]}_{d[6:]}"
    
    # 2. Extract Subject Name
    subject_name = acq_name.split('_')[1].lower()[:4] 
    
    # 3. Search the results directory
    possible_folders = []
    for folder in Path(results_dir).iterdir():
        if folder.is_dir() and edf_date_format in folder.name and subject_name in folder.name.lower():
            possible_folders.append(folder)
            
    # 4. Find the largest .EDF in the matched folders
    best_edf, max_size = None, 0
    for folder in possible_folders:
        for edf in list(folder.glob('*.EDF')) + list(folder.glob('*.edf')):
            if edf.stat().st_size > max_size:
                max_size = edf.stat().st_size
                best_edf = edf
                
    return best_edf

def load_pupil_edf(edf_filepath):
    """
    Loads an EyeLink EDF file using eyelinkio and formats it for the pipeline.
    Returns a DataFrame of the pupil arrays and the sampling rate.
    """
    try:
        edf_file = read_edf(str(edf_filepath))
        fs = edf_file['info']['sfreq']
        samples = edf_file['samples']
        times = edf_file['times']

        # Handle binocular vs monocular using your indexing logic
        if samples.shape[0] >= 6:
            pupil_array = samples[4:6, :] # Binocular (Shape: 2, Time)
            df = pd.DataFrame({
                'Pupil_Eye_0': pupil_array[0, :],
                'Pupil_Eye_1': pupil_array[1, :],
                'Time': times
            })
        else:
            pupil_array = samples[2:, :] # Monocular
            df = pd.DataFrame({
                'Pupil_Eye_0': pupil_array[0, :],
                'Time': times
            })

        return df, fs
    except Exception as e:
        print(f"Error loading EDF {edf_filepath}: {e}")
        return None, None