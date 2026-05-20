# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:43:28 2026

@author: mitch
"""

import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path

from src.ingestion import load_physio_acq, match_acq_to_edf, load_pupil_edf
from src.ecg_processor import ecg_process
from src.resp_processor import resp_process
from src.pupil_processor import pupil_process

def process_session(acq_filepath, eyelink_dir, include_thermistor = False):
    print(f"--- Processing Session: {Path(acq_filepath).name} ---")
    session_data = {}
    
    # ---------------------------------------------------------
    # 1. INGESTION & SMART CROPPING
    # ---------------------------------------------------------
    print("Ingesting ACQ file...")
    acq_df, acq_fs = load_physio_acq(acq_filepath)
    
    # Find the first TTL pulse, the eyelink starts recording 180.24s before that pulse
    # We crop the .acq from that time to 190s after the start of the last trial
    if acq_df is not None and 'block_sync' in acq_df.columns:
        print("Trimming ACQ file to match EyeLink duration...")
        sync_sig = acq_df['block_sync']
        
        threshold = 3.0
        trigger = (sync_sig >= threshold) & (sync_sig.shift(1) < threshold)
        pulse_indices = sync_sig.index[trigger].tolist()
        
        if pulse_indices:
            pre_buffer_samples = int(180.24 * acq_fs)
            post_buffer_samples = int(190.0 * acq_fs)
            
            start_idx = max(0, pulse_indices[0] - pre_buffer_samples)
            end_idx = min(len(acq_df), pulse_indices[-1] + post_buffer_samples)
            
            original_len = len(acq_df)
            acq_df = acq_df.iloc[start_idx:end_idx].copy()
            acq_df.reset_index(drop=True, inplace=True)
            
            mins_saved = (original_len - len(acq_df)) / acq_fs / 60
            new_dur = len(acq_df) / acq_fs / 60
            print(f"    -> Dropped {mins_saved:.1f} minutes of dead-time.")
            print(f"    -> New ACQ Duration: {new_dur:.1f} minutes (Ready for Processing).")
    
    print("Matching and Ingesting EDF file...")
    edf_filepath = match_acq_to_edf(acq_filepath, eyelink_dir)
    
    if edf_filepath:
        raw_pupil_df, pupil_fs = load_pupil_edf(edf_filepath)
    else:
        print("Warning: No matching EDF found. Proceeding without pupil data.")
        raw_pupil_df, pupil_fs = None, None

    # ---------------------------------------------------------
    # 2. CONTINUOUS PREPROCESSING
    # ---------------------------------------------------------
    if acq_df is not None:
        print("Running ECG Pipeline...")
        ecg_signals, ecg_info = ecg_process(acq_df['ECG'], sampling_rate=acq_fs)
        session_data['ecg'] = ecg_signals
        
        print("Running Respiration Pipeline...")
        resp_signals, resp_info = resp_process(acq_df['Respiration'], sampling_rate=acq_fs)
        session_data['resp'] = resp_signals
        
        if include_thermistor == True:
            print("Running Respiration Pipeline (Thermistor)...")
            therm_signals, therm_info = resp_process(acq_df['Thermistor'], sampling_rate=acq_fs)
            session_data['thermistor'] = therm_signals
        
        if 'block_sync' in acq_df.columns:
            session_data['sync'] = acq_df['block_sync']
            
        session_data['time_acq'] = acq_df['Time']
        session_data['fs_acq'] = acq_fs

    if raw_pupil_df is not None:
        print("Running Pupil Pipeline...")
        pupil_signals, pupil_info = pupil_process(raw_pupil_df['Pupil_Eye_0'], sampling_rate=pupil_fs)
        session_data['pupil'] = pupil_signals
        
        session_data['time_pupil'] = raw_pupil_df['Time']
        session_data['fs_pupil'] = pupil_fs

    # ---------------------------------------------------------
    # 3. ALIGNMENT
    # ---------------------------------------------------------
    print("Aligning multimodal timelines...")
    aligned_session = align_signals(session_data)

    # ---------------------------------------------------------
    # 4. EPOCH EXTRACTION & FINAL DATAFRAME GENERATION
    # ---------------------------------------------------------
    pre_epoch = 60
    post_epoch = 30
    ep_acq, ep_pupil, un_acq, un_pupil, pulse_acq, pulse_pupil = extract_epochs(aligned_session, 
                                                        pre_trigger_sec=pre_epoch, 
                                                        post_trigger_sec=post_epoch)

    # Package the final explicit dataframes
    final_data = {
        'epoched_acq': ep_acq,
        'epoched_pupil': ep_pupil,
        'unepoched_acq': un_acq,
        'unepoched_pupil': un_pupil,
        'acq_sync_index': pulse_acq,
        'pupil_sync_index': pulse_pupil
    }
    
    return final_data

def align_signals(session_data, pre_trial_interval_sec=180.24):
    if 'sync' not in session_data:
        return session_data
        
    sync_sig = session_data['sync']
    time_acq = session_data['time_acq']
    assert len(sync_sig) == len(time_acq), "Error: Time != sync sig len"
    
    threshold = 3.0
    # find all indexes of sync channel where threshold is crossed
    trigger = (sync_sig >= threshold) & (sync_sig.shift(1) < threshold)
    pulse_indices = sync_sig.index[trigger].tolist() 
    
    if not pulse_indices:
        return session_data
        
    session_data['pulse_indices_acq'] = pulse_indices
        
    t_first_pulse_acq = time_acq.iloc[pulse_indices[0]] 
    eyelink_start_acq_time = t_first_pulse_acq - pre_trial_interval_sec
    
    if 'pupil' in session_data and 'time_pupil' in session_data:
        raw_pupil_time = session_data['time_pupil']
        normalized_pupil_time = raw_pupil_time - raw_pupil_time.iloc[0]
        aligned_pupil_time = normalized_pupil_time + eyelink_start_acq_time
        session_data['time_pupil_aligned'] = aligned_pupil_time
        
    return session_data

def extract_epochs(aligned_session, pre_trigger_sec=60.0, post_trigger_sec=30.0, include_thermistor = False):
    print("    -> [Assembly] Epoching multimodal data via Projected Event Markers...")
    
    if 'pulse_indices_acq' not in aligned_session:
        print("    -> [Assembly] Error: No TTL pulses found in session.")
        return None, None, None, None
        
    pulse_indices_acq = aligned_session['pulse_indices_acq']
    time_acq = aligned_session['time_acq'].reset_index(drop=True)
    
    print(f"    -> [Assembly] Using Native ACQ Sampling Rate: {aligned_session['fs_acq']} Hz")
    print(f"Epoch timeseries to {pre_trigger_sec}s pre-stimulus and {post_trigger_sec}s post stimulus")
    t_pulses = time_acq.iloc[pulse_indices_acq].values
    
    # ---------------------------------------------------------
    # A. Build Continuous (Unepoched) ACQ DataFrame
    # ---------------------------------------------------------
    acq_data_dict = {}
    
    # 1. Add ECG
    if isinstance(aligned_session['ecg'], pd.DataFrame):
        acq_data_dict.update(aligned_session['ecg'].to_dict('series'))
    else:
        acq_data_dict['ecg'] = aligned_session['ecg']
        
    # 2. Add Respiration (Pillow) - Keeps standard 'RSP_' prefix
    if isinstance(aligned_session['resp'], pd.DataFrame):
        acq_data_dict.update(aligned_session['resp'].to_dict('series'))
    else:
        acq_data_dict['resp'] = aligned_session['resp']
        
    
    
    # 3. Add Thermistor - RENAME prefix to prevent overwriting
    if include_thermistor == True:
        if isinstance(aligned_session['thermistor'], pd.DataFrame):
            therm_df = aligned_session['thermistor'].copy()
            
            # Find any column containing 'RSP' and replace it with 'Thermistor'
            rename_map = {col: col.replace('RSP', 'Thermistor') for col in therm_df.columns if 'RSP' in col}
            therm_df.rename(columns=rename_map, inplace=True)
            
            acq_data_dict.update(therm_df.to_dict('series'))
        else:
            acq_data_dict['thermistor'] = aligned_session['thermistor']
            
    # Combine all into final unepoched DataFrame
    unepoched_acq = pd.DataFrame(acq_data_dict)
    unepoched_acq['time_acq_absolute'] = time_acq # Append the continuous absolute time
    
    # ---------------------------------------------------------
    # B. Generate Epoched ACQ DataFrame
    # ---------------------------------------------------------
    nk_epochs_acq = nk.epochs_create(
        data=unepoched_acq, 
        events=pulse_indices_acq, 
        sampling_rate=aligned_session['fs_acq'], 
        epochs_start=-pre_trigger_sec, 
        epochs_end=post_trigger_sec
    )
    # Stack the NeuroKit dict into a single Pandas DataFrame
    epoched_acq = nk.epochs_to_df(nk_epochs_acq)
    epoched_acq.rename(columns={'Label': 'Trial'}, inplace=True)
    # these are strings by default, we can change to int
    epoched_acq['Trial'] = epoched_acq['Trial'].astype(int)

    # ---------------------------------------------------------
    # C. Build Continuous (Unepoched) Pupil DataFrame
    # ---------------------------------------------------------
    time_pupil = aligned_session['time_pupil_aligned'].reset_index(drop=True)
    pupil_pulse_indices = [np.argmin(np.abs(time_pupil.values - t)) for t in t_pulses]
    
    pupil_sr = int(np.round(1.0 / np.median(np.diff(time_pupil.values))))
    print(f"    -> [Assembly] Detected Pupil Sampling Rate: {pupil_sr} Hz")
    
    if isinstance(aligned_session['pupil'], pd.DataFrame):
        unepoched_pupil = aligned_session['pupil'].copy()
    else:
        unepoched_pupil = pd.DataFrame({'pupil': aligned_session['pupil']})
        
    unepoched_pupil['time_pupil_absolute'] = time_pupil # Append continuous absolute time
    
    # ---------------------------------------------------------
    # D. Generate Epoched Pupil DataFrame
    # ---------------------------------------------------------
    nk_epochs_pupil = nk.epochs_create(
        data=unepoched_pupil, 
        events=pupil_pulse_indices, 
        sampling_rate=pupil_sr, 
        epochs_start=-pre_trigger_sec, 
        epochs_end=post_trigger_sec
    )
    # Stack the NeuroKit dict into a single Pandas DataFrame
    epoched_pupil = nk.epochs_to_df(nk_epochs_pupil)
    epoched_pupil.rename(columns={'Label': 'Trial'}, inplace=True)
    # these are strings by default, we can change to int
    epoched_pupil['Trial'] = epoched_pupil['Trial'].astype(int)
        
    return epoched_acq, epoched_pupil, unepoched_acq, unepoched_pupil, pulse_indices_acq, pupil_pulse_indices