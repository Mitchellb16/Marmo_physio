#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:04:30 2026

@author: mitchell
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_feature_correlations(continuous_epochs_df, subject):
    """
    Generates a correlation heatmap for the epoched features.
    """
    # Select only the numerical feature columns for correlation
    feature_cols = [col for col in continuous_epochs_df.columns if 'Mean' in col or 'SD' in col or 'Max' in col]
    corr_df = continuous_epochs_df[feature_cols]
    
    plt.figure(figsize=(12, 10))
    corr_matrix = corr_df.corr()
    
    # Mask the upper triangle to make the heatmap cleaner
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    epoch_len = continuous_epochs_df['Epoch_len'].iloc[0]
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"Cross-Feature Correlation Matrix {subject}, {epoch_len}s bins", fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_feature_pairplot(continuous_epochs_df):
    """
    Generates a pairplot colored by epoch label to visualize state clustering.
    """
    # Select key numerical features to avoid an overwhelmingly massive plot
    feature_cols = [
        'ECG_Rate_Mean', 'ECG_Rate_SD', 
        'RSP_Rate_Mean', 
        'Pupil_PctMax_Mean', 'Pupil_PctMax_Max'
    ]
    
    # Ensure only available columns are plotted
    available_cols = [col for col in feature_cols if col in continuous_epochs_df.columns]
    
    sns.set_context("talk")
    
    # Define a custom color palette so 'silence' is neutral and 'washout' is distinct
    custom_palette = {
        'silence': '#B0B0B0',  # Light Gray
        'washout': '#E0E0E0',  # Very Light Gray
        'trill': '#1f77b4',    # Blue
        'tsik': '#d62728',     # Red
        'phee': '#2ca02c'      # Green
    }
    
    g = sns.pairplot(
        continuous_epochs_df, 
        vars=available_cols, 
        hue='Label', 
        palette=custom_palette,
        plot_kws={'alpha': 0.7, 'edgecolor': 'k'},
        diag_kind='kde'
    )
    
    g.fig.suptitle("Feature Interactions by Epoch Type", y=1.02, fontsize=20)
    plt.show()

def plot_baseline_vs_response(features_df, metric_prefix="ECG_Rate", group_col="Condition"):
    """
    Plots paired slopegraphs over boxplots, optimized for POSTER presentation.
    Lines connect individual trials, colored by Anonymized Subject.
    """
    # --- POSTER FORMATTING CONSTANTS ---
    # Adjust these numbers if you need the text even larger!
    TITLE_SIZE = 28
    LABEL_SIZE = 22
    TICK_SIZE = 18
    LEGEND_SIZE = 18
    LINE_WIDTH = 2.5
    MARKER_SIZE = 200 # Increased massively for visibility from afar
    
    # Set global seaborn scale for poster (makes default lines and fonts thicker)
    sns.set_context("poster", font_scale=0.8) 
    sns.set_style("ticks") # Keeps a clean white background without gridlines

    bl_col = f"{metric_prefix}_Baseline"
    rsp_col = f"{metric_prefix}_Response"
    
    if bl_col not in features_df.columns or rsp_col not in features_df.columns:
        print(f"Error: Could not find {bl_col} or {rsp_col} in the DataFrame.")
        return
        
    df_to_plot = features_df.copy()
    
    # ANONYMIZE SUBJECTS
    df_to_plot['Subject'] = "Subject_" + df_to_plot['Subject'].astype(str).str[:2].str.title()
    
    # Create unique trial ID
    df_to_plot['Unique_Trial'] = df_to_plot['Session'] + "_T" + df_to_plot['Trial'].astype(str)
    
    melted_df = df_to_plot.melt(
        id_vars=['Unique_Trial', 'Session', 'Trial', 'Subject', group_col], 
        value_vars=[bl_col, rsp_col],
        var_name='Window',
        value_name='Value'
    )
    
    melted_df['Window'] = melted_df['Window'].str.replace(f"{metric_prefix}_", "")
    melted_df = melted_df.dropna(subset=['Value'])
    
    # COMBINED X-AXIS CATEGORIES
    melted_df['Group_Window'] = melted_df[group_col] + "_" + melted_df['Window']
    
    # ORDERING: FORCE SILENCE TO FRONT
    unique_groups = melted_df[group_col].unique().tolist()
    if 'silence' in unique_groups:
        unique_groups.remove('silence')
        unique_groups.insert(0, 'silence')
    elif 'Silence' in unique_groups:
        unique_groups.remove('Silence')
        unique_groups.insert(0, 'Silence')
        
    order = []
    tick_labels = []
    for grp in unique_groups:
        order.append(f"{grp}_Baseline")
        order.append(f"{grp}_Response")
        tick_labels.append("Baseline")
        tick_labels.append(f"{grp.title()} Stimulus") 
        
    melted_df['Group_Window'] = pd.Categorical(melted_df['Group_Window'], categories=order, ordered=True)
    melted_df = melted_df.sort_values('Group_Window')
    
    melted_df['Trial'] = melted_df['Trial'].astype(int)
    marker_palette = {1: 'o', 2: '^', 3: 's', 4: 'D', 5: 'v', 6: 'p', 7: 'X'}
    present_trials = melted_df['Trial'].unique()
    custom_markers = {t: marker_palette.get(t, 'P') for t in present_trials}
    
    # --- PLOTTING ---
    # Increased figure size for high-res poster printing
    plt.figure(figsize=(14, 8)) 
    
    # Layer 1: Boxplot
    sns.boxplot(
        data=melted_df, 
        x='Group_Window', 
        y='Value', 
        color='lightgray',
        boxprops={'alpha': 0.5},
        showfliers=False,
        width=0.4, 
        linewidth=LINE_WIDTH, # Thicker box outlines
        zorder=1
    )
    
    # Layer 2: Connecting Lines
    sns.lineplot(
        data=melted_df, 
        x='Group_Window', 
        y='Value', 
        hue='Subject', 
        units='Unique_Trial', 
        estimator=None, 
        alpha=0.4, 
        linewidth=LINE_WIDTH, # Thicker connecting lines
        legend=False, 
        zorder=2
    )
    
    # Layer 3: Trial Markers
    sns.scatterplot(
        data=melted_df, 
        x='Group_Window', 
        y='Value', 
        hue='Subject', 
        style='Trial', 
        markers=custom_markers, 
        s=MARKER_SIZE, # Massive dots for visibility
        alpha=0.9, 
        linewidth=0,   # Remove outlines on dots so shapes stay crisp
        zorder=3
    )
    
    # --- FORMATTING & DIVIDERS ---
    for i in range(1, len(unique_groups)):
        # Thicker dashed dividers
        plt.axvline(x=i * 2 - 0.5, color='black', linestyle='--', alpha=0.3, linewidth=LINE_WIDTH, zorder=0)
    
    # Apply standard Poster font sizes
    plt.title(f"{metric_prefix}: Baseline vs Response", fontsize=TITLE_SIZE, pad=20, fontweight='bold')
    plt.ylabel(f"{metric_prefix} Value", fontsize=LABEL_SIZE, fontweight='bold')
    plt.xlabel("") 
    
    # Larger, bold X-ticks
    plt.xticks(ticks=range(len(order)), labels=tick_labels, rotation=35, ha='right', fontsize=TICK_SIZE, fontweight='bold')
    plt.yticks(fontsize=TICK_SIZE)
    
    # Configure the legend for poster sizing
    plt.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0., 
        fontsize=LEGEND_SIZE, 
        title_fontsize=LABEL_SIZE,
        markerscale=2 # Makes the legend markers larger too
    )
    
    # Despine removes the top and right borders for a cleaner look
    sns.despine()
    
    # tight_layout() ensures nothing gets cut off when you save the high-res image
    plt.tight_layout()
    plt.show()

def parse_condition_title(filename):
    """
    Extracts the subject and condition from the AcqKnowledge filename.
    Example: '20260416_Freddy_silence(ecg_resp).acq' -> 'Freddy - Silence'
    """
    try:
        parts = filename.split('_')
        subject = parts[1].title()
        # Grab the 'silence(ecg_resp).acq' part, split at '(', take the first half
        condition = parts[2].split('(')[0].title()
        return f"{subject}: {condition}"
    except Exception:
        # Fallback if the filename format is unexpected
        return filename

def plot_full_session(session_data, filename):
    """
    Generates a full-session overview of Heart Rate, Respiratory Rate, and Pupil Diameter.
    Overlays a shaded region for the 10-second duration of every trial.
    """
    title = parse_condition_title(filename)
    print(f"Generating full-session plot for: {title}")
    
    has_pupil = 'pupil' in session_data and session_data['pupil'] is not None
    num_plots = 3 if has_pupil else 2
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 4 * num_plots), sharex=True)
    
    # Ensure axes is iterable even if there are only 2 plots
    if num_plots == 2:
        axes = np.array(axes)
        
    t_acq = session_data['time_acq']
    
    # Extract stimulus onset times
    pulse_times = []
    if 'pulse_indices_acq' in session_data:
        pulse_indices = session_data['pulse_indices_acq']
        pulse_times = [t_acq.iloc[idx] for idx in pulse_indices]

    def shade_trials(ax):
        """Helper to draw the 10-second stimulus windows."""
        for i, pt in enumerate(pulse_times):
            # Only add the label to the legend once
            label = 'Stimulus (10s)' if i == 0 else ""
            ax.axvspan(pt, pt + 10.0, color='gold', alpha=0.25, label=label)

    # --- 1. HEART RATE ---
    ax_hr = axes[0]
    ax_hr.plot(t_acq, session_data['ecg']['ECG_Rate'], color='firebrick', linewidth=1.5)
    ax_hr.set_title("Heart Rate (Continuous)", fontsize=13, fontweight='bold')
    ax_hr.set_ylabel("BPM", fontsize=11)
    ax_hr.grid(True, linestyle='--', alpha=0.5)
    shade_trials(ax_hr)
    if pulse_times:
        ax_hr.legend(loc='upper right')

    # --- 2. RESPIRATORY RATE ---
    ax_rr = axes[1]
    ax_rr.plot(t_acq, session_data['resp']['RSP_Rate'], color='seagreen', linewidth=1.5, label='Pillow')
    
    if 'thermistor' in session_data:
        ax_rr.plot(t_acq, session_data['thermistor']['RSP_Rate'], color='mediumspringgreen', 
                   linewidth=1.5, linestyle='--', alpha=0.8, label='Thermistor (Airflow)')
        ax_rr.legend(loc='upper right')
        
    ax_rr.set_title("Respiratory Rate (Continuous)", fontsize=13, fontweight='bold')
    ax_rr.set_ylabel("Breaths / min", fontsize=11)
    ax_rr.grid(True, linestyle='--', alpha=0.5)
    shade_trials(ax_rr)

    # --- 3. PUPILLOMETRY ---
    if has_pupil:
        ax_pupil = axes[2]
        t_pupil = session_data['time_pupil_aligned']
        
        # Safely grab the correct pupil column
        pupil_df = session_data['pupil']
        pupil_col = 'Pupil_Avg_Clean' if 'Pupil_Avg_Clean' in pupil_df.columns else [c for c in pupil_df.columns if 'Clean' in c][0]
        
        ax_pupil.plot(t_pupil, pupil_df[pupil_col], color='dodgerblue', linewidth=1.5)
        ax_pupil.set_title("Pupil Diameter", fontsize=13, fontweight='bold')
        ax_pupil.set_ylabel("Size (a.u.)", fontsize=11)
        ax_pupil.grid(True, linestyle='--', alpha=0.5)
        shade_trials(ax_pupil)

    axes[-1].set_xlabel('Experiment Time (seconds)', fontsize=12)
    
    # Add a master title to the entire figure
    fig.suptitle(f"Full Session Overview | {title}", fontsize=16, fontweight='black', y=0.98)
    
    plt.tight_layout()
    # Adjust layout to make room for the master title
    plt.subplots_adjust(top=0.93) 
    plt.show()
    
def plot_single_trial(session_data, filename, trial_key='trial_0'):
    """
    Generates a zoomed-in plot of Heart Rate, Resp Rate, Resp Amplitude, 
    and Pupil Diameter for a specific trial.
    """
    if 'epochs' not in session_data or trial_key not in session_data['epochs']:
        print(f"    -> Cannot plot {trial_key}: No epoch data found.")
        return
        
    trial_data = session_data['epochs'][trial_key]
    title = parse_condition_title(filename)
    formatted_trial_name = trial_key.replace('_', ' ').title()
    print(f"Generating single trial plot for: {title} ({formatted_trial_name})")
    
    has_pupil = 'pupil' in trial_data and trial_data['pupil'] is not None
    num_plots = 4 if has_pupil else 3
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)
    
    t_acq = trial_data['time_acq']
    
    # Helper to draw the stimulus onset line
    def mark_onset(ax):
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Stimulus Onset (t=0)')
        ax.grid(True, linestyle='--', alpha=0.5)

    # --- 1. HEART RATE ---
    ax_hr = axes[0]
    ax_hr.plot(t_acq, trial_data['ecg']['ECG_Rate'], color='firebrick', linewidth=2)
    ax_hr.set_title("Heart Rate", fontsize=12, fontweight='bold')
    ax_hr.set_ylabel("BPM")
    mark_onset(ax_hr)
    ax_hr.legend(loc='upper right')

    # --- 2. RESPIRATORY RATE ---
    ax_rr = axes[1]
    ax_rr.plot(t_acq, trial_data['resp']['RSP_Rate'], color='seagreen', linewidth=2, label='Pillow')
    if 'thermistor' in trial_data:
        ax_rr.plot(t_acq, trial_data['thermistor']['RSP_Rate'], color='mediumspringgreen', 
                   linewidth=2, linestyle='--', alpha=0.8, label='Thermistor')
        ax_rr.legend(loc='upper right')
        
    ax_rr.set_title("Respiratory Rate", fontsize=12, fontweight='bold')
    ax_rr.set_ylabel("Breaths / min")
    mark_onset(ax_rr)

    # --- 3. RESPIRATORY AMPLITUDE ---
    ax_ra = axes[2]
    ax_ra.plot(t_acq, trial_data['resp']['RSP_Amplitude'], color='mediumseagreen', linewidth=2, label='Pillow')
    if 'thermistor' in trial_data:
        ax_ra.plot(t_acq, trial_data['thermistor']['RSP_Amplitude'], color='palegreen', 
                   linewidth=2, linestyle='--', alpha=0.8, label='Thermistor')
        ax_ra.legend(loc='upper right')
        
    ax_ra.set_title("Respiratory Amplitude", fontsize=12, fontweight='bold')
    ax_ra.set_ylabel("Depth (a.u.)")
    mark_onset(ax_ra)

    # --- 4. PUPILLOMETRY ---
    if has_pupil:
        ax_pupil = axes[3]
        t_pupil = trial_data['time_pupil']
        
        pupil_df = trial_data['pupil']
        pupil_col = 'Pupil_Avg_Clean' if 'Pupil_Avg_Clean' in pupil_df.columns else [c for c in pupil_df.columns if 'Clean' in c][0]
        
        ax_pupil.plot(t_pupil, pupil_df[pupil_col], color='dodgerblue', linewidth=2)
        ax_pupil.set_title("Pupil Diameter", fontsize=12, fontweight='bold')
        ax_pupil.set_ylabel("Size (a.u.)")
        mark_onset(ax_pupil)

    axes[-1].set_xlabel('Time relative to Stimulus (seconds)', fontsize=12)
    
    # Set X-limits to match our extracted epoch window perfectly
    plt.xlim(t_acq.min(), t_acq.max()) 
    
    fig.suptitle(f"{formatted_trial_name} Zoom | {title}", fontsize=15, fontweight='black', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) 
    plt.show()