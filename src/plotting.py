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
from pathlib import Path

def plot_feature_correlations(continuous_epochs_df, features, subject="All Subjects"):
    """
    Generates a correlation heatmap for the epoched features, formatted for POSTER presentation
    and exported as an editable vector graphic (SVG).
    """
    # --- POSTER FORMATTING CONSTANTS ---
    TITLE_SIZE = 26
    TICK_SIZE = 18
    ANNOT_SIZE = 16  # Size of the numbers inside the heatmap boxes
    
    # We use a context manager ('with') so this scaling only affects this specific plot
    with sns.axes_style("white"):
        sns.set_context("poster", font_scale=0.8)
        
        feature_cols = features  
        corr_df = continuous_epochs_df[feature_cols]
        
        plt.figure(figsize=(14, 12)) # Slightly larger for poster proportions
        corr_matrix = corr_df.corr()
        
        # Mask the upper triangle to make the heatmap cleaner
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Safely grab the epoch length if it exists in the dataframe
        if 'Epoch_len' in continuous_epochs_df.columns:
            epoch_len = continuous_epochs_df['Epoch_len'].iloc[0]
            bin_text = f", {epoch_len}s bins"
        else:
            bin_text = ""
            
        # --- PLOTTING ---
        ax = sns.heatmap(
            corr_matrix, 
            mask=mask, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f", 
            vmin=-1, 
            vmax=1,
            linewidths=3,         # Thick lines separating the boxes
            linecolor='white',    # Clean white borders between cells
            cbar_kws={'shrink': 0.8}, # Shrink the colorbar slightly so it isn't overwhelming
            annot_kws={"size": ANNOT_SIZE, "fontweight": "bold"} # Massive, bold correlation numbers
        )
        
        # Format the title logic to handle individual vs pooled subjects safely
        if subject == "All Subjects":
            display_name = "All Subjects"
        else:
            display_name = f"Subject_{subject[:3].title()}"
            
        plt.title(f"Cross-Feature Correlation Matrix: {display_name}{bin_text}", 
                  fontsize=TITLE_SIZE, pad=20, fontweight='bold')
        
        # Format the axes for maximum readability
        plt.xticks(rotation=45, ha='right', fontsize=TICK_SIZE, fontweight='bold')
        plt.yticks(rotation=0, fontsize=TICK_SIZE, fontweight='bold')
        
        plt.tight_layout()
        
        # --- VECTOR EXPORT ---
        # Ensure the results directory exists
        out_dir = Path('../results')
        out_dir.mkdir(exist_ok=True)
        
        safe_name = subject.replace(" ", "_")
        out_file = out_dir / f"Correlation_Matrix_{safe_name}.svg"
        
        # bbox_inches='tight' ensures your angled X-axis labels don't get cut off in the export
        plt.savefig(out_file, format='svg', bbox_inches='tight')
        print(f"Saved editable vector plot to: {out_file}")
        
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

def plot_baseline_vs_response(features_df, metric="ECG_Rate_Mean", group_col="Condition"):
    """
    Plots paired slopegraphs over boxplots, optimized for POSTER presentation.
    Uses new Epoch_Type structure and automatically filters by Artifact Inclusion columns.
    """
    # --- POSTER FORMATTING CONSTANTS ---
    TITLE_SIZE = 28
    LABEL_SIZE = 22
    TICK_SIZE = 18
    LEGEND_SIZE = 18
    LINE_WIDTH = 2.5
    MARKER_SIZE = 200 
    
    sns.set_context("poster", font_scale=0.8) 
    sns.set_style("ticks") 

    if metric not in features_df.columns:
        print(f"Error: Could not find '{metric}' in the DataFrame.")
        return
        
    df_to_plot = features_df.copy()
    
    # --- 1. QUALITY CONTROL GATING ---
    # Determine signal name (e.g., "ECG" from "ECG_Rate_Mean") to find the matching QC column
    signal_name = metric.split('_')[0]
    included_col = f"{signal_name}_Included"
    
    if included_col in df_to_plot.columns:
        initial_count = len(df_to_plot)
        df_to_plot = df_to_plot[df_to_plot[included_col] == True]
        dropped = initial_count - len(df_to_plot)
        if dropped > 0:
            print(f"Plotting Note: Filtered out {dropped} epochs due to {signal_name} artifacts.")
            
    # --- 2. FILTER & REFORMAT EPOCH TYPES ---
    if 'Epoch_Type' not in df_to_plot.columns:
        print("Error: 'Epoch_Type' column missing. Make sure you passed the combined results.")
        return
        
    # Isolate only the paired windows
    df_to_plot = df_to_plot[df_to_plot['Epoch_Type'].isin(['baseline', 'stimulus'])].copy()
    
    # Rename them for cleaner plot labels
    df_to_plot['Window'] = df_to_plot['Epoch_Type'].replace({'baseline': 'Baseline', 'stimulus': 'Response'})
    
    # Make sure 'Trial' column exists (if it was an index from reading the CSV)
    if 'Trial' not in df_to_plot.columns and df_to_plot.index.name in ['Trial', 'Event']:
        df_to_plot = df_to_plot.reset_index().rename(columns={df_to_plot.index.name: 'Trial'})
    
    # ANONYMIZE SUBJECTS
    df_to_plot['Subject'] = "Subject_" + df_to_plot['Subject'].astype(str).str[:2].str.title()
    df_to_plot['Unique_Trial'] = df_to_plot['Session'] + "_T" + df_to_plot['Trial'].astype(str)
    
    # Rename metric to 'Value' to match standard plotting logic
    df_to_plot = df_to_plot.rename(columns={metric: 'Value'})
    df_to_plot = df_to_plot.dropna(subset=['Value'])
    
    # --- 3. COMBINED X-AXIS CATEGORIES ---
    df_to_plot['Group_Window'] = df_to_plot[group_col] + "_" + df_to_plot['Window']
    
    # ORDERING: FORCE SILENCE TO FRONT
    unique_groups = df_to_plot[group_col].unique().tolist()
    for silence_term in ['silence', 'Silence']:
        if silence_term in unique_groups:
            unique_groups.remove(silence_term)
            unique_groups.insert(0, silence_term)
            
    order = []
    tick_labels = []
    for grp in unique_groups:
        order.append(f"{grp}_Baseline")
        order.append(f"{grp}_Response")
        tick_labels.append("Baseline")
        tick_labels.append(f"{grp.title()} Stimulus") 
        
    df_to_plot['Group_Window'] = pd.Categorical(df_to_plot['Group_Window'], categories=order, ordered=True)
    df_to_plot = df_to_plot.sort_values('Group_Window')
    
    df_to_plot['Trial'] = df_to_plot['Trial'].astype(int)
    marker_palette = {1: 'o', 2: '^', 3: 's', 4: 'D', 5: 'v', 6: 'p', 7: 'X'}
    present_trials = df_to_plot['Trial'].unique()
    custom_markers = {t: marker_palette.get(t, 'P') for t in present_trials}
    
    # --- PLOTTING ---
    plt.figure(figsize=(14, 8)) 
    
    # Layer 1: Boxplot
    sns.boxplot(
        data=df_to_plot, 
        x='Group_Window', 
        y='Value', 
        color='lightgray',
        boxprops={'alpha': 0.5},
        showfliers=False,
        width=0.4, 
        linewidth=LINE_WIDTH,
        zorder=1
    )
    
    # Layer 2: Connecting Lines
    sns.lineplot(
        data=df_to_plot, 
        x='Group_Window', 
        y='Value', 
        hue='Subject', 
        units='Unique_Trial', 
        estimator=None, 
        alpha=0.4, 
        linewidth=LINE_WIDTH,
        legend=False, 
        zorder=2
    )
    
    # Layer 3: Trial Markers
    sns.scatterplot(
        data=df_to_plot, 
        x='Group_Window', 
        y='Value', 
        hue='Subject', 
        style='Trial', 
        markers=custom_markers, 
        s=MARKER_SIZE,
        alpha=0.9, 
        linewidth=0, 
        zorder=3
    )
    
    # --- FORMATTING & DIVIDERS ---
    for i in range(1, len(unique_groups)):
        plt.axvline(x=i * 2 - 0.5, color='black', linestyle='--', alpha=0.3, linewidth=LINE_WIDTH, zorder=0)
    
    if 'Rate' in metric:
        ylabel = metric + ' (BPM)'
    elif 'Pupil' in metric:
        ylabel = metric + ' (% Max)'
    else:
        ylabel = metric
    
    plt.title(f"{metric}: Baseline vs Response", fontsize=TITLE_SIZE, pad=20, fontweight='bold')
    plt.ylabel(f"{ylabel}", fontsize=LABEL_SIZE, fontweight='bold')
    plt.xlabel("") 
    
    plt.xticks(ticks=range(len(order)), labels=tick_labels, rotation=35, ha='right', fontsize=TICK_SIZE, fontweight='bold')
    plt.yticks(fontsize=TICK_SIZE)
    
    plt.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        borderaxespad=0., 
        fontsize=LEGEND_SIZE, 
        title_fontsize=LABEL_SIZE,
        markerscale=2 
    )
    
    sns.despine()
    plt.tight_layout()
    
    # --- VECTOR EXPORT ---
    # Ensure the results directory exists
    out_dir = Path('../results')
    out_dir.mkdir(exist_ok=True)
    
    out_file = out_dir / f"{metric}response_plot.svg"
    
    # bbox_inches='tight' ensures your angled X-axis labels don't get cut off in the export
    plt.savefig(out_file, format='svg', bbox_inches='tight')
    print(f"Saved editable vector plot to: {out_file}")
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