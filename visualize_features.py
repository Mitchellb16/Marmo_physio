# -*- coding: utf-8 -*-
"""
Execution script for generating physiological feature visualizations.
Run this after feature extraction is complete.
"""
import pandas as pd
import sys
from pathlib import Path

# Path Setup (Ensures we can import from src)
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from src.plotting import plot_feature_correlations, plot_baseline_vs_response

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Visualization Pipeline")
    print("="*50)

    RESULTS_DIR = CURRENT_DIR / 'results'

    try:
        # --- 1. Load the Extracted Features ---
        print("Loading extracted feature CSVs...")
        continuous_df = pd.read_csv(RESULTS_DIR / 'interval_features.csv')
        baseline_df = pd.read_csv(RESULTS_DIR / 'baseline_features.csv')
        stimulus_df = pd.read_csv(RESULTS_DIR / 'stimulus_features.csv')
        
        # Combine baseline and stimulus for the paired slopegraphs
        combined_features = pd.concat([baseline_df, stimulus_df], ignore_index=True)
        
        # Define the core features you want to visualize for your poster
        target_metrics = [
            "ECG_Rate_Mean",
            'HRV_RMSSD',
            'HRV_SDSD',
            'HRV_HF',
            "RSP_Rate_Mean",
            'RRV_RMSSD',
            "Pupil_Mean",
        ]
        
        # --- 2. Generate Correlation Plots ---
        print("\nGenerating Correlation Matrices...")
        
        # A. Pooled (All Subjects)
        print("  -> Plotting Pooled Correlations (All Subjects)")
        try:
            # We pass "All Subjects" so the plotting function can title/save it appropriately
            plot_feature_correlations(continuous_df, target_metrics, subject="All Subjects")
        except TypeError:
            # Fallback just in case your plotting function hasn't been updated to accept a subject argument yet
            plot_feature_correlations(continuous_df, target_metrics)
            
        # B. Subject-Specific Correlation Plots
        if 'Subject' in continuous_df.columns:
            unique_subjects = continuous_df['Subject'].dropna().unique()
            
            for subject_name in unique_subjects:
                print(f"  -> Plotting Correlations for Subject: {subject_name}")
                
                # Isolate data for this specific marmoset
                subject_df = continuous_df[continuous_df['Subject'] == subject_name].reset_index(drop=True)
                
                try:
                    plot_feature_correlations(subject_df, target_metrics, subject=subject_name)
                except TypeError:
                    plot_feature_correlations(subject_df, target_metrics)
        else:
            print("  [!] 'Subject' column not found in continuous_df. Skipping subject-specific plots.")

        # --- 3. Generate Baseline vs Response Plots ---
        print("\nGenerating Baseline vs Response plots...")
        
        for metric in target_metrics:
            try:
                print(f"  -> Plotting {metric}...")
                plot_baseline_vs_response(combined_features, metric=metric, group_col="Label")
            except Exception as e:
                print(f"  [!] Skipping {metric}: Data missing or plotting failed. ({e})")
                
        print("\nVisualization Pipeline Complete! All plots generated.")

    except FileNotFoundError as e:
        print("\n[!] Error: Could not find feature CSVs in the results folder. Please run extraction first.")
        print(f"Details: {e}")
