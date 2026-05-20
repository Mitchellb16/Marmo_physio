import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path

def plot_single_trial(acq_df, pupil_df=None, trial_id=None, show_rsp_phase=True, t_min=-2.0, t_max=5.0):
    """
    Generates a clean, aligned multi-channel plot for a single trial epoch.
    Optimized for poster presentations with larger fonts and thicker lines.
    
    Parameters:
    - acq_df: DataFrame containing the 2000Hz AcqKnowledge data for one trial.
    - pupil_df: Optional DataFrame containing the Eyelink data for one trial.
    - trial_id: String or Int to display in the title.
    - show_rsp_phase: Boolean to toggle background shading for respiratory phase.
    - t_min: Float specifying the start time to plot (default -2.0).
    - t_max: Float specifying the end time to plot (default 5.0).
    """
    
    # Filter DataFrames to the specified time window to ensure proper Y-axis autoscaling
    if t_min is not None and t_max is not None:
        acq_df = acq_df[(acq_df['Time'] >= t_min) & (acq_df['Time'] <= t_max)].copy()
        if pupil_df is not None:
            pupil_df = pupil_df[(pupil_df['Time'] >= t_min) & (pupil_df['Time'] <= t_max)].copy()

    # ==========================================
    # Poster Styling Parameters
    # ==========================================
    title_size = 24
    label_size = 20
    tick_size = 16
    legend_size = 16
    line_w = 2.5      
    scatter_s = 100   
    
    n_plots = 3 if pupil_df is not None else 2
    
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(14, 3.5 * n_plots), sharex=True)
    if n_plots == 2: 
        axes = np.append(axes, None)
        
    ax_ecg, ax_rsp, ax_pupil = axes
    t_acq = acq_df['Time'].values

    # ==========================================
    # 1. ECG Subplot
    # ==========================================
    ax_ecg.plot(t_acq, acq_df['ECG_Clean'], color='#E63946', linewidth=line_w, label='ECG')
    
    if 'ECG_R_Peaks' in acq_df.columns:
        peak_idx = acq_df['ECG_R_Peaks'] == 1
        ax_ecg.scatter(t_acq[peak_idx], acq_df['ECG_Clean'][peak_idx], 
                       color='#F4A261', zorder=3, s=scatter_s, label='R-Peaks')
        
    ax_ecg.set_ylabel('Amplitude (mV)', fontsize=label_size)
    ax_ecg.legend(loc='upper right', fontsize=legend_size)
    ax_ecg.set_title("Example Raw signals", 
                     fontsize=title_size, fontweight='bold', pad=15)
    ax_ecg.tick_params(axis='both', labelsize=tick_size)

    # ==========================================
    # 2. Respiration (Belt) Subplot
    # ==========================================
    ax_rsp.plot(t_acq, acq_df['RSP_Clean'], color='#1D3557', linewidth=line_w, label='Resp (Pillow)')
    
    if 'RSP_Peaks' in acq_df.columns:
        peak_idx = acq_df['RSP_Peaks'] == 1
        ax_rsp.scatter(t_acq[peak_idx], acq_df['RSP_Clean'][peak_idx], 
                       color='#E63946', zorder=3, s=scatter_s, label='Inhale Peak')
        
    if show_rsp_phase and 'RSP_Phase' in acq_df.columns:
        insp_mask = acq_df['RSP_Phase'] == 1.0
        ax_rsp.fill_between(t_acq, acq_df['RSP_Clean'].min(), acq_df['RSP_Clean'].max(),
                            where=insp_mask, color='#A8DADC', alpha=0.4, label='Inspiration')

    ax_rsp.set_ylabel('Amplitude', fontsize=label_size)
    ax_rsp.legend(loc='upper right', fontsize=legend_size)
    ax_rsp.tick_params(axis='both', labelsize=tick_size)

    # ==========================================
    # 3. Pupil Subplot (Optional)
    # ==========================================
    if ax_pupil is not None and pupil_df is not None and not pupil_df.empty:
        t_pupil = pupil_df['Time'].values
        ax_pupil.plot(t_pupil, pupil_df['Pupil_Clean'], color='#2B2D42', linewidth=line_w, label='Pupil Diameter')
        ax_pupil.set_ylabel('Arbitrary Units', fontsize=label_size)
        ax_pupil.legend(loc='upper right', fontsize=legend_size)
        ax_pupil.set_xlabel('Time relative to Stimulus (s)', fontsize=label_size, fontweight='bold')
        ax_pupil.tick_params(axis='both', labelsize=tick_size)
    else:
        ax_rsp.set_xlabel('Time relative to Stimulus (s)', fontsize=label_size, fontweight='bold')

    # ==========================================
    # Global Formatting
    # ==========================================
    for ax in axes:
        if ax is not None:
            ax.axvline(x=0.0, color='black', linestyle='--', linewidth=line_w, zorder=1)
            ax.grid(True, alpha=0.2)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # ==========================================
    # ISOLATED TESTING BLOCK (REAL DATA)
    # Loops through all sessions to find the best poster plot.
    # ==========================================
    
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent.parent 
    PICKLE_FILE = PROJECT_ROOT / 'preprocessed_data' / 'master_session_dict.pkl'
    
    if not PICKLE_FILE.exists():
        print(f"Error: Cannot find {PICKLE_FILE}.")
        sys.exit()

    print(f"Loading real data from {PICKLE_FILE.name}...")
    with open(PICKLE_FILE, 'rb') as f:
        master_dict = pickle.load(f)
        
    if not master_dict:
        print("Error: The master dictionary is empty.")
        sys.exit()

    trial_id_to_plot = 3
    
    print("Looping through all sessions. Close the plot window to load the next session...")
    
    for session_name, session_data in master_dict.items():
        print(f"\nEvaluating session: {session_name}")
        
        epoched_acq = session_data.get('epoched_acq')
        epoched_pupil = session_data.get('epoched_pupil')
        
        if epoched_acq is None or epoched_acq.empty:
            print(f"Skipping {session_name} - No ACQ data found.")
            continue
            
        trial_acq = epoched_acq[epoched_acq['Trial'] == trial_id_to_plot]
        
        if trial_acq.empty:
            print(f"Skipping {session_name} - No data for Trial {trial_id_to_plot}.")
            continue

        if epoched_pupil is not None and not epoched_pupil.empty:
            trial_pupil = epoched_pupil[epoched_pupil['Trial'] == trial_id_to_plot]
        else:
            trial_pupil = None

        print(f"Rendering poster-sized plot for {session_name}...")
        fig = plot_single_trial(
            acq_df=trial_acq, 
            pupil_df=trial_pupil, 
            trial_id=f"{session_name} - Trial {trial_id_to_plot}", 
            show_rsp_phase=True,
            t_min=-2.0,
            t_max=5.0
        )
        
        # This will pause the loop until you close the Matplotlib window
        plt.show()
    
    print("\nFinished iterating through all sessions.")