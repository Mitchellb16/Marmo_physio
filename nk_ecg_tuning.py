# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:48:57 2026

@author: mitch
"""

# This code was written with help from Gemini AI (Gemini 3.1 Pro).

import sys
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path

# Setup paths to use your existing ingestion and cleaning functions
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))

from src.ingestion import load_physio_acq
from src.ecg_processor import ecg_clean
from neurokit2.signal import signal_smooth, signal_sanitize
import neurokit2 as nk

def compute_neurokit_logic(signal, fs, smoothw, avgw, gradw, minlenw, mindelay_s):
    """The exact mathematical under-the-hood logic from NeuroKit."""
    grad = np.gradient(signal)
    absgrad = np.abs(grad)
    
    # Ensure kernels are at least 1 sample wide
    smooth_kernel = max(1, int(np.rint(smoothw * fs)))
    avg_kernel = max(1, int(np.rint(avgw * fs)))

    smoothgrad = signal_smooth(absgrad, kernel="boxcar", size=smooth_kernel)
    avggrad = signal_smooth(smoothgrad, kernel="boxcar", size=avg_kernel)
    gradthreshold = gradw * avggrad
    mindelay = int(np.rint(fs * mindelay_s))

    qrs = smoothgrad > gradthreshold
    beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]

    if len(beg_qrs) == 0:
        return smoothgrad, gradthreshold, np.array([])

    end_qrs = end_qrs[end_qrs > beg_qrs[0]]
    num_qrs = min(beg_qrs.size, end_qrs.size)
    
    if num_qrs == 0:
        return smoothgrad, gradthreshold, np.array([])

    min_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * minlenw
    peaks = [0]

    for i in range(num_qrs):
        beg = beg_qrs[i]
        end = end_qrs[i]
        len_qrs = end - beg

        if len_qrs < min_len:
            continue

        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            peak = beg + locmax[np.argmax(props["prominences"])]
            if peak - peaks[-1] > mindelay:
                peaks.append(peak)

    peaks.pop(0)
    return smoothgrad, gradthreshold, np.asarray(peaks).astype(int)

if __name__ == "__main__":
    # 1. Load a snippet of Poppy's ACTUAL data
    print("Loading Poppy's data...")
    TEST_FILE = PROJECT_ROOT / 'raw_data' / 'test' / '20260417_Poppy_tsik(ecg_resp).acq'
    
    # We will just grab 4 seconds of data from minute 5 to make it fast
    df, fs = load_physio_acq(str(TEST_FILE))
    start_idx = int(fs * 600) # 10 minutes in
    end_idx = start_idx + int(fs * 4.0) # 4 seconds long
    
    raw_snippet = df['ECG'].iloc[start_idx:end_idx].values
    
    # Apply your standard preprocessing so we are tuning on the exact signal the algorithm will see
    sanitized = signal_sanitize(raw_snippet)
    inverted, _ = nk.ecg_invert(sanitized, sampling_rate=fs)
    cleaned_signal = ecg_clean(inverted, sampling_rate=fs, lowcut=2.5, highcut=450.0)
    time_axis = np.linspace(0, 4.0, len(cleaned_signal))

    # 2. Setup the Matplotlib Figure and Axes
    fig, (ax_sig, ax_grad) = plt.subplots(2, 1, figsize=(12, 8))
    plt.subplots_adjust(bottom=0.4) # Make room for sliders

    # Initial plots
    sig_line, = ax_sig.plot(time_axis, cleaned_signal, lw=1.5, color='black')
    peak_scatter = ax_sig.scatter([], [], color='red', zorder=5)
    ax_sig.set_title("Poppy Cleaned ECG & Detected Peaks")
    
    smooth_line, = ax_grad.plot(time_axis, np.zeros_like(time_axis), label='Smooth Gradient', color='blue')
    thresh_line, = ax_grad.plot(time_axis, np.zeros_like(time_axis), label='Gradient Threshold', color='orange')
    ax_grad.legend(loc='upper right')
    ax_grad.set_title("Under-the-hood: Gradient vs. Threshold")

    # 3. Create the Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_sw = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_aw = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_gw = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_ml = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_md = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

    # Defaults (Human-tuned from NeuroKit)
    sl_sw = Slider(ax_sw, 'Smooth Window (s)', 0.01, 0.15, valinit=0.1)
    sl_aw = Slider(ax_aw, 'Avg Window (s)', 0.05, 1.0, valinit=0.75)
    sl_gw = Slider(ax_gw, 'Grad Thresh Wgt', 0.5, 3.0, valinit=1.5)
    sl_ml = Slider(ax_ml, 'Min Len Wgt', 0.1, 1.0, valinit=0.4)
    sl_md = Slider(ax_md, 'Min Delay (s)', 0.05, 0.50, valinit=0.3)

    # 4. The Update Function (Fires every time a slider moves)
    def update(val):
        sw = sl_sw.val
        aw = sl_aw.val
        gw = sl_gw.val
        ml = sl_ml.val
        md = sl_md.val
        
        smoothgrad, gradthreshold, peaks = compute_neurokit_logic(
            cleaned_signal, fs, sw, aw, gw, ml, md
        )
        
        # Update lines
        smooth_line.set_ydata(smoothgrad)
        thresh_line.set_ydata(gradthreshold)
        
        # Dynamically scale the bottom plot so you can always see the lines
        ax_grad.set_ylim(0, max(np.max(smoothgrad), np.max(gradthreshold)) * 1.1)
        
        # Update scatter points
        if len(peaks) > 0:
            peak_scatter.set_offsets(np.c_[time_axis[peaks], cleaned_signal[peaks]])
        else:
            peak_scatter.set_offsets(np.empty((0, 2)))
            
        fig.canvas.draw_idle()

    # Link sliders to the update function
    sl_sw.on_changed(update)
    sl_aw.on_changed(update)
    sl_gw.on_changed(update)
    sl_ml.on_changed(update)
    sl_md.on_changed(update)

    # Force initial draw
    update(0)
    plt.show()