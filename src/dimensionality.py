#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:46:43 2026

@author: mitchell
"""

# src/dimensionality.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_and_plot_pca(epoched_df):
    """
    Takes the master epoched dataframe, scales the physiological features, 
    runs PCA, and plots the first two principal components.
    """
    print("Preparing data for PCA...")
    
    epoched_df = epoched_df[(epoched_df.Label == 'trill') | (epoched_df.Label == 'tsik')]
    
    
    # 1. Separate your features from your metadata/labels
    # We drop the string/metadata columns to isolate just the math
    metadata_cols = ['Session', 'Subject', 'Epoch_Start', 'Epoch_Length', 'Label', 'Condition']
    
    # Extract only the physiological feature columns
    features = epoched_df.drop(columns=metadata_cols, errors='ignore')
    
    # Extract the labels we will use for color-coding
    labels = epoched_df['Label']
    
    # 2. Scale the Data (CRITICAL STEP)
    print("Scaling features via Z-score...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 3. Run PCA
    print("Fitting Principal Components...")
    pca = PCA(n_components=2) # We only want the first 2 components for a 2D plot
    principal_components = pca.fit_transform(scaled_features)
    
    # 4. Create a new DataFrame for plotting
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Label'] = labels.values # Reattach the labels
# =============================================================================
#     pca_plot_df = pca_df[(pca_df.Label != 'silence') | (pca_df.Label != 'washout')]
# =============================================================================
    # Print out how much variance these two components actually explain
    explained_variance = pca.explained_variance_ratio_
    print(f"\n--- PCA RESULTS ---")
    print(f"Variance explained by PC1: {explained_variance[0]*100:.2f}%")
    print(f"Variance explained by PC2: {explained_variance[1]*100:.2f}%")
    print(f"Total Variance Explained:  {sum(explained_variance)*100:.2f}%\n")
    
    # 5. Plotting
    plt.figure(figsize=(10, 8))
    
    # Use seaborn to automatically color-code (hue) by Label
    sns.scatterplot(
        x='PC1', 
        y='PC2', 
        hue='Label', 
        data=pca_df, 
        palette='Set1', 
        s=100,          
        alpha=0.8       
    )
    
    plt.title('PCA of Marmoset Physiological Arousal (10s Epochs)', fontsize=16, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}%)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}%)', fontsize=14)
    
    # Make the legend look nice
    plt.legend(title='Stimulus Type', title_fontsize='13', fontsize='12', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    return pca, pca_df