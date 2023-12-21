#%% Question 3
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import svd, pinv, inv
from sklearn.decomposition import PCA, FastICA
import scipy.io
import random

#%% Imports
ND1 = loadmat('C:/Users/aryak/Downloads/Comp_HW2/Ex3/NewData1.mat')['EEG_Sig']
ND2 = loadmat('C:/Users/aryak/Downloads/Comp_HW2/Ex3/NewData2.mat')['EEG_Sig']
ND3 = loadmat('C:/Users/aryak/Downloads/Comp_HW2/Ex3/NewData3.mat')['EEG_Sig']
ND4 = loadmat('C:/Users/aryak/Downloads/Comp_HW2/Ex3/NewData4.mat')['EEG_Sig']

# Function to plot sources with names separately
def EEGplot(signal, method, snr, denoised=False):
    plt.figure(figsize=(21, 14))
    num_sources = signal.shape[0] 
    
    if method == False and snr != False and denoised==False:
        plt.suptitle(f'Noisy Original Channels (SNR: {snr}dB)')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(signal[j, :], label=f'Ch{j + 1}', c='k')
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f'Ch{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            
    elif method != False and snr != False and denoised==False:
        colors = [
                    'b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink',  # Standard colors: blue, green, red, cyan, magenta, yellow, black
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',  # Tableau Colors
                    '#1a1a1a', '#666666', '#a6a6a6', '#d9d9d9',  # Shades of gray
                    '#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#ff33ff', '#ffff99', '#9966cc', '#ff6666', '#c2c2f0', '#ffb3e6',  # Additional colors
                ]
        plt.suptitle(f'Extracted Sources ({method}, SNR: {snr}dB)')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(signal[j, :], label=f'S{j + 1}', c=colors[j])
            plt.yticks([])
            plt.xticks([])
            plt.ylabel(f'S{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            
    elif method == False and snr == False and denoised==False:
        plt.suptitle(f'Original Channels (without noise)')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(signal[j, :], label=f'Ch{j + 1}', c='k')
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f'Ch{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            
    elif denoised==True:
        plt.suptitle(f'Denoised Channels')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(signal[j, :], label=f'Ch{j + 1}', c='k')
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f'Ch{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            
    plt.tight_layout()
    plt.show();
    
#%%
print("\n=============================== Data 1 ===============================\n")
EEGplot(ND1, False, False, False)
print("\n=============================== Data 2 ===============================\n")
EEGplot(ND2, False, False, False)
print("\n=============================== Data 3 ===============================\n")
EEGplot(ND3, False, False, False)
print("\n=============================== Data 4 ===============================\n")
EEGplot(ND4, False, False, False)