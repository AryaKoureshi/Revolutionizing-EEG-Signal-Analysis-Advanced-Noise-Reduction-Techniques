'''
Revolutionizing EEG Signal Analysis: A Dive into Advanced Noise Reduction Techniques

Arya Koureshi (arya.koureshi@gmail.com)

arya.koureshi@gmail.com

'''

#%% Question 1
#%% Imports
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import warnings
warnings.filterwarnings("ignore")

#%% Load MATLAB data
fs = 200 #Hz
signal = loadmat('C:/Users/aryak/Downloads/Comp_HW2/Ex1.mat')['EEG_Sig']
t = np.arange(0, len(signal[0])/200, 1/fs)

#%% part a
color = "brg"
for i in range(len(signal)):
    plt.figure(figsize=(20, 3))
    plt.plot(t, signal[i], c=color[i], label="Channel: {}".format(i+1))
    plt.xlim([0, t[-1]])
    plt.ylim([-10, 10])
    plt.xlabel("Time (sec)")
    plt.ylabel("Amp")
    plt.tight_layout()
    plt.legend()
    plt.show()
    
#%% part b 
plt.figure(figsize=(15, 15))
ax = plt.axes(projection ="3d")

ax.scatter3D(signal[0, :], signal[1, :], signal[2, :], c='k', marker='.')

ax.set_xlabel('Channel 1')
ax.set_ylabel('Channel 2')
ax.set_zlabel('Channel 3')
plt.title('3D Scatter Plot of EEG Data')
ax.set_ylim([-10, 10])
ax.set_xlim([-10, 10])
ax.set_zlim([-10, 10])
plt.tight_layout()
plt.show()

#%% part c
# Apply PCA algorithm
cov_matrix = np.cov(signal)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project data onto principal components
pca_data = np.dot(eigenvectors.T, signal)

# Whitening and normalization
whitened_data = np.dot(np.linalg.inv(np.diag(np.sqrt(eigenvalues))), pca_data)

# Plot whitened data as channels in time
plt.figure(figsize=(20, 3))
plt.title('Whitened Data: Channels in Time')
plt.plot(t, whitened_data[0], c="b", label="Channel: 1")
plt.tight_layout()
plt.xlabel("Time (sec)")
plt.ylabel("Amp")
plt.xlim([0, t[-1]])
plt.ylim([-10, 10])
plt.legend()

plt.figure(figsize=(20, 3))
plt.plot(t, whitened_data[1], c="r", label="Channel: 2")
plt.tight_layout()
plt.xlabel("Time (sec)")
plt.ylabel("Amp")
plt.xlim([0, t[-1]])
plt.ylim([-10, 10])
plt.legend()

plt.figure(figsize=(20, 3))
plt.plot(t, whitened_data[2], c="g", label="Channel: 3")
plt.xlim([0, t[-1]])
plt.ylim([-10, 10])
plt.xlabel("Time (sec)")
plt.ylabel("Amp")
plt.legend()
plt.tight_layout()
plt.show()

# Plot whitened data as a 3D shape
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(whitened_data[0], whitened_data[1], whitened_data[2], c='k', marker='.')
ax.set_title('Whitened Data: 3D Shape')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_ylim([-10, 10])
ax.set_xlim([-10, 10])
ax.set_zlim([-10, 10])
plt.tight_layout()
plt.show()

#%% part d: whiten: True
pca_sklearn = PCA(whiten=True)
pca_sklearn.fit(signal.T)
pca_data_sklearn = pca_sklearn.transform(signal.T).T

plt.figure(figsize=(20, 3))
plt.title('Sklearn PCA Whitened Data: Channels in Time')
plt.plot(t, pca_data_sklearn[0], c="b", label="Channel: 1")
plt.tight_layout()
plt.xlabel("Time (sec)")
plt.ylabel("Amp")
plt.xlim([0, t[-1]])
plt.ylim([-10, 10])
plt.legend()

plt.figure(figsize=(20, 3))
plt.plot(t, pca_data_sklearn[1], c="r", label="Channel: 2")
plt.tight_layout()
plt.xlabel("Time (sec)")
plt.ylabel("Amp")
plt.xlim([0, t[-1]])
plt.ylim([-10, 10])
plt.legend()

plt.figure(figsize=(20, 3))
plt.plot(t, pca_data_sklearn[2], c="g", label="Channel: 3")
plt.xlim([0, t[-1]])
plt.ylim([-10, 10])
plt.xlabel("Time (sec)")
plt.ylabel("Amp")
plt.legend()
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(pca_data_sklearn[0], pca_data_sklearn[1], pca_data_sklearn[2], c='k', marker='.')
ax.set_title('Sklearn PCA Whitened Data: 3D Shape')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_ylim([-10, 10])
ax.set_xlim([-10, 10])
ax.set_zlim([-10, 10])
plt.show()

#%% part d: whiten: False
pca_sklearn = PCA(whiten=False)
pca_sklearn.fit(signal.T)
pca_data_sklearn = pca_sklearn.transform(signal.T).T

plt.figure(figsize=(20, 3))
plt.title('Sklearn PCA Data: Channels in Time')
plt.plot(t, pca_data_sklearn[0], c="b", label="Channel: 1")
plt.tight_layout()
plt.xlabel("Time (sec)")
plt.ylabel("Amp")
plt.xlim([0, t[-1]])
plt.ylim([-10, 10])
plt.legend()

plt.figure(figsize=(20, 3))
plt.plot(t, pca_data_sklearn[1], c="r", label="Channel: 2")
plt.tight_layout()
plt.xlabel("Time (sec)")
plt.ylabel("Amp")
plt.xlim([0, t[-1]])
plt.ylim([-10, 10])
plt.legend()

plt.figure(figsize=(20, 3))
plt.plot(t, pca_data_sklearn[2], c="g", label="Channel: 3")
plt.xlim([0, t[-1]])
plt.ylim([-10, 10])
plt.xlabel("Time (sec)")
plt.ylabel("Amp")
plt.legend()
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(pca_data_sklearn[0], pca_data_sklearn[1], pca_data_sklearn[2], c='k', marker='.')
ax.set_title('Sklearn PCA Data: 3D Shape')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_ylim([-10, 10])
ax.set_xlim([-10, 10])
ax.set_zlim([-10, 10])
plt.show()

#%% part e
num_channels, num_samples = signal.shape

# Center the data (subtract mean)
mean_data = np.mean(signal, axis=1, keepdims=True)
centered_data = signal - mean_data

# Apply SVD analysis
U, S, Vt = np.linalg.svd(centered_data)

# Calculate the explained variance
explained_variance = (S ** 2) / (num_samples - 1)
total_variance = np.sum(explained_variance)
explained_variance_ratio = explained_variance / total_variance

# Print the singular values and explained variance ratio
print("Singular Values:")
print(S)
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)

# Visualize the singular values and explained variance
# Plot singular values
plt.figure(figsize=(7, 4))
plt.plot(S, 'bo-', label='Singular Values')
plt.title("Singular Values")
plt.xlabel("Singular Value Index")
plt.ylabel("Singular Value")

# Plot explained variance ratio
plt.figure(figsize=(7, 4))
plt.plot(np.cumsum(explained_variance_ratio), marker='o')
plt.title("Cumulative Explained Variance Ratio")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance Ratio")

plt.figure(figsize=(7, 4))
signal = loadmat('C:/Users/aryak/Downloads/Comp_HW2/Ex1.mat')['EEG_Sig']
u, s, vh = np.linalg.svd(signal)
variance_data = np.var(signal, axis=1)
variance_whitened = np.var(np.dot(u.T, signal), axis=1)
plt.bar(range(1, 4), variance_data, alpha=0.6, label='Variance of Data')
plt.bar(range(1, 4), variance_whitened, alpha=0.6, label='Variance of Whitened Data', color='r')
plt.xlabel('Channel')
plt.ylabel('Variance')
plt.title('Variance Comparison')
plt.legend()

plt.tight_layout()
plt.show()

#%% Question 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import svd, pinv, inv
from sklearn.decomposition import PCA, FastICA
import scipy.io
import random

# Load MATLAB data
mat_data = loadmat('C:/Users/aryak/Downloads/Comp_HW2/Ex2/Ex2.mat')
X_org = mat_data['X_org']
X_noise_1 = mat_data['X_noise_1']
X_noise_2 = mat_data['X_noise_2']
X_noise_3 = mat_data['X_noise_3']
X_noise_4 = mat_data['X_noise_4']
X_noise_5 = mat_data['X_noise_5']
X_noises = [X_noise_1, X_noise_2, X_noise_3, X_noise_4, X_noise_5]

# Function to plot the original and extracted sources
def plot_sources(original, extracted, method, snr):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(original.T)
    plt.title(f'Original Signal (SNR: {snr}dB)')
    plt.subplot(2, 1, 2)
    plt.plot(extracted.T)
    plt.title(f'Extracted Sources using {method} (SNR: {snr}dB)')
    plt.tight_layout()
    plt.show()
    
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

def add_noise(original_signal, noise, snr_db):
    noise_power = np.sum(noise**2) / len(noise)
    signal_power = np.sum(original_signal**2) / len(original_signal)
    noise_factor = np.sqrt((signal_power / noise_power) * 10**(-snr_db / 10))
    noisy_signal = original_signal + noise_factor * noise
    return noisy_signal

#%% Question 2 - part a
# a) Add noise with different SNRs (-10dB and -20dB) to the original signal
SNRs = [-10, -20] #dB

first_noise = X_noises[random.randint(0, 2)]
second_noise = X_noises[random.randint(3, 4)]

for snr_db in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)

    plt.figure(figsize=(20, 3))
    plt.plot(X_noisy_1[12, :], label=f'Noisy Signal (SNR={snr_db}dB)', c='r')
    plt.plot(X_org[12, :], label='Original Signal', lw=2, c='b')
    plt.title(f'Signal Comparison for Channel 13 (SNR={snr_db}dB)')
    plt.legend()
    plt.xlim([0, len(X_org[0])])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 3))
    plt.plot(X_noisy_1[23, :], label=f'Noisy Signal (SNR={snr_db}dB)', c='r')
    plt.plot(X_org[23, :], label='Original Signal', lw=2, c='b')
    plt.title(f'Signal Comparison for Channel 24 (SNR={snr_db}dB)')
    plt.legend()
    plt.xlim([0, len(X_org[0])])
    plt.tight_layout()
    plt.show()
    
#%% part b
# Function to perform PCA and ICA
def perform_source_extraction(signal, method='pca'):
    if method == 'pca':
        model = PCA(n_components=32)
        sources = model.fit_transform(signal.T).T
    elif method == 'ica':
        model = FastICA()
        sources = model.fit_transform(signal.T).T
        unmixing_matrix = model.mixing_
    
    return sources , unmixing_matrix if method == 'ica' else None, model

for snr in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr)
    X_noisy_2 = add_noise(X_org, second_noise, snr)
    
    print("=================================================================================================================================")
    print(f"\n======================================================== SNR: {snr}dB ========================================================\n")
    print("=================================================================================================================================")
    print("\n======================================================== Original Signal (without noise) ========================================================\n")
    EEGplot(X_org, False, False)
    
    print("\n======================================================== First Noisy Signal ========================================================\n")
    EEGplot(X_noisy_1, False, snr)
    print("\n======================================================== First Noisy Signal (PCA & ICA) ========================================================\n")
    sources_pca_1, unmixing_matrix_pca_1, _ = perform_source_extraction(X_noisy_1, method='pca') # Extract sources using PCA
    sources_ica_1, unmixing_matrix_ica_1, _ = perform_source_extraction(X_noisy_1, method='ica') # Extract sources using ICA
    EEGplot(sources_pca_1, 'PCA', snr)
    EEGplot(sources_ica_1, 'ICA', snr)
    
    print("\n======================================================== Second Noisy Signal ========================================================\n")
    EEGplot(X_noisy_2, False, snr)
    print("\n======================================================== Second Noisy Signal (PCA & ICA) ========================================================\n")
    sources_pca_2, unmixing_matrix_pca_2, _ = perform_source_extraction(X_noisy_2, method='pca') # Extract sources using PCA
    sources_ica_2, unmixing_matrix_ica_2, _ = perform_source_extraction(X_noisy_2, method='ica') # Extract sources using ICA
    EEGplot(sources_pca_2, 'PCA', snr)
    EEGplot(sources_ica_2, 'ICA', snr)
    
#%% Part c: Identifying and Keeping Desired (Spiky) Resources
# Part c and d - with kurtosis

from scipy.stats import kurtosis

def identify_spiky_sources(sources, threshold):
    """
    Identifies spiky sources based on kurtosis.

    :param sources: numpy array containing the sources (each row is a source)
    :param threshold: threshold value for kurtosis
    :return: numpy array containing only the identified spiky sources
    """
    spiky_sources = []
    for source in sources:
        if kurtosis(source) > threshold:
            spiky_sources.append(source)
    
    return np.array(spiky_sources)

def evaluate_spiky_sources(spiky_sources):
    """
    Evaluates the spiky sources based on their number and average kurtosis.

    :param spiky_sources: numpy array of the identified spiky sources
    :return: score representing the quality of spiky sources
    """
    if len(spiky_sources) == 0:
        return 0

    average_kurtosis = np.mean([kurtosis(source) for source in spiky_sources])
    score = len(spiky_sources) * average_kurtosis

    return score

def normalize_sources(sources):
    normalized_sources = np.copy(sources)
    for i in range(sources.shape[0]):
        max_val = np.max(np.abs(sources[i, :]))
        if max_val != 0:
            normalized_sources[i, :] /= max_val
    return normalized_sources

def optimize_threshold(sources, min_threshold, max_threshold, step):
    optimal_threshold = min_threshold
    best_score = 0

    for threshold in np.arange(min_threshold, max_threshold, step):
        spiky_sources = identify_spiky_sources(sources, threshold)
        score = evaluate_spiky_sources(spiky_sources)

        if score > best_score:
            best_score = score
            optimal_threshold = threshold

    return optimal_threshold, best_score

min_th = 10
max_th = 0
step_th = -0.01

for snr in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)
    
    print(f"\n=========================================== SNR: {snr}dB ===========================================\n")
    sources_pca_1, unmixing_matrix_pca_1, _ = perform_source_extraction(X_noisy_1, method='pca') # Extract sources using PCA
    sources_ica_1, unmixing_matrix_ica_1, _ = perform_source_extraction(X_noisy_1, method='ica') # Extract sources using ICA
    sources_pca_2, unmixing_matrix_pca_2, _ = perform_source_extraction(X_noisy_2, method='pca') # Extract sources using PCA
    sources_ica_2, unmixing_matrix_ica_2, _ = perform_source_extraction(X_noisy_2, method='ica') # Extract sources using ICA

    # Normalize the sources
    normalized_sources_pca_1 = normalize_sources(sources_pca_1)
    normalized_sources_ica_1 = normalize_sources(sources_ica_1)
    normalized_sources_pca_2 = normalize_sources(sources_pca_2)
    normalized_sources_ica_2 = normalize_sources(sources_ica_2)
    
    # Optimize the threshold
    optimal_threshold_pca_1, best_score_pca_1 = optimize_threshold(normalized_sources_pca_1, min_th, max_th, step_th)
    optimal_threshold_ica_1, best_score_ica_1 = optimize_threshold(normalized_sources_ica_1, min_th, max_th, step_th)
    optimal_threshold_pca_2, best_score_pca_2 = optimize_threshold(normalized_sources_pca_2, min_th, max_th, step_th)
    optimal_threshold_ica_2, best_score_ica_2 = optimize_threshold(normalized_sources_ica_2, min_th, max_th, step_th)
    
    # Identify spiky sources using the optimal thresholds
    spiky_sources_pca_1 = identify_spiky_sources(normalized_sources_pca_1, optimal_threshold_pca_1)
    spiky_sources_ica_1 = identify_spiky_sources(normalized_sources_ica_1, optimal_threshold_ica_1)
    spiky_sources_pca_2 = identify_spiky_sources(normalized_sources_pca_2, optimal_threshold_pca_2)
    spiky_sources_ica_2 = identify_spiky_sources(normalized_sources_ica_2, optimal_threshold_ica_2)

    # Visualization can be done using the EEGplot function
    print(f"\n=========================================== Spiky Sources from First Noisy Signal with SNR: {snr}dB (PCA) ===========================================\n")
    EEGplot(spiky_sources_pca_1, 'PCA - Spiky', snr)
    print(f"\n Optimal Threshold = {optimal_threshold_pca_1} \n")
    print(f"\n Best Score = {best_score_pca_1} \n")

    print(f"\n=========================================== Spiky Sources from First Noisy Signal with SNR: {snr}dB (ICA) ===========================================\n")
    EEGplot(spiky_sources_ica_1, 'ICA - Spiky', snr)
    print(f"\n Optimal Threshold = {optimal_threshold_ica_1} \n")
    print(f"\n Best Score = {best_score_ica_1} \n")

    print(f"\n=========================================== Spiky Sources from Second Noisy Signal with SNR: {snr}dB (PCA) ===========================================\n")
    EEGplot(spiky_sources_pca_2, 'PCA - Spiky', snr)
    print(f"\n Optimal Threshold = {optimal_threshold_pca_2} \n")
    print(f"\n Best Score = {best_score_pca_2} \n")

    print(f"\n=========================================== Spiky Sources from Second Noisy Signal with SNR: {snr}dB (ICA) ===========================================\n")
    EEGplot(spiky_sources_ica_2, 'ICA - Spiky', snr)
    print(f"\n Optimal Threshold = {optimal_threshold_ica_2} \n")
    print(f"\n Best Score = {best_score_ica_2} \n")


#%% Part c: Identifying and Keeping Desired (Spiky) Resources
# part c & d - with normilizing and number of max spikes
def identify_spiky_sources2(sources, threshold, maxspikes):
    """
    Identifies spiky sources based on kurtosis.

    :param sources: numpy array containing the sources (each row is a source)
    :param threshold: threshold value for kurtosis
    :return: numpy array containing only the identified spiky sources
    """
    spiky_sources = []
    index = []
    inx = 0
    for source in sources:
        flag = 0
        cnt = 0
        for i in range(len(source)):
            if source[i] >= threshold and flag == 0:
                cnt += 1
                flag = 1
            if source[i] < threshold and flag == 1:
                flag = 0
        if cnt <= maxspikes and cnt != 0:
            #print(str(inx) + " :  " + str(cnt))
            spiky_sources.append(source)
            index.append(inx)
        inx += 1
    return np.array(spiky_sources), index

def optimize_threshold2(sources, min_threshold, max_threshold, step, maxspikes, minsources):
    optimal_threshold = min_threshold
    best_score = len(sources)
    
    for threshold in np.arange(min_threshold, max_threshold, step):
        spiky_sources, index = identify_spiky_sources2(sources, threshold, maxspikes)
        score = len(spiky_sources)
        
        if minsources < score < best_score and optimal_threshold < threshold:
            best_score = score
            optimal_threshold = threshold

    return optimal_threshold, best_score

min_th = 0.2
max_th = 0.9
step_th = 0.01
maxspikes = 10
minsources = 3

for snr in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)
    
    print(f"\n=========================================== SNR: {snr}dB ===========================================\n")
    sources_pca_1, unmixing_matrix_pca_1, _ = perform_source_extraction(X_noisy_1, method='pca') # Extract sources using PCA
    sources_ica_1, unmixing_matrix_ica_1, _ = perform_source_extraction(X_noisy_1, method='ica') # Extract sources using ICA
    sources_pca_2, unmixing_matrix_pca_2, _ = perform_source_extraction(X_noisy_2, method='pca') # Extract sources using PCA
    sources_ica_2, unmixing_matrix_ica_2, _ = perform_source_extraction(X_noisy_2, method='ica') # Extract sources using ICA

    # Normalize the sources
    normalized_sources_pca_1 = normalize_sources(sources_pca_1)
    normalized_sources_ica_1 = normalize_sources(sources_ica_1)
    normalized_sources_pca_2 = normalize_sources(sources_pca_2)
    normalized_sources_ica_2 = normalize_sources(sources_ica_2)
    
    # Optimize the threshold
    optimal_threshold_pca_1, best_score_pca_1 = optimize_threshold2(normalized_sources_pca_1, min_th, max_th, step_th, maxspikes, minsources)
    optimal_threshold_ica_1, best_score_ica_1 = optimize_threshold2(normalized_sources_ica_1, min_th, max_th, step_th, maxspikes, minsources)
    optimal_threshold_pca_2, best_score_pca_2 = optimize_threshold2(normalized_sources_pca_2, min_th, max_th, step_th, maxspikes, minsources)
    optimal_threshold_ica_2, best_score_ica_2 = optimize_threshold2(normalized_sources_ica_2, min_th, max_th, step_th, maxspikes, minsources)
    
    # Identify spiky sources using the optimal thresholds
    spiky_sources_pca_1, index_pca_1 = identify_spiky_sources2(normalized_sources_pca_1, optimal_threshold_pca_1, maxspikes)
    spiky_sources_ica_1, index_ica_1 = identify_spiky_sources2(normalized_sources_ica_1, optimal_threshold_ica_1, maxspikes)
    spiky_sources_pca_2, index_pca_2 = identify_spiky_sources2(normalized_sources_pca_2, optimal_threshold_pca_2, maxspikes)
    spiky_sources_ica_2, index_ica_2 = identify_spiky_sources2(normalized_sources_ica_2, optimal_threshold_ica_2, maxspikes)

    # Visualization can be done using the EEGplot function
    print(f"\n=========================================== Spiky Sources from First Noisy Signal with SNR: {snr}dB (PCA) ===========================================\n")
    EEGplot(spiky_sources_pca_1, 'PCA - Spiky', snr)
    print(f"\n Optimal Threshold = {optimal_threshold_pca_1} \n")
    print(f"\n Number of selected sources = {best_score_pca_1} \n")

    print(f"\n=========================================== Spiky Sources from First Noisy Signal with SNR: {snr}dB (ICA) ===========================================\n")
    EEGplot(spiky_sources_ica_1, 'ICA - Spiky', snr)
    print(f"\n Optimal Threshold = {optimal_threshold_ica_1} \n")
    print(f"\n Number of selected sources = {best_score_ica_1} \n")

    print(f"\n=========================================== Spiky Sources from Second Noisy Signal with SNR: {snr}dB (PCA) ===========================================\n")
    EEGplot(spiky_sources_pca_2, 'PCA - Spiky', snr)
    print(f"\n Optimal Threshold = {optimal_threshold_pca_2} \n")
    print(f"\n Number of selected sources = {best_score_pca_2} \n")

    print(f"\n=========================================== Spiky Sources from Second Noisy Signal with SNR: {snr}dB (ICA) ===========================================\n")
    EEGplot(spiky_sources_ica_2, 'ICA - Spiky', snr)
    print(f"\n Optimal Threshold = {optimal_threshold_ica_2} \n")
    print(f"\n Number of selected sources = {best_score_ica_2} \n")

#%% # part c&d - Denoised signals
min_th = 0.2
max_th = 0.9
step_th = 0.01
maxspikes = 10
minsources = 3

for snr in SNRs:
    X_noisy_1 = add_noise(X_org, first_noise, snr_db)
    X_noisy_2 = add_noise(X_org, second_noise, snr_db)
    
    print(f"\n=========================================== SNR: {snr}dB ===========================================\n")
    sources_pca_1, unmixing_matrix_pca_1, _ = perform_source_extraction(X_noisy_1, method='pca') # Extract sources using PCA
    sources_ica_1, unmixing_matrix_ica_1, _ = perform_source_extraction(X_noisy_1, method='ica') # Extract sources using ICA
    sources_pca_2, unmixing_matrix_pca_2, _ = perform_source_extraction(X_noisy_2, method='pca') # Extract sources using PCA
    sources_ica_2, unmixing_matrix_ica_2, _ = perform_source_extraction(X_noisy_2, method='ica') # Extract sources using ICA

    # Normalize the sources
    normalized_sources_pca_1 = normalize_sources(sources_pca_1)
    normalized_sources_ica_1 = normalize_sources(sources_ica_1)
    normalized_sources_pca_2 = normalize_sources(sources_pca_2)
    normalized_sources_ica_2 = normalize_sources(sources_ica_2)
    
    # Optimize the threshold
    optimal_threshold_pca_1, best_score_pca_1 = optimize_threshold2(normalized_sources_pca_1, min_th, max_th, step_th, maxspikes, minsources)
    optimal_threshold_ica_1, best_score_ica_1 = optimize_threshold2(normalized_sources_ica_1, min_th, max_th, step_th, maxspikes, minsources)
    optimal_threshold_pca_2, best_score_pca_2 = optimize_threshold2(normalized_sources_pca_2, min_th, max_th, step_th, maxspikes, minsources)
    optimal_threshold_ica_2, best_score_ica_2 = optimize_threshold2(normalized_sources_ica_2, min_th, max_th, step_th, maxspikes, minsources)
    
    # Identify spiky sources using the optimal thresholds
    spiky_sources_pca_1, index_pca_1 = identify_spiky_sources2(normalized_sources_pca_1, optimal_threshold_pca_1, maxspikes)
    spiky_sources_ica_1, index_ica_1 = identify_spiky_sources2(normalized_sources_ica_1, optimal_threshold_ica_1, maxspikes)
    spiky_sources_pca_2, index_pca_2 = identify_spiky_sources2(normalized_sources_pca_2, optimal_threshold_pca_2, maxspikes)
    spiky_sources_ica_2, index_ica_2 = identify_spiky_sources2(normalized_sources_ica_2, optimal_threshold_ica_2, maxspikes)

    # Extract sources from ICAs
    extracted_sources_ica_1 = np.zeros(np.shape(sources_ica_1))
    for inx in index_ica_1:
        extracted_sources_ica_1[inx] = sources_ica_1[inx]
    denoised_signal_ica_1 = np.dot(unmixing_matrix_ica_1.T, extracted_sources_ica_1)
    
    extracted_sources_ica_2 = np.zeros(np.shape(sources_ica_2))
    for inx in index_ica_2:
        extracted_sources_ica_2[inx] = sources_ica_2[inx]
    denoised_signal_ica_2 = np.dot(unmixing_matrix_ica_2.T, extracted_sources_ica_2)
    
    # Visualization can be done using the EEGplot function
    print(f"\n=========================================== Denoised Spiky Sources from First Noisy Signal with SNR: {snr}dB (ICA) ===========================================\n")
    EEGplot(denoised_signal_ica_1, 'ICA - Spiky - Denoised', denoised=True, snr=None)
    print(f"\n Optimal Threshold = {optimal_threshold_ica_1} \n")
    print(f"\n Number of selected sources = {best_score_ica_1} \n")

    print(f"\n=========================================== Denoised Spiky Sources from Second Noisy Signal with SNR: {snr}dB (ICA) ===========================================\n")
    EEGplot(denoised_signal_ica_2, 'ICA - Spiky - Denoised', denoised=True, snr=None)
    print(f"\n Optimal Threshold = {optimal_threshold_ica_2} \n")
    print(f"\n Number of selected sources = {best_score_ica_2} \n")

#%% part c&d - Denoised signals versus number of min sources
min_th = 0.4
max_th = 0.9
step_th = 0.01
maxspikes = 10
minsourcess = [2, 4, 6, 8, 10, 12, 14]

for minsources in minsourcess:
    print("\n=================================================================================================================================\n")
    print(f"\n====================================================== Min sources: {minsources} ======================================================\n")
    print("\n=================================================================================================================================\n")
    for snr in SNRs:
        X_noisy_1 = add_noise(X_org, first_noise, snr_db)
        X_noisy_2 = add_noise(X_org, second_noise, snr_db)
        
        print(f"\n====================================================== SNR: {snr}dB ======================================================\n")
        sources_pca_1, unmixing_matrix_pca_1, _ = perform_source_extraction(X_noisy_1, method='pca') # Extract sources using PCA
        sources_ica_1, unmixing_matrix_ica_1, _ = perform_source_extraction(X_noisy_1, method='ica') # Extract sources using ICA
        sources_pca_2, unmixing_matrix_pca_2, _ = perform_source_extraction(X_noisy_2, method='pca') # Extract sources using PCA
        sources_ica_2, unmixing_matrix_ica_2, _ = perform_source_extraction(X_noisy_2, method='ica') # Extract sources using ICA
    
        # Normalize the sources
        normalized_sources_pca_1 = normalize_sources(sources_pca_1)
        normalized_sources_ica_1 = normalize_sources(sources_ica_1)
        normalized_sources_pca_2 = normalize_sources(sources_pca_2)
        normalized_sources_ica_2 = normalize_sources(sources_ica_2)
        
        # Optimize the threshold
        optimal_threshold_pca_1, best_score_pca_1 = optimize_threshold2(normalized_sources_pca_1, min_th, max_th, step_th, maxspikes, minsources)
        optimal_threshold_ica_1, best_score_ica_1 = optimize_threshold2(normalized_sources_ica_1, min_th, max_th, step_th, maxspikes, minsources)
        optimal_threshold_pca_2, best_score_pca_2 = optimize_threshold2(normalized_sources_pca_2, min_th, max_th, step_th, maxspikes, minsources)
        optimal_threshold_ica_2, best_score_ica_2 = optimize_threshold2(normalized_sources_ica_2, min_th, max_th, step_th, maxspikes, minsources)
        
        # Identify spiky sources using the optimal thresholds
        spiky_sources_pca_1, index_pca_1 = identify_spiky_sources2(normalized_sources_pca_1, optimal_threshold_pca_1, maxspikes)
        spiky_sources_ica_1, index_ica_1 = identify_spiky_sources2(normalized_sources_ica_1, optimal_threshold_ica_1, maxspikes)
        spiky_sources_pca_2, index_pca_2 = identify_spiky_sources2(normalized_sources_pca_2, optimal_threshold_pca_2, maxspikes)
        spiky_sources_ica_2, index_ica_2 = identify_spiky_sources2(normalized_sources_ica_2, optimal_threshold_ica_2, maxspikes)
    
        # Extract sources from ICAs
        extracted_sources_ica_1 = np.zeros(np.shape(sources_ica_1))
        for inx in index_ica_1:
            extracted_sources_ica_1[inx] = sources_ica_1[inx]
        denoised_signal_ica_1 = np.dot(unmixing_matrix_ica_1.T, extracted_sources_ica_1)
        
        extracted_sources_ica_2 = np.zeros(np.shape(sources_ica_2))
        for inx in index_ica_2:
            extracted_sources_ica_2[inx] = sources_ica_2[inx]
        denoised_signal_ica_2 = np.dot(unmixing_matrix_ica_2.T, extracted_sources_ica_2)
        
        # Visualization can be done using the EEGplot function
        print(f"\n=========================================== Denoised Spiky Sources from First Noisy Signal with SNR: {snr}dB (ICA) ===========================================\n")
        EEGplot(denoised_signal_ica_1, 'ICA - Spiky - Denoised', denoised=True, snr=None)
        print(f"\n Optimal Threshold = {optimal_threshold_ica_1} \n")
        print(f"\n Number of selected sources = {best_score_ica_1} \n")
    
        print(f"\n=========================================== Denoised Spiky Sources from Second Noisy Signal with SNR: {snr}dB (ICA) ===========================================\n")
        EEGplot(denoised_signal_ica_2, 'ICA - Spiky - Denoised', denoised=True, snr=None)
        print(f"\n Optimal Threshold = {optimal_threshold_ica_2} \n")
        print(f"\n Number of selected sources = {best_score_ica_2} \n")

#%%
min_th = 0.4
max_th = 0.9
step_th = 0.01
maxspikes = 3
minsourcess = [2, 3, 4, 5, 6]

for minsources in minsourcess:
    print("\n=================================================================================================================================\n")
    print(f"\n====================================================== Min sources: {minsources} ======================================================\n")
    print("\n=================================================================================================================================\n")
    for snr in SNRs:
        X_noisy_1 = add_noise(X_org, first_noise, snr_db)
        X_noisy_2 = add_noise(X_org, second_noise, snr_db)
        
        print(f"\n====================================================== SNR: {snr}dB ======================================================\n")
        sources_pca_1, unmixing_matrix_pca_1, _ = perform_source_extraction(X_noisy_1, method='pca') # Extract sources using PCA
        sources_ica_1, unmixing_matrix_ica_1, _ = perform_source_extraction(X_noisy_1, method='ica') # Extract sources using ICA
        sources_pca_2, unmixing_matrix_pca_2, _ = perform_source_extraction(X_noisy_2, method='pca') # Extract sources using PCA
        sources_ica_2, unmixing_matrix_ica_2, _ = perform_source_extraction(X_noisy_2, method='ica') # Extract sources using ICA
    
        # Normalize the sources
        normalized_sources_pca_1 = normalize_sources(sources_pca_1)
        normalized_sources_ica_1 = normalize_sources(sources_ica_1)
        normalized_sources_pca_2 = normalize_sources(sources_pca_2)
        normalized_sources_ica_2 = normalize_sources(sources_ica_2)
        
        # Optimize the threshold
        optimal_threshold_pca_1, best_score_pca_1 = optimize_threshold2(normalized_sources_pca_1, min_th, max_th, step_th, maxspikes, minsources)
        optimal_threshold_ica_1, best_score_ica_1 = optimize_threshold2(normalized_sources_ica_1, min_th, max_th, step_th, maxspikes, minsources)
        optimal_threshold_pca_2, best_score_pca_2 = optimize_threshold2(normalized_sources_pca_2, min_th, max_th, step_th, maxspikes, minsources)
        optimal_threshold_ica_2, best_score_ica_2 = optimize_threshold2(normalized_sources_ica_2, min_th, max_th, step_th, maxspikes, minsources)
        
        # Identify spiky sources using the optimal thresholds
        spiky_sources_pca_1, index_pca_1 = identify_spiky_sources2(normalized_sources_pca_1, optimal_threshold_pca_1, maxspikes)
        spiky_sources_ica_1, index_ica_1 = identify_spiky_sources2(normalized_sources_ica_1, optimal_threshold_ica_1, maxspikes)
        spiky_sources_pca_2, index_pca_2 = identify_spiky_sources2(normalized_sources_pca_2, optimal_threshold_pca_2, maxspikes)
        spiky_sources_ica_2, index_ica_2 = identify_spiky_sources2(normalized_sources_ica_2, optimal_threshold_ica_2, maxspikes)
    
        # Extract sources from ICAs
        extracted_sources_ica_1 = np.zeros(np.shape(sources_ica_1))
        for inx in index_ica_1:
            extracted_sources_ica_1[inx] = sources_ica_1[inx]
        denoised_signal_ica_1 = np.dot(unmixing_matrix_ica_1.T, extracted_sources_ica_1)
        
        extracted_sources_ica_2 = np.zeros(np.shape(sources_ica_2))
        for inx in index_ica_2:
            extracted_sources_ica_2[inx] = sources_ica_2[inx]
        denoised_signal_ica_2 = np.dot(unmixing_matrix_ica_2.T, extracted_sources_ica_2)
        
        # Visualization can be done using the EEGplot function
        print(f"\n=========================================== Denoised Spiky Sources from First Noisy Signal with SNR: {snr}dB (ICA) ===========================================\n")
        EEGplot(denoised_signal_ica_1, 'ICA - Spiky - Denoised', denoised=True, snr=None)
        print(f"\n Optimal Threshold = {optimal_threshold_ica_1} \n")
        print(f"\n Number of selected sources = {best_score_ica_1} \n")
    
        print(f"\n=========================================== Denoised Spiky Sources from Second Noisy Signal with SNR: {snr}dB (ICA) ===========================================\n")
        EEGplot(denoised_signal_ica_2, 'ICA - Spiky - Denoised', denoised=True, snr=None)
        print(f"\n Optimal Threshold = {optimal_threshold_ica_2} \n")
        print(f"\n Number of selected sources = {best_score_ica_2} \n")

#%% part e
selected_channels = [13, 24]
SNRs = [-10, -20] #dB

# ICA
for snr in SNRs:
    if snr == SNRs[0]:
        X_noisy_1_10 = add_noise(X_org, first_noise, snr)
        X_noisy_2_10 = add_noise(X_org, second_noise, snr)
        sources_ica_1_10, unmixing_matrix_ica_1_10, ica_X_noisy_1_10 = perform_source_extraction(X_noisy_1_10, method='ica') # Extract sources using ICA
        sources_ica_2_10, unmixing_matrix_ica_2_10, ica_X_noisy_2_10 = perform_source_extraction(X_noisy_2_10, method='ica') # Extract sources using ICA
        EEGplot(sources_ica_1_10, 'ICA', snr)
        EEGplot(sources_ica_2_10, 'ICA', snr)
    elif snr == SNRs[-1]:
        X_noisy_1_20 = add_noise(X_org, first_noise, snr)
        X_noisy_2_20 = add_noise(X_org, second_noise, snr)
        sources_ica_1_20, unmixing_matrix_ica_1_20, ica_X_noisy_1_20 = perform_source_extraction(X_noisy_1_20, method='ica') # Extract sources using ICA
        sources_ica_2_20, unmixing_matrix_ica_2_20, ica_X_noisy_2_20 = perform_source_extraction(X_noisy_2_20, method='ica') # Extract sources using ICA
        EEGplot(sources_ica_1_20, 'ICA', snr)
        EEGplot(sources_ica_2_20, 'ICA', snr)


# PCA
for snr in SNRs:
    if snr == SNRs[0]:
        X_noisy_1_10 = add_noise(X_org, first_noise, snr)
        X_noisy_1_10_mean = np.mean(X_noisy_1_10, axis=1)
        X_noisy_1_10_std = np.std(X_noisy_1_10, axis=1)
        X_noisy_1_10_normalized = np.zeros(np.shape(X_noisy_1_10))
        for i in range(len(X_noisy_1_10)):
            X_noisy_1_10_normalized[i] = (X_noisy_1_10[i] - X_noisy_1_10_mean[i]) / X_noisy_1_10_std[i]
        cov_matrix_X_noisy_1_10 = np.cov(X_noisy_1_10_normalized, rowvar=True)
        eigenvalues_X_noisy_1_10, eigenvectors_X_noisy_1_10 = np.linalg.eigh(cov_matrix_X_noisy_1_10)
        idx_X_noisy_1_10 = np.argsort(eigenvalues_X_noisy_1_10)[::-1]
        eigenvectors_X_noisy_1_10 = eigenvectors_X_noisy_1_10[idx_X_noisy_1_10, :]
        # Unmixing matrix
        unmixing_matrix_X_noisy_1_10 = np.linalg.inv(eigenvectors_X_noisy_1_10)
        n_components_X_noisy_1_10 = min(X_noisy_1_10_normalized.shape)  # You can choose a different number of components
        pca_X_noisy_1_10 = PCA(n_components=n_components_X_noisy_1_10)
        X_noisy_1_10_transformed = pca_X_noisy_1_10.fit_transform(X_noisy_1_10_normalized.T).T
        EEGplot(X_noisy_1_10_transformed, 'PCA', snr)
        
        X_noisy_2_10 = add_noise(X_org, second_noise, snr)
        X_noisy_2_10_mean = np.mean(X_noisy_2_10, axis=1)
        X_noisy_2_10_std = np.std(X_noisy_2_10, axis=1)
        X_noisy_2_10_normalized = np.zeros(np.shape(X_noisy_2_10))
        for i in range(len(X_noisy_2_10)):
            X_noisy_2_10_normalized[i] = (X_noisy_2_10[i] - X_noisy_2_10_mean[i]) / X_noisy_2_10_std[i]
        cov_matrix_X_noisy_2_10 = np.cov(X_noisy_2_10_normalized, rowvar=True)
        eigenvalues_X_noisy_2_10, eigenvectors_X_noisy_2_10 = np.linalg.eigh(cov_matrix_X_noisy_2_10)
        idx_X_noisy_2_10 = np.argsort(eigenvalues_X_noisy_2_10)[::-1]
        eigenvectors_X_noisy_2_10 = eigenvectors_X_noisy_2_10[:, idx_X_noisy_2_10]
        # Unmixing matrix
        unmixing_matrix_X_noisy_2_10 = np.linalg.inv(eigenvectors_X_noisy_2_10)
        n_components_X_noisy_2_10 = min(X_noisy_2_10_normalized.shape)  # You can choose a different number of components
        pca_X_noisy_2_10 = PCA(n_components=n_components_X_noisy_2_10)
        X_noisy_2_10_transformed = pca_X_noisy_2_10.fit_transform(X_noisy_2_10_normalized.T).T
        EEGplot(X_noisy_2_10_transformed, 'PCA', snr)
        
    elif snr == SNRs[-1]:
        X_noisy_1_20 = add_noise(X_org, first_noise, snr)
        X_noisy_1_20_mean = np.mean(X_noisy_1_20, axis=1)
        X_noisy_1_20_std = np.std(X_noisy_1_20, axis=1)
        X_noisy_1_20_normalized = np.zeros(np.shape(X_noisy_1_20))
        for i in range(len(X_noisy_1_20)):
            X_noisy_1_20_normalized[i] = (X_noisy_1_20[i] - X_noisy_1_20_mean[i]) / X_noisy_1_20_std[i]
        cov_matrix_X_noisy_1_20 = np.cov(X_noisy_1_20_normalized, rowvar=True)
        eigenvalues_X_noisy_1_20, eigenvectors_X_noisy_1_20 = np.linalg.eigh(cov_matrix_X_noisy_1_20)
        idx_X_noisy_1_20 = np.argsort(eigenvalues_X_noisy_1_20)[::-1]
        eigenvectors_X_noisy_1_20 = eigenvectors_X_noisy_1_20[:, idx_X_noisy_1_20]
        # Unmixing matrix
        unmixing_matrix_X_noisy_1_20 = np.linalg.inv(eigenvectors_X_noisy_1_20)
        n_components_X_noisy_1_20 = min(X_noisy_1_20_normalized.shape)  # You can choose a different number of components
        pca_X_noisy_1_20 = PCA(n_components=n_components_X_noisy_1_20)
        X_noisy_1_20_transformed = pca_X_noisy_1_20.fit_transform(X_noisy_1_20_normalized.T).T
        EEGplot(X_noisy_1_20_transformed, 'PCA', snr)

        X_noisy_2_20 = add_noise(X_org, second_noise, snr)
        X_noisy_2_20_mean = np.mean(X_noisy_2_20, axis=1)
        X_noisy_2_20_std = np.std(X_noisy_2_20, axis=1)
        X_noisy_2_20_normalized = np.zeros(np.shape(X_noisy_2_20))
        for i in range(len(X_noisy_2_20)):
            X_noisy_2_20_normalized[i] = (X_noisy_2_20[i] - X_noisy_2_20_mean[i]) / X_noisy_2_20_std[i]
        cov_matrix_X_noisy_2_20 = np.cov(X_noisy_2_20_normalized, rowvar=True)
        eigenvalues_X_noisy_2_20, eigenvectors_X_noisy_2_20 = np.linalg.eigh(cov_matrix_X_noisy_2_20)
        idx_X_noisy_2_20 = np.argsort(eigenvalues_X_noisy_2_20)[::-1]
        eigenvectors_X_noisy_2_20 = eigenvectors_X_noisy_2_20[:, idx_X_noisy_2_20]
        # Unmixing matrix
        unmixing_matrix_X_noisy_2_20 = np.linalg.inv(eigenvectors_X_noisy_2_20)
        n_components_X_noisy_2_20 = min(X_noisy_2_20_normalized.shape)  # You can choose a different number of components
        pca_X_noisy_2_20 = PCA(n_components=n_components_X_noisy_2_20)
        X_noisy_2_20_transformed = pca_X_noisy_2_20.fit_transform(X_noisy_2_20_normalized.T).T       
        EEGplot(X_noisy_2_20_transformed, 'PCA', snr)
        
#%%
# ica
selected_sources_ica_1_10 = [13, 18]
selected_sources_ica_2_10 = [17, 21]
selected_sources_ica_1_20 = [6, 15]
selected_sources_ica_2_20 = [8]


print("\n================================== First Noisy Signal ==============================================\n")
for chn in selected_channels:
    for snr_db in SNRs:
        print(f"\n====================================================== SNR: {snr_db}dB ======================================================\n")        
        if snr_db == SNRs[0]:
            # Extract sources from ICAs
            extracted_sources_ica_1_10 = np.zeros(np.shape(sources_ica_1_10))
            for inx in selected_sources_ica_1_10:
                extracted_sources_ica_1_10[inx-1] = sources_ica_1_10[inx-1]
            denoised_signal_ica_1_10 = np.dot(unmixing_matrix_ica_1_10.T, extracted_sources_ica_1_10)
            plt.figure(figsize=(20, 3))
            plt.plot(X_org[chn-1, :]/np.max(np.abs(X_org[chn-1, :])), label='Original Signal (without noise)', lw=3, c='k')
            plt.plot(denoised_signal_ica_1_10[chn-1, :]/np.max(np.abs(denoised_signal_ica_1_10[chn-1, :])), label=f'First Denoised Signal (SNR={snr_db}dB)', c='r')
            plt.title(f'Signal Comparison for Channel {chn} (SNR={snr_db}dB)')
            plt.legend()
            plt.xlim([0, len(X_org[0])])
            plt.tight_layout()
            plt.show()
            
        elif snr_db == SNRs[-1]:
            # Extract sources from ICAs
            extracted_sources_ica_1_20 = np.zeros(np.shape(sources_ica_1_20))
            for inx in selected_sources_ica_1_20:
                extracted_sources_ica_1_20[inx-1] = sources_ica_1_20[inx-1]
            denoised_signal_ica_1_20 = np.dot(unmixing_matrix_ica_1_20.T, extracted_sources_ica_1_20)
            plt.figure(figsize=(20, 3))
            plt.plot(X_org[chn-1, :]/np.max(np.abs(X_org[chn-1, :])), label='Original Signal (without noise)', lw=3, c='k')
            plt.plot(denoised_signal_ica_1_20[chn-1, :]/np.max(np.abs(denoised_signal_ica_1_20[chn-1, :])), label=f'First Denoised Signal (SNR={snr_db}dB)', c='r')
            plt.title(f'Signal Comparison for Channel {chn} (SNR={snr_db}dB)')
            plt.legend()
            plt.xlim([0, len(X_org[0])])
            plt.tight_layout()
            plt.show()

print("\n\n\n================================== Second Noisy Signal ==============================================\n")
for chn in selected_channels:
    for snr_db in SNRs:
        print(f"\n====================================================== SNR: {snr_db}dB ======================================================\n")        
        if snr_db == SNRs[0]:
            # Extract sources from ICAs
            extracted_sources_ica_2_10 = np.zeros(np.shape(sources_ica_2_10))
            for inx in selected_sources_ica_2_10:
                extracted_sources_ica_2_10[inx-1] = sources_ica_2_10[inx-1]
            denoised_signal_ica_2_10 = np.dot(unmixing_matrix_ica_2_10.T, extracted_sources_ica_2_10)
            plt.figure(figsize=(20, 3))
            plt.plot(X_org[chn-1, :]/np.max(np.abs(X_org[chn-1, :])), label='Original Signal (without noise)', lw=3, c='k')
            plt.plot(denoised_signal_ica_2_10[chn-1, :]/np.max(np.abs(denoised_signal_ica_2_10[chn-1, :])), label=f'Second Denoised Signal (SNR={snr_db}dB)', c='r')
            plt.title(f'Signal Comparison for Channel {chn} (SNR={snr_db}dB)')
            plt.legend()
            plt.xlim([0, len(X_org[0])])
            plt.tight_layout()
            plt.show()
            
        elif snr_db == SNRs[-1]:
            # Extract sources from ICAs
            extracted_sources_ica_2_20 = np.zeros(np.shape(sources_ica_2_20))
            for inx in selected_sources_ica_2_20:
                extracted_sources_ica_2_20[inx-1] = sources_ica_2_20[inx-1]
            denoised_signal_ica_2_20 = np.dot(unmixing_matrix_ica_2_20.T, extracted_sources_ica_2_20)
            plt.figure(figsize=(20, 3))
            plt.plot(X_org[chn-1, :]/np.max(np.abs(X_org[chn-1, :])), label='Original Signal (without noise)', lw=3, c='k')
            plt.plot(denoised_signal_ica_2_20[chn-1, :]/np.max(np.abs(denoised_signal_ica_2_20[chn-1, :])), label=f'Second Denoised Signal (SNR={snr_db}dB)', c='r')
            plt.title(f'Signal Comparison for Channel {chn} (SNR={snr_db}dB)')
            plt.legend()
            plt.xlim([0, len(X_org[0])])
            plt.tight_layout()
            plt.show()
            
# pca
selected_sources_pca_1_10 = [32]
selected_sources_pca_2_10 = [29]
selected_sources_pca_1_20 = [32]
selected_sources_pca_2_20 = [29]


print("\n================================== First Noisy Signal ==============================================\n")
for chn in selected_channels:
    for snr_db in SNRs:
        print(f"\n====================================================== SNR: {snr_db}dB ======================================================\n")        
        if snr_db == SNRs[0]:
            # Extract sources from ICAs
            extracted_sources_pca_1_10 = np.zeros(np.shape(X_noisy_1_10_transformed))
            for inx in selected_sources_pca_1_10:
                extracted_sources_pca_1_10[inx-1] = X_noisy_1_10_transformed[inx-1]
            denoised_signal_pca_1_10 = np.dot(unmixing_matrix_X_noisy_1_10.T, extracted_sources_pca_1_10)
            plt.figure(figsize=(20, 3))
            plt.plot(X_org[chn-1, :]/np.max(np.abs(X_org[chn-1, :])), label='Original Signal (without noise)', lw=3, c='k')
            plt.plot(denoised_signal_pca_1_10[chn-1, :]/np.max(np.abs(denoised_signal_pca_1_10[chn-1, :])), label=f'First Denoised Signal (SNR={snr_db}dB)', c='r')
            plt.title(f'Signal Comparison for Channel {chn} (SNR={snr_db}dB)')
            plt.legend()
            plt.xlim([0, len(X_org[0])])
            plt.tight_layout()
            plt.show()
            
        elif snr_db == SNRs[-1]:
            # Extract sources from ICAs
            extracted_sources_pca_1_20 = np.zeros(np.shape(X_noisy_1_20_transformed))
            for inx in selected_sources_pca_1_20:
                extracted_sources_pca_1_20[inx-1] = X_noisy_1_20_transformed[inx-1]
            denoised_signal_pca_1_20 = np.dot(unmixing_matrix_X_noisy_1_20.T, extracted_sources_pca_1_20)
            plt.figure(figsize=(20, 3))
            plt.plot(X_org[chn-1, :]/np.max(np.abs(X_org[chn-1, :])), label='Original Signal (without noise)', lw=3, c='k')
            plt.plot(denoised_signal_pca_1_20[chn-1, :]/np.max(np.abs(denoised_signal_pca_1_20[chn-1, :])), label=f'First Denoised Signal (SNR={snr_db}dB)', c='r')
            plt.title(f'Signal Comparison for Channel {chn} (SNR={snr_db}dB)')
            plt.legend()
            plt.xlim([0, len(X_org[0])])
            plt.tight_layout()
            plt.show()

print("\n\n\n================================== Second Noisy Signal ==============================================\n")
for chn in selected_channels:
    for snr_db in SNRs:
        print(f"\n====================================================== SNR: {snr_db}dB ======================================================\n")        
        if snr_db == SNRs[0]:
            # Extract sources from ICAs
            extracted_sources_pca_2_10 = np.zeros(np.shape(X_noisy_2_10_transformed))
            for inx in selected_sources_pca_2_10:
                extracted_sources_pca_2_10[inx-1] = X_noisy_2_10_transformed[inx-1]
            denoised_signal_pca_2_10 = np.dot(unmixing_matrix_X_noisy_2_10.T, extracted_sources_pca_2_10)
            plt.figure(figsize=(20, 3))
            plt.plot(X_org[chn-1, :]/np.max(np.abs(X_org[chn-1, :])), label='Original Signal (without noise)', lw=3, c='k')
            plt.plot(denoised_signal_pca_2_10[chn-1, :]/np.max(np.abs(denoised_signal_pca_2_10[chn-1, :])), label=f'Second Denoised Signal (SNR={snr_db}dB)', c='r')
            plt.title(f'Signal Comparison for Channel {chn} (SNR={snr_db}dB)')
            plt.legend()
            plt.xlim([0, len(X_org[0])])
            plt.tight_layout()
            plt.show()
            
        elif snr_db == SNRs[-1]:
            # Extract sources from ICAs
            extracted_sources_pca_2_20 = np.zeros(np.shape(X_noisy_2_20_transformed))
            for inx in selected_sources_pca_2_20:
                extracted_sources_pca_2_20[inx-1] = X_noisy_2_20_transformed[inx-1]
            denoised_signal_pca_2_20 = np.dot(unmixing_matrix_X_noisy_2_20.T, extracted_sources_pca_2_20)
            plt.figure(figsize=(20, 3))
            plt.plot(X_org[chn-1, :]/np.max(np.abs(X_org[chn-1, :])), label='Original Signal (without noise)', lw=3, c='k')
            plt.plot(denoised_signal_pca_2_20[chn-1, :]/np.max(np.abs(denoised_signal_pca_2_20[chn-1, :])), label=f'Second Denoised Signal (SNR={snr_db}dB)', c='r')
            plt.title(f'Signal Comparison for Channel {chn} (SNR={snr_db}dB)')
            plt.legend()
            plt.xlim([0, len(X_org[0])])
            plt.tight_layout()
            plt.show()

#%% part f
# Function to compute RRMSE
def rrmse(original, estimate):
    return np.sqrt(np.mean((original - estimate) ** 2) / np.mean(original ** 2))

results = []
# ICA
print(f"\n==================================================================================================================================================================\n")        
print(f"\n====================================================== ICA ======================================================\n") 
print(f"\n==================================================================================================================================================================\n") 

print(f'\n====================================================== Denoised First Noisy Signal ======================================================\n')
for snr_db in SNRs:
    print(f"\n====================================================== SNR: {snr_db}dB ======================================================\n")        
    if snr_db == SNRs[0]:
        denoised_sources_ica_1_10 = ica_X_noisy_1_10.inverse_transform(extracted_sources_ica_1_10.T).T
        rrmse_pca_ica_1_10 = rrmse(X_org, denoised_sources_ica_1_10)
        results.append((snr_db, 'ICA - First Noisy', rrmse_pca_ica_1_10))
        EEGplot(denoised_sources_ica_1_10, None, None, denoised=True)
    elif snr_db == SNRs[-1]:
        denoised_sources_ica_1_20 = ica_X_noisy_1_20.inverse_transform(extracted_sources_ica_1_20.T).T
        rrmse_pca_ica_1_20 = rrmse(X_org, denoised_sources_ica_1_20)
        results.append((snr_db, 'ICA - First Noisy', rrmse_pca_ica_1_20))
        EEGplot(denoised_sources_ica_1_20, None, None, denoised=True)
        
print(f'\n====================================================== Denoised Second Noisy Signal ======================================================\n')
for snr_db in SNRs:
    print(f"\n====================================================== SNR: {snr_db}dB ======================================================\n")        
    if snr_db == SNRs[0]:
        denoised_sources_ica_2_10 = ica_X_noisy_2_10.inverse_transform(extracted_sources_ica_2_10.T).T
        rrmse_pca_ica_2_10 = rrmse(X_org, denoised_sources_ica_2_10)
        results.append((snr_db, 'ICA - Second Noisy', rrmse_pca_ica_2_10))
        EEGplot(denoised_sources_ica_2_10, None, None, denoised=True)

    elif snr_db == SNRs[-1]:
        denoised_sources_ica_2_20 = ica_X_noisy_2_10.inverse_transform(extracted_sources_ica_2_20.T).T
        rrmse_pca_ica_2_20 = rrmse(X_org, denoised_sources_ica_2_20)
        results.append((snr_db, 'ICA - Second Noisy', rrmse_pca_ica_2_20))
        EEGplot(denoised_sources_ica_2_20, None, None, denoised=True)

# PCA
print(f"\n\n==================================================================================================================================================================\n")        
print(f"\n====================================================== PCA ======================================================\n") 
print(f"\n==================================================================================================================================================================\n")        

print(f'\n====================================================== Denoised First Noisy Signal ======================================================\n')
for snr_db in SNRs:
    print(f"\n====================================================== SNR: {snr_db}dB ======================================================\n")        
    if snr_db == SNRs[0]:
        denoised_sources_pca_1_10 = pca_X_noisy_1_10.inverse_transform(extracted_sources_pca_1_10.T).T
        rrmse_pca_pca_1_10 = rrmse(X_org, denoised_sources_pca_1_10)
        results.append((snr_db, 'PCA - First Noisy', rrmse_pca_pca_1_10))
        EEGplot(denoised_sources_pca_1_10, None, None, denoised=True)

    elif snr_db == SNRs[-1]:
        denoised_sources_pca_1_20 = pca_X_noisy_1_20.inverse_transform(extracted_sources_pca_1_20.T).T
        rrmse_pca_pca_1_20 = rrmse(X_org, denoised_sources_pca_1_20)
        results.append((snr_db, 'PCA - First Noisy', rrmse_pca_pca_1_20))
        EEGplot(denoised_sources_pca_1_20, None, None, denoised=True)

print(f'\n====================================================== Denoised Second Noisy Signal ======================================================\n')
for snr_db in SNRs:
    print(f"\n====================================================== SNR: {snr_db}dB ======================================================\n")        
    if snr_db == SNRs[0]:
        denoised_sources_pca_2_10 = pca_X_noisy_2_10.inverse_transform(extracted_sources_pca_2_10.T).T
        rrmse_pca_pca_2_10 = rrmse(X_org, denoised_sources_pca_2_10)
        results.append((snr_db, 'PCA - Second Noisy', rrmse_pca_pca_2_10))
        EEGplot(denoised_sources_pca_2_10, None, None, denoised=True)

    elif snr_db == SNRs[-1]:
        denoised_sources_pca_2_20 = pca_X_noisy_2_20.inverse_transform(extracted_sources_pca_2_20.T).T
        rrmse_pca_pca_2_20 = rrmse(X_org, denoised_sources_pca_2_20)
        results.append((snr_db, 'PCA - Second Noisy', rrmse_pca_pca_2_20))
        EEGplot(denoised_sources_pca_2_20, None, None, denoised=True)


#%%
import pandas as pd

# Convert results to a DataFrame for easier manipulation
df_results = pd.DataFrame(results, columns=['SNR', 'Method and 1st/2nd Noisy Signal', 'RRMSE'])

# Display results in a table
print(df_results)

# Plotting the results
plt.figure(figsize=(10, 6))
for method in df_results['Method and 1st/2nd Noisy Signal'].unique():
    subset = df_results[df_results['Method and 1st/2nd Noisy Signal'] == method]
    plt.plot(subset['SNR'], subset['RRMSE'], marker='o', label=method)

plt.xlabel('SNR (dB)')
plt.ylabel('RRMSE')
plt.title('RRMSE of De-noised Signals by PCA and ICA Methods')
plt.legend()
plt.grid(True)
plt.show()

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
def EEGplot(signal, method, snr, denoised=False, fs=False, flag=False):
    plt.figure(figsize=(21, 14))
    num_sources = signal.shape[0] 
    
    if method == False and snr != False and denoised==False and fs==False and flag==False:
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
            
    elif method != False and snr != False and denoised==False and fs==False and flag==False:
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
            
    elif method == False and snr == False and denoised==False and fs==False and flag==False:
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
            
    elif denoised==True and fs==False and flag==False:
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
            
    elif fs!=False and method==False and snr==False and denoised==False and flag==False:
        plt.suptitle(f'Original Channels (without noise)')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(np.arange(0, signal.shape[1]/fs, 1/fs), signal[j, :], label=f'Ch{j + 1}', c='k')
            if j+1 != signal.shape[0]:
                plt.xticks([])
            plt.yticks([])
            plt.ylabel(f'Ch{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]/fs])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
        plt.xlabel("Time (sec)")

    elif method != False and snr == False and fs!=False and flag==False:
        colors = [
                    'b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink',  # Standard colors: blue, green, red, cyan, magenta, yellow, black
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',  # Tableau Colors
                    '#1a1a1a', '#666666', '#a6a6a6', '#d9d9d9',  # Shades of gray
                    '#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#ff33ff', '#ffff99', '#9966cc', '#ff6666', '#c2c2f0', '#ffb3e6',  # Additional colors
                ]
        plt.suptitle(f'Extracted Sources ({method})')
        for j in range(signal.shape[0]):
            plt.subplot(num_sources, 1, j + 1)
            plt.plot(np.arange(0, signal.shape[1]/fs, 1/fs), signal[j, :], label=f'S{j + 1}', c=colors[j])
            plt.yticks([])
            if j+1 != signal.shape[0]:
                plt.xticks([])
            plt.ylabel(f'S{j + 1}', rotation=90)
            plt.tight_layout()
            plt.xlim([0, signal.shape[1]/fs])
            plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
        plt.xlabel("Time (sec)")

    elif denoised==True and fs!=False and snr==False and method==False and flag==True:
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
    
#%% part a
fs = 250
print("\n========================================================================================== Data 1 ==========================================================================================\n")
EEGplot(ND1, False, False, False, fs)
print("\n========================================================================================== Data 2 ==========================================================================================\n")
EEGplot(ND2, False, False, False, fs)
print("\n========================================================================================== Data 3 ==========================================================================================\n")
EEGplot(ND3, False, False, False, fs)
print("\n========================================================================================== Data 4 ==========================================================================================\n")
EEGplot(ND4, False, False, False, fs)

# Function to calculate the power spectrum of each channel
def power_spectrum(eeg_data):
    power_spectra = []
    num_channels = eeg_data.shape[0]
    for i in range(num_channels):
        ps = np.abs(np.fft.fft(eeg_data[i, :]))**2
        power_spectra.append(ps)
    return power_spectra

# Calculating the power spectra for each dataset
ps1 = power_spectrum(ND1)
ps2 = power_spectrum(ND2)
ps3 = power_spectrum(ND3)
ps4 = power_spectrum(ND4)

# Function to plot power spectrum
def plot_power_spectrum(ps, title):
    plt.figure(figsize=(15, 10))
    channels = len(ps)
    n = len(ps[0])
    freqs = np.fft.fftfreq(n, d=1./250)  # Assuming a sampling frequency of 250 Hz
    plt.suptitle(f"{title} - Power Spectrum")
    for i in range(channels):
        plt.subplot(channels, 1, i+1)
        plt.plot(freqs[:n // 2], ps[i][:n // 2], label=f"Channel {i+1}")
        plt.ylabel('Power')
        plt.xlim([0, fs/2])
        if i+1 != channels:
            plt.xticks([])
        plt.yticks([])
        plt.ylabel(f'Ch{i + 1}', rotation=90)
        plt.tight_layout()
        plt.gca().yaxis.label.set(rotation='horizontal', ha='right', va='center');
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(True)
        
    plt.xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

# Plotting the power spectra for each dataset
plot_power_spectrum(ps1, "NewData1")
plot_power_spectrum(ps2, "NewData2")
plot_power_spectrum(ps3, "NewData3")
plot_power_spectrum(ps4, "NewData4")

'''
# %load C:/Users/aryak/Downloads/Comp_HW2/Ex3/COM2R.m
function [F,W,K]=COM2R(Y,Pest)
disp('COM2')
% Comon, version 6 march 92
% English comments added in 1994
% [F,delta]=aci(Y)
% Y is the observations matrix
% This routine outputs a matrix F such that Y=F*Z, Z=pinv(F)*Y,
% and components of Z are uncorrelated and approximately independent
% F is Nxr, where r is the dimension Z;
% Entries of delta are sorted in decreasing order;
% Columns of F are of unit norm;
% The entry of largest modulus in each column of F is positive real.
% Initial and final values of contrast can be fprinted for checking.
% REFERENCE: P.Comon, "Independent Component Analysis, a new concept?",
% Signal Processing, Elsevier, vol.36, no 3, April 1994, 287-314.
%
[N,TT]=size(Y);T=max(N,TT);N=min(N,TT);
if TT==N, Y=Y';[N,T]=size(Y);end; % Y est maintenant NxT avec N<T.
%%%% STEPS 1 & 2: whitening and projection (PCA)
[U,S,V]=svd(Y',0);tol=max(size(S))*norm(S)*eps;
s=diag(S);I=find(s<tol);

%--- modif de Laurent le 03/02/2009
r = min(Pest,N);
U=U(:,1:r);
S=S(1:r,1:r);
V=V(:,1:r);
%---

Z=U'*sqrt(T);L=V*S'/sqrt(T);F=L; %%%%%% on a Y=L*Z;
%%%%%% INITIAL CONTRAST
T=length(Z);contraste=0;
for i=1:r,
 gii=Z(i,:)*Z(i,:)'/T;Z2i=Z(i,:).^2;;giiii=Z2i*Z2i'/T;
 qiiii=giiii/gii/gii-3;contraste=contraste+qiiii*qiiii;
end;
%%%% STEPS 3 & 4 & 5: Unitary transform
S=Z;
if N==2,K=1;else,K=1+round(sqrt(N));end;  % K= max number of sweeps
Rot=eye(r);
for k=1:K,                           %%%%%% strating sweeps
Q=eye(r);
  for i=1:r-1,
  for j= i+1:r,
    S1ij=[S(i,:);S(j,:)];
    [Sij,qij]=tfuni4(S1ij);    %%%%%% processing a pair
    S(i,:)=Sij(1,:);S(j,:)=Sij(2,:);
    Qij=eye(r);Qij(i,i)=qij(1,1);Qij(i,j)=qij(1,2);
    Qij(j,i)=qij(2,1);Qij(j,j)=qij(2,2);
    Q=Qij*Q;
  end;
  end;
Rot=Rot*Q';
end;                                    %%%%%% end sweeps
F=F*Rot;
%%%%%% FINAL CONTRAST
S=Rot'*Z;
T=length(S);contraste=0;
for i=1:r,
 gii=S(i,:)*S(i,:)'/T;S2i=S(i,:).^2;;giiii=S2i*S2i'/T;
 qiiii=giiii/gii/gii-3;contraste=contraste+qiiii*qiiii;
end;
%%%% STEP 6: Norming columns
delta=diag(sqrt(sum(F.*conj(F))));
%%%% STEP 7: Sorting
[d,I]=sort(-diag(delta));E=eye(r);P=E(:,I)';delta=P*delta*P';F=F*P';
%%%% STEP 8: Norming
F=F*inv(delta);
%%%% STEP 9: Phase of columns
[y,I]=max(abs(F));
for i=1:r,Lambda(i)=conj(F(I(i),i));end;Lambda=Lambda./abs(Lambda);
F=F*diag(Lambda);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCUL DE LA MATRICE DE FILTRAGE
%---------------------------------
W = pinv(F);
'''

from sklearn.decomposition import FastICA

# Function to apply ICA and return independent components and mixing matrix
def apply_ica(eeg_data):
    ica = FastICA(n_components=eeg_data.shape[0], random_state=0)
    S = ica.fit_transform(eeg_data.T)  # Reconstruct signals
    A = ica.mixing_  # Get estimated mixing matrix
    return S, A

# Applying ICA to each dataset
ica_results = {}
for i, data in enumerate([ND1, ND2, ND3, ND4], start=1):
    S, A = apply_ica(data)
    ica_results[f"NewData{i}"] = {"components": S, "mixing_matrix": A}

# Let's check the shape of the components and mixing matrices for each dataset
ica_shapes = {key: {"components_shape": value["components"].shape, "mixing_matrix_shape": value["mixing_matrix"].shape} 
              for key, value in ica_results.items()}

# Function to perform PCA and ICA
def perform_source_extraction(signal, method='pca'):
    if method == 'pca':
        model = PCA()
        sources = model.fit_transform(signal.T).T
    elif method == 'ica':
        model = FastICA()
        sources = model.fit_transform(signal.T).T
        unmixing_matrix = model.mixing_
    
    return sources , unmixing_matrix if method == 'ica' else None, model

cnt = 0
for X in [ND1, ND2, ND3, ND4]:
    cnt += 1 
    print("=================================================================================================================================")
    print(f"\n======================================================== Data: NewData{cnt} ========================================================\n")
    print("=================================================================================================================================")
    print("\n======================================================== Original Signal (without noise) ========================================================\n")
    EEGplot(X, False, False, False, fs)
    
    sources_pca_1, unmixing_matrix_pca_1, model_pca_1 = perform_source_extraction(X, method='pca') # Extract sources using PCA
    sources_ica_1, unmixing_matrix_ica_1, model_ica_1 = perform_source_extraction(X, method='ica') # Extract sources using ICA
    EEGplot(sources_pca_1, 'PCA', False, fs=fs)
    EEGplot(sources_ica_1, 'ICA', False, fs=fs)
    
from scipy.signal import welch

# Function to plot time and frequency characteristics of components
def plot_component_characteristics(components, sampling_frequency=250):
    num_components = components.shape[1]
    n = min(num_components, 5)  # Limiting the number of components for demonstration

    for i in range(n):
        component = components[:, i]

        # Time characteristics
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(component)
        plt.title(f"Component {i+1} - Time Domain")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        # Frequency characteristics
        freqs, psd = welch(component, fs=sampling_frequency)
        plt.subplot(1, 2, 2)
        plt.semilogy(freqs, psd)
        plt.title(f"Component {i+1} - Frequency Domain")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.tight_layout()
        plt.show()

# Plotting the characteristics for components from NewData1
plot_component_characteristics(ica_results["NewData1"]["components"])
plot_component_characteristics(ica_results["NewData2"]["components"])
plot_component_characteristics(ica_results["NewData3"]["components"])
plot_component_characteristics(ica_results["NewData4"]["components"])

# Hypothetical selection of desirable sources
fs = 250
print("\n========================================================================================== Data 1 ==========================================================================================\n")
EEGplot(ND1, False, False, False, fs)

sources_ica_1, unmixing_matrix_ica_1, model_ica_1 = perform_source_extraction(ND1, method='ica') # Extract sources using ICA
EEGplot(sources_ica_1, 'ICA', False, fs=fs)
SelSources = [i for i in range(21) if i not in [10, 12]]
# Extracting the mixing matrix A and components S for NewData1
A = ica_results["NewData1"]["mixing_matrix"]
S = ica_results["NewData1"]["components"]

# Reconstructing the de-noised signal
X_denoised = A[:, SelSources] @ S[SelSources, :]

# Plotting the original and de-noised signals for comparison
EEGplot(unmixing_matrix_ica_1[:, SelSources] @ sources_ica_1[SelSources, :], method=False, snr=False, fs=fs, denoised=True, flag=True)

# Hypothetical selection of desirable sources
fs = 250
print("\n========================================================================================== Data 1 ==========================================================================================\n")
EEGplot(ND2, False, False, False, fs)

sources_ica_2, unmixing_matrix_ica_2, model_ica_2 = perform_source_extraction(ND2, method='ica') # Extract sources using ICA
EEGplot(sources_ica_2, 'ICA', False, fs=fs)
SelSources = [i for i in range(21) if i not in [13, 12, 7, 1, 20, 16, 17]]
# Extracting the mixing matrix A and components S for NewData1
A = ica_results["NewData2"]["mixing_matrix"]
S = ica_results["NewData2"]["components"]

# Reconstructing the de-noised signal
X_denoised = A[:, SelSources] @ S[SelSources, :]

# Plotting the original and de-noised signals for comparison
EEGplot(unmixing_matrix_ica_2[:, SelSources] @ sources_ica_2[SelSources, :], method=False, snr=False, fs=fs, denoised=True, flag=True)


# Hypothetical selection of desirable sources
fs = 250
print("\n========================================================================================== Data 1 ==========================================================================================\n")
EEGplot(ND3, False, False, False, fs)

sources_ica_3, unmixing_matrix_ica_3, model_ica_3 = perform_source_extraction(ND3, method='ica') # Extract sources using ICA
EEGplot(sources_ica_3, 'ICA', False, fs=fs)
SelSources = [i for i in range(21) if i not in [1, 7]]
# Extracting the mixing matrix A and components S for NewData1
A = ica_results["NewData3"]["mixing_matrix"]
S = ica_results["NewData3"]["components"]

# Reconstructing the de-noised signal
X_denoised = A[:, SelSources] @ S[SelSources, :]

# Plotting the original and de-noised signals for comparison
EEGplot(unmixing_matrix_ica_3[:, SelSources] @ sources_ica_3[SelSources, :], method=False, snr=False, fs=fs, denoised=True, flag=True)

# Hypothetical selection of desirable sources
fs = 250
print("\n========================================================================================== Data 1 ==========================================================================================\n")
EEGplot(ND4, False, False, False, fs)

sources_ica_4, unmixing_matrix_ica_4, model_ica_4 = perform_source_extraction(ND4, method='ica') # Extract sources using ICA
EEGplot(sources_ica_4, 'ICA', False, fs=fs)
SelSources = [i for i in range(21) if i not in [3, 6, 20]]
# Extracting the mixing matrix A and components S for NewData1
A = ica_results["NewData4"]["mixing_matrix"]
S = ica_results["NewData4"]["components"]

# Reconstructing the de-noised signal
X_denoised = A[:, SelSources] @ S[SelSources, :]

# Plotting the original and de-noised signals for comparison
EEGplot(unmixing_matrix_ica_4[:, SelSources] @ sources_ica_4[SelSources, :], method=False, snr=False, fs=fs, denoised=True, flag=True)

