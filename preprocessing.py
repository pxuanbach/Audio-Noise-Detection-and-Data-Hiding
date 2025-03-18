import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import librosa
import warnings

warnings.filterwarnings('ignore')

#region DATA LOADING
# Read metadata file
data = pd.read_csv("urbansound8k\\metadata\\UrbanSound8K.csv")

x_train = []
x_test = []
y_train = []
y_test = []
N_FEATURES = 40 # N_MELS, N_CHROMA, N_MFCC

path = "urbansound8k\\audio\\fold"
#endregion

#region FEATURE EXTRACTION
for i in tqdm(range(len(data))):
    # Get file info
    fold_no = str(data.iloc[i]["fold"])
    file = data.iloc[i]["slice_file_name"]
    label = data.iloc[i]["classID"]
    filename = path + fold_no + "/" + file

    # Load audio file
    y, sr = librosa.load(filename)

    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_FEATURES).T, axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_FEATURES, fmax=8000).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(
        y=y, sr=sr, n_chroma=N_FEATURES).T, axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(
        y=y, sr=sr, n_chroma=N_FEATURES, bins_per_octave=N_FEATURES).T, axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(
        y=y, sr=sr, n_chroma=N_FEATURES, bins_per_octave=N_FEATURES).T, axis=0)

    # Stack features
    features = np.reshape(np.vstack((
        mfccs,
        melspectrogram,
        chroma_stft,
        chroma_cq,
        chroma_cens
    )), (N_FEATURES, 5))

    # Split into train/test
    if fold_no != '10':
        x_train.append(features)
        y_train.append(label)
    else:
        x_test.append(features)
        y_test.append(label)
#endregion

#region DATA PROCESSING
# Convert to numpy arrays
print('Length of Data: ', len(x_train) + len(x_test))
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Reshape into 2D for CSV storage
x_train_2d = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test_2d = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# Save arrays to CSV
np.savetxt("data/train_data.csv", x_train_2d, delimiter=",")
np.savetxt("data/test_data.csv", x_test_2d, delimiter=",")
np.savetxt("data/train_labels.csv", y_train, delimiter=",")
np.savetxt("data/test_labels.csv", y_test, delimiter=",")
#endregion
