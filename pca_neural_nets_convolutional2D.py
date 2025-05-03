import os
import numpy as np
import pandas as pd
from sgfmill import boards
import sys
import pickle
import runner
import torch
from importlib import reload
reload(runner)
from runner import CustomRunner

os.chdir('/Users/jaketodd/MATH 232 Project')
size = 19
def move_str_to_coords(move_str, size):
    if move_str.lower() == "pass":
        return None
    letter = move_str[0]
    num    = int(move_str[1:])
    col = ord(letter) - ord('A')
    # Ith column is skipped
    if letter > 'I':
        col -= 1
    row = size - num
    return row, col

def build_snapshots_from_csv(csv_path, size=19):
    df = pd.read_csv(csv_path)
    board = boards.Board(size)
    snapshots = []

    for _, row in df.iterrows():
        move = row["Move"]
        player = row["Player"]
        color = 'b' if player.lower().startswith('b') else 'w'

        coords = move_str_to_coords(move, size)
        if coords is not None:
            if board.get(coords[0], coords[1]) is None:
                board.play(coords[0], coords[1], color)
        # snapshot the entire board
        arr = np.zeros((size, size), dtype=int)
        for r in range(size):
            for c in range(size):
                occ = board.get(r, c)
                if occ == 'b':
                    arr[r, c] =  1
                elif occ == 'w':
                    arr[r, c] = -1
        snapshots.append(arr.copy())

    return snapshots
label = []
tensors = []
if __name__ == "__main__":
    csv_folder = "csvfiles"
    for fname in sorted(os.listdir(csv_folder))[1:5300]:
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(csv_folder, fname)
        df = pd.read_csv(path)
        if not df['Board size'][0] == 19:
            continue
        snaps = build_snapshots_from_csv(path, size=19)
        if not len(snaps) >= 100:
            continue
        label.append(df['Final result'].iloc[-1])
        tempvec = []
        tensor_snaps = snaps[-51:-1]
        tensor_snaps_black = []
        tensor_snaps_white = []
        tensor_16 = []
        for i in range(50):
            tensor_snaps_black.append((-1*(tensor_snaps[-(i+1)] == -1).astype(int)))
            tensor_snaps_white.append((tensor_snaps[-(i+1)] == 1).astype(int))
        tensor_black = np.stack(tensor_snaps_black, axis = 0)
        tensor_white = np.stack(tensor_snaps_white, axis = 0)
        tensors.append(torch.tensor(np.stack((tensor_black, tensor_white), axis = 0),dtype=torch.float32))

batch_encoded = torch.stack(tensors)

batch_encoded = batch_encoded.reshape(-1, 100, 19, 19)
batch_encoded.shape

tensor_black.shape
tensor_snaps_white

labels = []
for l in label:
    winner_margin = l.split('=')[1]   # e.g., 'B+40.5'
    winner, margin = winner_margin[0], winner_margin[2:]
    labels.append((winner, margin))

labels2
labels2 = pd.DataFrame(labels, columns=['winner','margin'])

#encode black as 0 and white as 1
labels2['winner'] = labels2['winner'].replace({'B':0, 'W':1, '0':'NA'})

na_indices = labels2[(labels2['winner'] != 0) & (labels2['winner'] != 1)].index

labels2 = labels2.drop(na_indices).reset_index(drop=True)
labels = labels2['winner']

filtered_tensors = [tensor for i, tensor in enumerate(tensors) if i not in na_indices]
batch_encoded = torch.stack(filtered_tensors)

batch_encoded = batch_encoded.reshape(-1, 100, 19, 19)
batch_encoded.shape

labels.shape

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.decomposition import PCA

class ConvNet(nn.Module):
    def __init__(self, in_channels):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(100, 64, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device('cpu')
labels.value_counts()
labels = np.array(labels).astype(float)
labels.shape
labels = torch.tensor(labels, dtype=torch.float32)
inputs = batch_encoded

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=123)

x_train.shape

from sklearn.model_selection import KFold
from catalyst import dl, metrics
from catalyst.utils import set_global_seed
from catalyst.callbacks.checkpoint import CheckpointCallback
import torch.nn.functional as F

kf = KFold(n_splits=5, random_state=123, shuffle=True)
fold_metrics = []
dataset = TensorDataset(torch.tensor(x_train).to(torch.float32), torch.tensor(y_train).unsqueeze(1).to(torch.float32))
all_fold_train_losses = []
all_fold_val_losses = []

# Set global random seed
set_global_seed(123)
criterion = torch.nn.MSELoss()

for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
    model = ConvNet(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-2)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=False)

    runner = CustomRunner()

    runner.train(
        model = model,
        optimizer=optimizer,
        criterion=criterion,
        loaders={"train": train_loader, "valid": val_loader},
        num_epochs=100,
        #callbacks=callbacks_list,
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        seed=123
    )

    fold_train_losses = runner.train_losses
    fold_val_losses = runner.valid_losses

    fold_metrics.append(runner.loader_metrics["loss"])
    all_fold_train_losses.append(fold_train_losses)
    all_fold_val_losses.append(fold_val_losses)


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fold_train_losses, label='train')
plt.plot(fold_val_losses, label='validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

label = []
if __name__ == "__main__":
    csv_folder = "csvfiles"
    for fname in sorted(os.listdir(csv_folder))[1001:1200]:
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(csv_folder, fname)
        df = pd.read_csv(path)
        snaps = build_snapshots_from_csv(path, size=19)
        tempvec = []
        if not len(snaps) >= 200:
            continue
        label.append(df['Final result'].iloc[-1])
        for i in range(200):
            for j in range(len(snaps[i].flatten())):
                tempvec.append(snaps[i].flatten()[j])
        input[fname] = tempvec

test = input.iloc[:, 738:]

labels = []
for l in label:
    winner_margin = l.split('=')[1]   # e.g., 'B+40.5'
    winner, margin = winner_margin[0], winner_margin[2:]
    labels.append((winner, margin))

labels2 = pd.DataFrame(labels, columns=['winner','margin'])
labels2['winner'] = labels2['winner'].replace({'B':0, 'W':1, '0':'NA'})
na_indices = labels2[labels2['winner'] == 'NA'].index
labels2 = labels2.drop(na_indices).reset_index(drop=True)
labels = labels2['winner']
fin_test = test.T.reset_index(drop=True)
fin_test = fin_test.drop(na_indices).reset_index(drop=True)
fin_test

test
na_indices
labels[0:10]
test_labels

fin_test = np.array(fin_test)
labels = np.array(labels)
labels = labels.astype(float)
fin_test = torch.tensor(fin_test, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)
test_dataset = TensorDataset(torch.tensor(fin_test).to(torch.float32), torch.tensor(labels).unsqueeze(1).to(torch.float32))
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4, shuffle=True)

model.eval()
test_preds, test_rawpreds, test_labels = [], [], []
# Iterate over test_loader to collect predictions and labels
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = model(inputs)
        preds = (outputs > 0.5).long()
        test_rawpreds.append(outputs.cpu())
        test_preds.append(preds.cpu())
        test_labels.append(labels.cpu())

test_rawpreds = torch.cat(test_rawpreds).numpy()
test_preds = torch.cat(test_preds).numpy()
test_labels = torch.cat(test_labels).numpy()

test_rawpreds
test_preds
len(test_labels)
len(test_preds)
test_labels

matches = (test_preds == test_labels)
percentage_equal = np.mean(matches) * 100
percentage_equal