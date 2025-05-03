import os
import numpy as np
import pandas as pd
from sgfmill import boards
import sys
import pickle
import runner
from importlib import reload
reload(runner)
from runner import CustomRunner

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from catalyst import dl, metrics
from catalyst.utils import set_global_seed
from catalyst.callbacks.checkpoint import CheckpointCallback


input = pd.DataFrame()

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
if __name__ == "__main__":
    csv_folder = "csvfiles"
    for fname in sorted(os.listdir(csv_folder))[1:1000]:
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

# input

labels = []
for l in label:
    winner_margin = l.split('=')[1]   # e.g., 'B+40.5'
    winner, margin = winner_margin[0], winner_margin[2:]
    labels.append((winner, margin))

labels2 = pd.DataFrame(labels, columns=['winner','margin'])

#encode black as 0 and white as 1
labels2['winner'] = labels2['winner'].replace({'B':0, 'W':1, '0':'NA'})

na_indices = labels2[labels2['winner'] == 'NA'].index

labels2 = labels2.drop(na_indices).reset_index(drop=True)
labels = labels2['winner']
# input

fin_input = input.T.reset_index(drop=True)
fin_input = fin_input.drop(na_indices).reset_index(drop=True)
# fin_input

X = fin_input.values
pca = PCA(n_components=0.95)
input_pca = pca.fit_transform(X)
input_pca = input_pca.astype(np.float32)
print("PCA output:", input_pca.shape)

class NeuralNet(nn.Module):
    def __init__(self, n_inp, n_out, filter_num = 128):
        super(NeuralNet, self).__init__()
        self.activation = nn.Sigmoid()
        self.fc1 = nn.Linear(n_inp, filter_num)
        self.fc2 = nn.Linear(filter_num, 2*filter_num)
        self.fc3 = nn.Linear(2*filter_num, 2*filter_num)
        self.fc4 = nn.Linear(2*filter_num, filter_num)
        self.out = nn.Linear(filter_num, n_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.out(x)
        x = self.activation(x)
        return x

device = torch.device('cpu')
inputs = fin_input
inputs = np.array(inputs)
labels = np.array(labels)
inputs = inputs.astype(np.float32)
labels = labels.astype(np.float32)
# inputs.shape

inputs = torch.tensor(inputs, dtype=torch.float32)
input_pca = torch.tensor(input_pca, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=123)
pca_x_train, pca_x_test, y_train, y_test = train_test_split(input_pca, labels, test_size=0.2, random_state=123)

kf = KFold(n_splits=5, random_state=123, shuffle=True)
fold_metrics = []
dataset = TensorDataset(torch.tensor(x_train).to(torch.float32), torch.tensor(y_train).unsqueeze(1).to(torch.float32))

# Set global random seed
set_global_seed(123)

for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
    model = NeuralNet(x_train.shape[-1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    criterion = torch.nn.BCELoss()
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4, shuffle=False)

    runner = CustomRunner()

    runner.train(
        model = model,
        optimizer=optimizer, 
        criterion=criterion,
        loaders={"train": train_loader, "valid": val_loader},
        num_epochs=20,
        #callbacks=callbacks_list,
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        seed=123
    )

    fold_metrics.append(runner.loader_metrics["loss"])

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

# fin_test
# test
# na_indices
# labels[0:10]
# test_labels

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

def run_cv(X_tr, y_tr, desc):

    # 1) Build a single TensorDataset
    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    )
    # 2) Prepare KFold
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    set_global_seed(123)   
    fold_losses = []
    model     = NeuralNet(X_tr.shape[1], 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr)):
        train_ds = Subset(ds, train_idx)
        val_ds   = Subset(ds, val_idx)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
        runner    = CustomRunner()
        runner.train(
            model            = model,
            optimizer        = optimizer,
            criterion        = criterion,
            loaders          = {"train": train_loader, "valid": val_loader},
            num_epochs       = 50,
            logdir           = f"./logs/{desc}/fold{fold}",
            valid_loader     = "valid",
            valid_metric     = "loss",
            minimize_valid_metric = True,
            seed             = 123
        )
        # record the final validation loss
        fold_losses.append(runner.loader_metrics["loss"])

    print(f"{desc} CV losses:", fold_losses, "→ mean:", np.mean(fold_losses))
    return fold_losses

# — run CV on RAW inputs —
raw_losses = run_cv(x_train, y_train, desc="raw_input")

# — run CV on PCA‐reduced inputs —
pca_losses = run_cv(pca_x_train, y_train, desc="pca_input")