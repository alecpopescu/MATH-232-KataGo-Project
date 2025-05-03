import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sgfmill import boards
from scipy.sparse import csgraph
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

input = pd.DataFrame()
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
gamelen = []
input_temp = pd.DataFrame()
if __name__ == "__main__":
    csv_folder = "csvfiles"
    for fname in sorted(os.listdir(csv_folder))[1:50]:
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(csv_folder, fname)
        df = pd.read_csv(path)
        if not df['Board size'][0] == 19:
            continue
        snaps = build_snapshots_from_csv(path, size=19)
        # print(f"{fname}: {len(snaps)} snapshots, each {size}×{size}")
        # print(snaps[1])
        temp = pd.DataFrame()
        if not len(snaps) >= 200:
            continue
        if not len(snaps) <= 500:
            continue
        temp = []
        for snap in snaps:
            temp.append(snap.flatten())
            label.append(df['Final result'].iloc[-1])
            gamelen.append(len(snaps))
        temp_arr = np.array(temp).T
        temp_df = pd.DataFrame(temp_arr)
        input_temp = pd.concat([input_temp, temp_df], axis=1)

    input = input_temp

# sorted(os.listdir(csv_folder))
# gamelen

input = input.iloc[:,643:945]
# input

labels = []
for l in label:
    winner_margin = l.split('=')[1]   # e.g., 'B+40.5'
    winner, margin = winner_margin[0], winner_margin[2:]
    labels.append((winner, margin))

labels2 = pd.DataFrame(labels, columns=['winner','margin'])
# labels2

labels2['winner'] = labels2['winner'].replace({'B':0, 'W':1, '0':'NA'})
# labels2

labels = labels2['winner']
# labels

fin_input = input.T.reset_index(drop=True)
# fin_input


## Laplacian Eigenmaps and Diffusion Maps

def laplacian_eigenmaps(data, n_neighbors=10, max_components=50, final_components=2):
    """
    Perform Laplacian Eigenmaps dimensionality reduction

    Parameters:
        data (ndarray): Input data of shape (n_samples, n_features)
        n_neighbors (int): Number of neighbors for the graph construction
        n_components (int): Number of dimensions for the reduced space

    Returns:
        ndarray: Reduced data of shape (n_samples, n_components)
    """
    # Construct kNN graph
    knn_graph = kneighbors_graph(data, n_neighbors, mode='connectivity', include_self=False)

    # Compute graph Laplacian
    laplacian = csgraph.laplacian(knn_graph, normed=True)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(laplacian.toarray())

    # Plot eigenvalue spectrum to check (Elbow plot)
    plt.plot(np.arange(1, max_components+1), eigenvalues[1:max_components+1], 'o-')
    plt.title("Laplacian Eigenvalue Spectrum")
    plt.xlabel("Component index")
    plt.ylabel("Eigenvalue")
    plt.show()

    # Select eigenvectors corresponding to smallest non-zero eigenvalues
    return eigenvectors[:, 1:final_components+1]

def diffusion_maps(X, n_components=2, t=1):
    """
    Perform Diffusion Maps

    Parameters:
        X (ndarray): Input data of shape (n_samples, n_features)
        n_components (int): Number of diffusion components to return
        t (int): Diffusion time scale (power of eigenvalues)

    Returns:
        embedding (ndarray): Diffusion embedding of shape (n_samples, n_components)
    """
    # Compute pairwise Euclidean distances
    dists = squareform(pdist(X, metric='euclidean'))

    # Compute Gaussian kernel matrix
    sigma = np.median(dists)  # median distance as bandwidth
    K = np.exp(-dists**2 / (2 * sigma**2))

    # Normalize to create a row-stochastic Markov (transition) matrix
    D = np.sum(K, axis=1)
    P = K / D[:, np.newaxis]

    # Eigen-decomposition
    eigenvalues, eigenvectors = eigh(P)

    # Sort eigenvectors by their eigenvalues (descending order)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Form diffusion coordinates (skip first, constant eigenvector)
    diffusion_coords = eigenvectors[:, 1:n_components+1] * (eigenvalues[1:n_components+1] ** t)

    return diffusion_coords

if __name__ == "__main__":
    reduced_data = laplacian_eigenmaps(fin_input, n_neighbors=10, max_components=50, final_components=2)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='Spectral')
    plt.title("Laplacian Eigenmaps")
    plt.show()


## Plotting

unique_labels = np.unique(labels)
cmap = plt.cm.Spectral
highlight_game1 = reduced_data[0:295]
highlight_game2 = reduced_data[294:642]
num_moves = len(highlight_game2)
colors = plt.cm.plasma(np.linspace(0, 1, num_moves))
for i, label_val in enumerate(unique_labels):
    idx = labels == label_val
    plt.scatter(
        reduced_data[idx, 0],
        reduced_data[idx, 1],
        color='lightgrey',
        s=10
    )
for i in range(num_moves - 1):
    x_vals = [highlight_game2[i, 0], highlight_game2[i + 1, 0]]
    y_vals = [highlight_game2[i, 1], highlight_game2[i + 1, 1]]
    plt.plot(x_vals, y_vals, color=colors[i], linewidth=2)

# Overlay points if you still want individual dots
plt.scatter(highlight_game2[:, 0], highlight_game2[:, 1], color=colors, s=25)

plt.title("Laplacian Eigenmaps")
plt.xlim(-20*3.34205320e-17, 20*3.34205320e-17)
plt.ylim(-20*3.34205320e-17, 20*3.34205320e-17)
plt.show()

reduced_data = np.column_stack((reduced_data, labels))
# highlight_game2

if __name__ == "__main__":
    embedding = diffusion_maps(fin_input, n_components=2)

plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral')
plt.title("Diffusion Maps")
plt.show()

# highlight_game1

num_moves = len(embedding)
colors = plt.cm.plasma(np.linspace(0, 1, num_moves))

for i in range(num_moves - 1):
    x_vals = [embedding[i, 0], embedding[i + 1, 0]]
    y_vals = [embedding[i, 1], embedding[i + 1, 1]]
    plt.plot(x_vals, y_vals, color=colors[i], linewidth=2)

# Overlay points if you still want individual dots
plt.scatter(embedding[:, 0], embedding[:, 1], color=colors, s=25)

plt.title("Diffusion Maps")
plt.show()