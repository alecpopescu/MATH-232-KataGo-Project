import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sgfmill import sgf
from sgfmill import boards
import os
from os import listdir

files = listdir('/Users/jaketodd/Downloads/2025-04-01sgfs 2/kata1-b28c512nbt-s8476434688-d4668249792/')
filenames = ['/Users/jaketodd/Downloads/2025-04-01sgfs 2/kata1-b28c512nbt-s8476434688-d4668249792/' + files[i] for i in np.arange(0, len(files))]
# filenames

for j in np.arange(5500):

    sgf_file = filenames[j]

    # Load the SGF game
    try:
        with open(sgf_file, "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        print("SGF file loaded successfully.")
    except Exception as e:
        print(f"Error loading SGF file: {e}")
        exit()

    # Get board size (e.g., 19x19)
    board_size = game.get_size()
    print(f"Board size: {board_size}x{board_size}")

    # Get the main sequence of moves
    main_sequence = game.get_main_sequence()
    print(f"Number of moves in the main sequence: {len(main_sequence)}")

    if len(main_sequence) == 0:
        print("No moves found in the SGF file.")
        exit()

    # initialize variables for storage in a datatable
    movenum = []
    moveplayer = []
    moveseq = []
    whitewin = []
    blackwin = []
    nores = []
    expscore = []
    visits = []
    weights = []
    result = []

    # Iterate through the moves and print them
    for i, node in enumerate(main_sequence):
        movenum.append(i+1)
        move = node.get_move()
        player, coords = move
        if player == 'b':
            moveplayer.append('Black')
        else:
            moveplayer.append('White')
        comment = node.get("C") if "C" in node.properties() else None

        if coords is not None:
            col, row = coords
            move_str = f"{chr(col + ord('A') + (1 if col >= 8 else 0))}{board_size - row}"
            moveseq.append(move_str)
        else:
            move_str = "pass"
            moveseq.append(move_str)

        print(f"Move {i+1}: {'Black' if player == 'b' else 'White'} -> {move_str}")

        if comment:
            print(f"   Comment: {comment}")
            if i == 0:
                whitewin.append('NA')
                blackwin.append('NA')
                nores.append('NA')
                expscore.append('NA')
                visits.append('NA')
                weights.append('NA')
                result.append('NA')
            elif i == len(main_sequence)-1:
                com = comment.split()
                result.append(com[6])
                whitewin.append(com[0])
                blackwin.append(com[1])
                nores.append(com[2])
                expscore.append(com[3])
                visits.append(com[4])
                weights.append(com[5])
            else:
                com = comment.split()
                whitewin.append(com[0])
                blackwin.append(com[1])
                nores.append(com[2])
                expscore.append(com[3])
                visits.append(com[4])
                weights.append(com[5])
                result.append('NA')
        else:
            whitewin.append('NA')
            blackwin.append('NA')
            nores.append('NA')
            expscore.append('NA')
            visits.append('NA')
            weights.append('NA')
            result.append('NA')

    data = {"Board size": [board_size]*len(movenum), "Move number": movenum, "Player": moveplayer, "Move": moveseq, "White win": whitewin, "Black win": blackwin, "No result": nores, "Final score": expscore, "Visits": visits, "Weight": weights, "Final result": result}
    globals()[f"df{j}"] = pd.DataFrame(data)
    globals()[f"df{j}"].to_csv(os.getcwd() + '/csvfiles/' + 'game' + str(j) + '.csv')