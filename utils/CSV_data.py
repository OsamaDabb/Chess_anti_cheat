import chess

import numpy as np
import random
import torch
from tqdm import tqdm
import pandas as pd

from .Dataloading import *

def save_to_csv(file_path, N=250_000):
    boards, meta, elos, moves, _, _ = generate_data(file_path, N=N)
    
    with open("../Data/lichess_games.csv","w") as f:
        writer = csv.writer(f)
        for i in range(len(meta)):
            writer.writerow(list(boards[i].reshape(-1)) + [meta[i], elos[i], moves[i].uci()])
            
            
def load_from_csv(file_path, N, skip=0):
    
    boards = np.zeros((N, 896))
    
    meta = np.zeros((N))
    
    elos = np.zeros((N))
    
    moves = []
    
    game_lengths = np.zeros((N))    
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        
        for i in range(skip):
            _ = next(reader, None)
        
        for ind, row in tqdm(enumerate(reader), total=N):
            boards[ind] = np.array(row[0:896])
            
            meta[ind] = 1 if row[896] == "True" else 0
            
            elos[ind] = row[897]
            
            moves.append(chess.Move.from_uci(row[898]))
            
            if ind == N - 1:
                break
    
    boards = boards.astype(float).reshape((N,14,8,8))
            
    elos = elos.astype(int)
            
    return boards, meta, elos, moves
            
            
def fast_read_csv(file_path, N, skip=0):
    
    data = pd.read_csv(file_path, chunksize=899*32)
    
    boards = []
    
    meta = []
    
    elos = []
    
    moves = []
    
    for chunk in data:
    
        boards.append(chunk[:,0:897].to_numpy().reshape(-1, 14, 8, 8))

        meta.append(chunk[:,897].to_numpy())

        elos.append(chunk[:,898].to_numpy())

        moves.append(np.array([chess.Move.from_uci(x) for x in chunk[:,898]]))
    
    return boards, meta, elos, moves
            
            
            
            