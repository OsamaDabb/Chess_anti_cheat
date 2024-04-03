import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

if torch.cuda.is_available():
    # Set default tensor type to CUDA tensors
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
else:
    
    torch.set_default_tensor_type(torch.FloatTensor)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ChessData(Dataset):

    def __init__(self, bitboards, white_turn, moves):

        self.bitboards = torch.tensor(bitboards, dtype = torch.float).view(-1, 768)

        labels = torch.zeros((self.bitboards.size(dim=0),128), dtype = torch.float)

        for ind, move in enumerate(moves):

            minn = move.from_square
            ila = move.to_square
            
            if not white_turn[ind]:
                minn = (63 - minn) // 8 * 8 + minn % 8
                ila = (63 - ila) // 8 * 8 + ila % 8
                
            labels[ind,minn] = 1
            labels[ind, ila + 64] = 1

        self.moves = labels

    def __len__(self):

        return self.moves.size(dim=0)

    def __getitem__(self, idx):
        

        return self.bitboards[idx], self.moves[idx]
    
    
class PiecewiseData(Dataset):

    def __init__(self, bitboards, pieces):

        self.bitboards = torch.tensor(bitboards, dtype = torch.float).view(-1, 768)
        
        self.pieces = torch.zeros((self.bitboards.size(dim=0), 6), dtype = torch.float)
        
        for ind, piece in enumerate(pieces):

            self.pieces[ind, piece-1] = 1 

    def __len__(self):

        return self.pieces.size(dim=0)

    def __getitem__(self, idx):

        return self.bitboards[idx], self.pieces[idx]

class ChessDataConv(Dataset):

    def __init__(self, bitboards, white_turn, moves):

        self.bitboards = torch.tensor(bitboards, dtype = torch.float).to(device)
        labels = torch.zeros((self.bitboards.size(dim=0),128), dtype = torch.float)

        for ind, move in enumerate(moves):

            minn = move.from_square
            ila = move.to_square

            if not white_turn[ind]:
                minn = (63 - minn) // 8 * 8 + minn % 8
                ila = (63 - ila) // 8 * 8 + ila % 8

            labels[ind,minn] = 1
            labels[ind, ila + 64] = 1

        self.moves = labels

    def __len__(self):

        return self.moves.size(dim=0)

    def __getitem__(self, idx):

        return self.bitboards[idx], self.moves[idx]

    
class ChessDataConvExhaustive(Dataset):

    def __init__(self, bitboards, white_turn, moves):

        self.bitboards = torch.tensor(bitboards, dtype = torch.float).to(device)
        labels = torch.zeros(self.bitboards.size(dim=0), dtype = torch.int)

        for ind, move in enumerate(moves):

            minn = move.from_square
            ila = move.to_square

            if not white_turn[ind]:
                minn = (63 - minn) // 8 * 8 + minn % 8
                ila = (63 - ila) // 8 * 8 + ila % 8

            labels[ind] =  minn * 64 + ila

        self.moves = labels

    def __len__(self):

        return self.moves.size(dim=0)

    def __getitem__(self, idx):
        
        move = torch.zeros(4096, dtype = torch.float)
        
        move[self.moves[idx]] = 1

        return self.bitboards[idx], move
