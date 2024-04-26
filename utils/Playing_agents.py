import torch
import numpy as np
import chess

from .Game_playing import *

def fen_to_board(fen):
    """
    Same as fen_to_board in Dataloading.py
    """

    mapper = {'p' : 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
              'P' : 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

    board = np.zeros((12,8,8))

    first_split = fen.split('/')
    

    for i, row in enumerate(first_split):
        col = 0
        for j, c in enumerate(row):
            
            if c == ' ':

                break

            elif c in '12345678': # think about encoding en passant (osama didn't want to. bitch)
                col += int(c)
                continue
            
            elif c == '-':
                print("HERE")
                continue

            board[mapper[c], i, col] = 1
            
            col += 1
        

            
    second_split = first_split[-1].split()

    white_turn = second_split[1] == 'w'
    
    
    if not white_turn:
        board = np.concatenate((board[6:], board[:6]), axis=0)
        
        board = np.flip(board, axis = 1)
        

    #meta_data = np.array([second_split[1] == 'w', 'K' in second_split[2], 'Q' in second_split[2],
    #            'k' in second_split[2], 'q' in second_split[2], second_split[4]], dtype = int)
    meta_data = white_turn

    return (board, meta_data)

def network_agent(board, network):
    """
    function that uses network to generate move output given board input.
    :param board: chess.Board
    :param network: nn.Module 768x1 -> 128x1
    :return: chess.Move
    """

    min_moves = []
    ila_moves = []

    for move in board.generate_legal_moves():
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()

        min_moves.append(move.from_square)
        ila_moves.append(move.to_square)
    
    white_turn = board.turn == chess.WHITE
    board, metadata = fen_to_board(board.fen())
    x = torch.tensor(np.copy(board),dtype=torch.float).view(-1, 768)
    minn, ila = network(x.view(1, -1))
    
    if not white_turn: 
        
        minn = minn[:,index_map]
        ila = ila[:,index_map]
        

    max_pair = (min_moves[0], ila_moves[0],
     (torch.log(minn[0][min_moves[0]]) + torch.log(ila[0][ila_moves[0]])))

    for ind, _ in enumerate(min_moves):

        prob = (torch.log(minn[0][min_moves[ind]]) + torch.log(ila[0][ila_moves[ind]]))

        if prob > max_pair[2]:

            max_pair = (min_moves[ind], ila_moves[ind], prob)
        
    return chess.Move(max_pair[0], max_pair[1])

def network_agent_prob(board, network):
    """
    Given a board and a network, selects the move probabilistically
    based off the agents outputs.

    :param board: chess.Board
    :param network: nn.Module 768x1 -> 128x1
    Returns: chess.Move
    """
    legal_moves = list(board.generate_legal_moves())

    white_turn = board.turn == chess.WHITE
    bitboard, metadata = fen_to_board(board.fen())
    x = torch.tensor(np.copy(bitboard),dtype=torch.float).view(-1, 768)
    minn, ila = network(x.view(1, -1))
    
    if not white_turn: 
        
        minn = minn[:,index_map]
        ila = ila[:,index_map]

    move_probs = []

    for ind, move in enumerate(legal_moves):

        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()

        prob = minn[0][move.from_square].cpu().detach().numpy() * \
        ila[0][move.to_square].cpu().detach().numpy()

        move_probs.append(prob)
        
        
    move_probs = np.array(move_probs)
    
    move_probs = move_probs - np.max(move_probs)
    
    move_probs = np.exp(move_probs) / np.sum(np.exp(move_probs))
    

    return random.choices(legal_moves, weights=move_probs, k=1)[0]


def network_agent_conv(board, network):
    """
    Similar to network_agent but for convolutional network
    :param board: chess.Board
    :param network: nn.Module 12x8x8 -> 128x1
    :return:
    """

    min_moves = []
    ila_moves = []

    for move in board.generate_legal_moves():
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()

        min_moves.append(move.from_square)
        ila_moves.append(move.to_square)
    
    white_turn = board.turn == chess.WHITE
    board, metadata = fen_to_board(board.fen())
    x = torch.tensor(np.copy(board),dtype=torch.float).view(1, 12, 8, 8)
    minn, ila = network(x)
    
    if not white_turn: 
        
        minn = minn[:,index_map]
        ila = ila[:,index_map]
        

    max_pair = (min_moves[0], ila_moves[0],
     (torch.log(minn[0][min_moves[0]]) + torch.log(ila[0][ila_moves[0]])))

    for ind, _ in enumerate(min_moves):

        prob = (torch.log(minn[0][min_moves[ind]]) + torch.log(ila[0][ila_moves[ind]]))

        if prob > max_pair[2]:

            max_pair = (min_moves[ind], ila_moves[ind], prob)
        
    return chess.Move(max_pair[0], max_pair[1])

def network_agent_prob_conv(board, network, with_attacks=True):
    """
    Given a board and a network, selects the move probabilistically
    based off the agents outputs.
    Inputs:
    board: chess.Board
    network: nn.Module chess.board -> 128x1
    Returns:
    chess.Move object
    """
    legal_moves = list(board.generate_legal_moves())

    white_turn = board.turn == chess.WHITE
    bitboard, metadata = fen_to_board(board.fen())
    channels = 12
    
    
    if with_attacks:
        attacks = np.zeros((2,64))
        is_white_turn = 1 if board.turn else 0
        for i in range(64):
            if board.is_attacked_by(chess.WHITE, i):
                attacks[(1-is_white_turn),i] = 1
            if board.is_attacked_by(chess.BLACK, i):
                attacks[is_white_turn,i] = 1
        bitboard = np.concatenate((bitboard, attacks.reshape(2,8,8)), axis=0)
        channels=14
        
    x = torch.tensor(np.copy(bitboard),dtype=torch.float).view(1, channels, 8, 8)
    prediction = network(x)

    if type(prediction) == tuple:

        minn, ila = prediction

    else:

        minn, ila = prediction[:, :64], prediction[:,64:]

    if not white_turn: 
        
        minn = minn[:,index_map]
        ila = ila[:,index_map]

    move_probs = []

    for ind, move in enumerate(legal_moves):
        
        if move.promotion is not None and move.promotion != 5:
            move_probs.append(float("-inf"))
            continue

        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()

        prob = minn[0][move.from_square].cpu().detach().numpy() * \
        ila[0][move.to_square].cpu().detach().numpy()

        move_probs.append(prob)
        
        
    move_probs = np.array(move_probs)
    
    move_probs = move_probs - np.max(move_probs)
    
    move_probs = np.exp(move_probs) / np.sum(np.exp(move_probs))
    

    return random.choices(legal_moves, weights=move_probs, k=1)[0]

def network_agent_prob_conv_exhaustive(board, network):
    """
    Given a board and a network, selects the move probabilistically
    based off the agents outputs.
    Inputs:
    board: chess.Board
    network: func chess.board -> 1x64 probability from tens, 1x64 probability to tens
    Returns:
    chess.Move object
    """
    legal_moves = list(board.generate_legal_moves())

    white_turn = board.turn == chess.WHITE
    bitboard, metadata = fen_to_board(board.fen())
    
    
    attacks = np.zeros((2,64))
    is_white_turn = 1 if board.turn else 0
    for i in range(64):
        if board.is_attacked_by(chess.WHITE, i):
            attacks[(1-is_white_turn),i] = 1
        if board.is_attacked_by(chess.BLACK, i):
            attacks[is_white_turn,i] = 1
    bitboard = np.concatenate((bitboard, attacks.reshape(2,8,8)), axis=0)
    channels = 14
        
    x = torch.tensor(np.copy(bitboard),dtype=torch.float).view(1, channels, 8, 8)
    output = network(x).view(1, 64, 64)
    
    if not white_turn: 
        
        output = output[:, index_map, :]
        output = output[:, :, index_map]

    move_probs = []

    for ind, move in enumerate(legal_moves):
        
        if move.promotion is not None and move.promotion != 5:
            move_probs.append(float("-inf"))
            continue

        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()

        prob = output[0,move.from_square, move.to_square].detach().cpu()

        move_probs.append(prob)
        
    move_probs = np.array(move_probs)
    
    move_probs = move_probs - np.max(move_probs)
    
    move_probs = np.exp(move_probs) / np.sum(np.exp(move_probs))
    
    return random.choices(legal_moves, weights=move_probs, k=1)[0]