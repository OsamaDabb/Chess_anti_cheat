"""
File responsible for chess game-playing functions, including visualizing games, and competing differing models to
measure W/L etc.
"""

import chess
import chess.svg
import numpy as np
import random
import cv2
import torch
import csv
from IPython.display import display, SVG
from tqdm import tqdm

def play_game(agent_1, agent_2, board = None, fen = True, show_boards = False):
    """
    Plays game between agent_1 (white) and agent_2 (black), that begins at board.

    :param agent_1: function: (chess.Board -> chess.Move)
    :param agent_2: function (chess.Board -> chess.Move)
    :param board: String, FEN of starting board, defaults to initial starting board
    :param fen: whether to return the game FENs
    :param show_boards: whether to display the game if played between two AIs
    :return: 1 if white won, 0 if black won, else 0.5 & potentially game FENs
    """
    if board == None:
      board = chess.Board()

    history = [str(board)]
    prev_player = None
    
    if show_boards:
        board_img = chess.svg.board(board, orientation = board.turn, size=300)
        display(SVG(board_img))    

    while not board.is_game_over():
        if prev_player == board.turn:
            print("Error, be worried")
        prev_player == board.turn
        
        move = agent_1(board) if board.turn else agent_2(board)

        board.push(move)
        
        if show_boards:
            input("")
            board_img = chess.svg.board(board, orientation = 1, size=300, lastmove = board.peek())
            display(SVG(board_img))    
        

        history.append(str(board))

    outcome = board.outcome()

    return (0.5 if outcome.winner is None else outcome.winner, history)

random_agent = lambda x : random.choice(list(x.generate_legal_moves()))

def human_agent(board):
    """
    agent that allows human to play and visualize board, expects moves in the UCI notation

    :param board: chess.Board
    :return: chess.Move
    """

    if board.fen() == chess.STARTING_FEN:

        board_img = chess.svg.board(board, orientation = board.turn, size=300)

    else:

        board_img = chess.svg.board(board,orientation = board.turn, lastmove = board.peek(), size=300)
    
    display(SVG(board_img))

    user_move = input('Your move: ')
    while True:
        try:
            return board.parse_san(user_move)
        except:
            if user_move == 'quit': break
            user_move = input('Your move: ')
    
    return None


def test_against(agent,opponent=random_agent, N=500, play_both=True, unique=False):
    """
    Given two agents and a number of games N, calculates the
    W/D/L of that agent. Defaults to random_agent as opponent
    """
    wins = 0
    draws = 0
    losses = 0
    for i in tqdm(range(N//2)):
        if unique:
            x = play_unique_game(agent, opponent)[0]
        else:
            x = play_game(agent, opponent)[0]
        if x == 0:
            losses += 1
        elif x == 0.5:
            draws += 1
        else:
            wins += 1
    if not play_both:
        return wins, losses, draws, wins/N
    for _ in tqdm(range(N//2)):
        if unique:
            x = (1-play_unique_game(opponent, agent)[0])
        else:
            x = (1-play_game(opponent, agent)[0])
        if x == 0:
            losses += 1
        elif x == 0.5:
            draws += 1
        else:
            wins += 1

    return wins, losses, draws, wins/N

def convert_to_black(move): 
    
    return (63 - move) // 8 * 8 + move % 8

index_map = torch.tensor([convert_to_black(i) for i in range(64)])


def play_unique_game(agent_1, agent_2, board = None, fen = True, show_boards = False, N=4):
    """
    Like play_game, except begins the board after N random moves for unique game
    """
    if board == None:
      board = chess.Board()
    
    for i in range(N):
        move = random_agent(board)
        board.push(move)

    history = [str(board)]
    prev_player = None
    
    if show_boards:
        board_img = chess.svg.board(board, orientation = board.turn, size=300)
        display(SVG(board_img))    

    while not board.is_game_over():
        if prev_player == board.turn:
            print("Error, be worried")
        prev_player == board.turn
        
        move = agent_1(board) if board.turn else agent_2(board)

        board.push(move)
        
        if show_boards:
            input("")
            board_img = chess.svg.board(board, orientation = 1, size=300, lastmove = board.peek())
            display(SVG(board_img))    
        

        history.append(str(board))

    outcome = board.outcome()

    return (0.5 if outcome.winner is None else outcome.winner, history)