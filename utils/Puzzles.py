import time

import chess
import chess.svg

import numpy as np
import random
import cv2
import torch
from tqdm import tqdm
from IPython.display import display, SVG

from .Dataloading import fen_to_board

import csv

def generate_puzzles(N_puzzles=1_000, first_move=False):
    """
    generates puzzles for model evaluation, uses lichess_db_puzzles downloaded from their database

    :param N_puzzles
    :param first_move: whether to store the first move or the full sequence of moves
    :return: puzzle_data
    """

    FENs = []
    boards = []
    meta = []
    moves = []
    ratings = []
    
    with open("Data/lichess_db_puzzle.csv", "r") as f:
        reader = csv.reader(f)
        row = next(reader, None)
        
        for i in range(N_puzzles):
            row = next(reader, None)
            
            FENs.append(row[1])
            
            move = []
            
            if first_move:
                move = [chess.Move.from_uci(row[2].split()[0])]
            
            else:
                
                for i in row[2].split():

                    move.append(chess.Move.from_uci(i))
            
            moves.append(move)
            
            ratings.append(row[3])
            
    return FENs, moves, ratings
            
def shuffle_puzzles(puzzles):
    """
    short function shuffling the order of the given puzzles
    :param puzzles
    :return: puzzles
    """
    
    FENs, moves, ratings = puzzles
    
    
    combined = list(zip(FENs, moves, ratings))
    random.shuffle(combined)
    FENs, moves, ratings = zip(*combined)
    return list(FENs), list(moves), list(ratings)
            
def evaluate_on_puzzles(model, puzzles, max_rating=2500):
    """
    evaluates the given model on given set of puzzles. Returns accuracy % and rudimentary ranking to track success

    :param model: function chess.Board -> chess.Move
    :param puzzles
    :param max_rating: maximum rating of puzzle to evaluate with
    :return: accuracy, ranking
    """
    
    accuracy = 0
    
    rank = 1500
    
    N = len(puzzles[0])
    
    count = N
    
    for i in range(N):
        
        moves = puzzles[1][i]
        
        rating = int(puzzles[2][i])
        
        if rating > max_rating:
            count -= 1
            continue
            
                        
        board = chess.Board(puzzles[0][i])
        
        board.push(moves[0])
                        
        for ind, move in enumerate(moves[1::2]):
            
            if model(board) == move:
                board.push(move)
                if len(moves) > 2*ind + 2:
                    board.push(moves[2*ind+2])
                else:
                    accuracy += 1
                    rank += rating/100
            
            else:
                
                rank -= (2000 - rating)/100
                break
       
    return accuracy / count, int(rank)
        
        
        
def visualize_puzzles(model, puzzles, max_rating=2500):
    """
    Evaluate model effectiveness of puzzles while also visualizing them to console.

    :param model: function chess.Board -> chess.Move
    :param puzzles
    :param max_rating: max rating for evaluation
    :return: None
    """
    
    N = len(puzzles[0])
    
    for i in range(N):
        
        moves = puzzles[1][i]
        
        rating = int(puzzles[2][i])
        
        if rating > max_rating:
            
            continue
                        
        board = chess.Board(puzzles[0][i])
        
        board.push(moves[0])
        
        arrow = chess.svg.Arrow(moves[1].from_square, moves[1].to_square)
        
        board_img = chess.svg.board(board, orientation=board.turn, size=300, arrows = [arrow])
        display(SVG(board_img))
        
        time.sleep(1)
                        
        for ind, move in enumerate(moves[1::2]):
            
            network_move = model(board)
            board.push(network_move)
            
            board_img = chess.svg.board(board, orientation = not board.turn, size=300, lastmove=network_move, arrows = [arrow])
            display(SVG(board_img)) 
            
            if input("") != "":
                return
            
            if network_move == move:

                if len(moves) <= 2*ind + 2:
                    break
                
                else:
                    board.push(moves[2*ind+2])
                    
                    arrow = chess.svg.Arrow(moves[2*ind + 3].from_square, moves[2*ind + 3].to_square)
        
                    board_img = chess.svg.board(board, orientation = board.turn, size=300, arrows = [arrow])
                    display(SVG(board_img))
                
                    input("")

            else:
                
                break
