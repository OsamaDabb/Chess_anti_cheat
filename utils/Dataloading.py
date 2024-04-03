import chess
import chess.svg
import numpy as np
import random
import csv
from tqdm import tqdm
import chess.pgn

def fen_to_board(fen):

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

def generate_data(path,N=None, get_attacks=True, skip_games=0, model = None):
    
  negate = False

  #Opens the pgn file as game_file
  with open(path) as games_file:
    pgn_game = False
    boards = []
    meta = []
    elos = []
    moves= []
    ai_moves = []
    game_lengths = []
    fens = []
    
    use_opening = 1

    LOWER_ELO_LIMIT = 1400
    UPPER_ELO_LIMIT = 3500
    
    for i in tqdm(range(skip_games)):
        chess.pgn.read_game(games_file)

    #continually parsing through the file
    for i in tqdm(range(N)):

      pgn_game = chess.pgn.read_game(games_file)

      if pgn_game == None:
        return boards, meta, elos, moves
    

      #check rating of games
      while (pgn_game.headers["WhiteElo"] == "?") or LOWER_ELO_LIMIT > int(pgn_game.headers["WhiteElo"]) or int(pgn_game.headers["WhiteElo"]) > UPPER_ELO_LIMIT or pgn_game.headers["Termination"] != "Normal" or pgn_game.headers["TimeControl"] == "-" or int(pgn_game.headers["TimeControl"].split("+")[0]) < 100:
        pgn_game = chess.pgn.read_game(games_file)
        

      board = chess.Board()
    
      game_length = 0

      for move in pgn_game.mainline_moves():
          
          if game_length <= 10:
            use_opening = random.random()
        
          game_length += 1
            
          board_fen = board.fen()
          b, m = fen_to_board(board_fen)
        
          if game_length > 10 or use_opening < 0.1:
                
            if get_attacks:
                attacks = np.zeros((2,64))
                is_white_turn = 1 if board.turn else 0
                for i in range(64):
                    if board.is_attacked_by(chess.WHITE, i):
                        attacks[(1-is_white_turn),i] = 1
                    if board.is_attacked_by(chess.BLACK, i):
                        attacks[is_white_turn,i] = 1
                b = np.concatenate((b, attacks.reshape(2,8,8)), axis=0)
            
                
            boards.append(b)
            fens.append(board_fen)
            meta.append(m)
            elos.append(pgn_game.headers["WhiteElo"])
            moves.append(move)
            if model is not None:
                ai_moves.append(model(board))
            

          # Make the move on the board
          board.push(move)
          # Get the FEN string after the move
        
      game_lengths.append(game_length)

    boards = np.array(boards)
    meta = np.array(meta)
    elos = np.array(elos)
    moves = np.array(moves)
    ai_moves = np.array(ai_moves)
    game_lengths = np.array(game_lengths)
    fens = np.array(fens)

    return boards, meta, elos, moves, ai_moves, game_lengths, fens

'''

def generate_game_data(path, N=20_000, get_attacks=True, skip_games=0):
    
  negate = False

  #Opens the pgn file as game_file
  with open(path) as games_file:
    pgn_game = False
    boards = []
    meta = []
    elos = []
    moves= []
    pieces = []
    game_lengths = []
    
    use_opening = 1

    LOWER_ELO_LIMIT = 1400
    UPPER_ELO_LIMIT = 3500
    
    for i in tqdm(range(skip_games)):
        chess.pgn.read_game(games_file)

    #continually parsing through the file
    for i in tqdm(range(N)):

        pgn_game = chess.pgn.read_game(games_file)

        if pgn_game == None:
            return boards, meta, elos, moves
    

        #check rating of games
        while (pgn_game.headers["WhiteElo"] == "?") or LOWER_ELO_LIMIT > int(pgn_game.headers["WhiteElo"]) or int(pgn_game.headers["WhiteElo"]) > UPPER_ELO_LIMIT or pgn_game.headers["Termination"] != "Normal" or pgn_game.headers["TimeControl"] == "-" or int(pgn_game.headers["TimeControl"].split("+")[0]) < 100:
            pgn_game = chess.pgn.read_game(games_file)
        

        elos.append(pgn_game.headers["WhiteElo"])
            
        board = chess.Board()
        
        g_board = []
        g_move = []
        g_meta = []
        
    
        game_length = 0

        for move in pgn_game.mainline_moves():
        
            game_length += 1
            
            b, m = fen_to_board(board.fen())
        
                
            if get_attacks:
                attacks = np.zeros((2,64))
                is_white_turn = 1 if board.turn else 0
                for i in range(64):
                    if board.is_attacked_by(chess.WHITE, i):
                        attacks[(1-is_white_turn),i] = 1
                    if board.is_attacked_by(chess.BLACK, i):
                        attacks[is_white_turn,i] = 1
                b = np.concatenate((b, attacks.reshape(2,8,8)), axis=0)
                                
            
            g_board.append(b)
            g_meta.append(m)
            g_move.append(move)

          # Make the move on the board
          board.push(move)

        
        boards.append(g_board)
        meta.append(g_meta)
        moves.append(g_move)
        
        game_lengths.append(game_length)

    boards = np.array(boards)
    meta = np.array(meta)
    elos = np.array(elos)
    moves = np.array(moves)
    pieces= np.array(pieces)
    game_lengths = np.array(game_lengths)

    return boards, meta, elos, moves, pieces, game_lengths
    
'''