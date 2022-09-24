import chess
import chess.variant
import numpy as np

mapper = {}
mapper["p"] = 0
mapper["r"] = 1
mapper["n"] = 2
mapper["b"] = 3
mapper["q"] = 4
mapper["k"] = 5
mapper["P"] = 0
mapper["R"] = 1
mapper["N"] = 2
mapper["B"] = 3
mapper["Q"] = 4
mapper["K"] = 5


class Env(object):
    def __init__(self):
        self.board = chess.variant.AntichessBoard()
        self.init_layer_board()
        self.turn = self.board.turn 
        
    def project_legal_moves(self):
        """
        Create a mask of legal actions
        Returns: np.ndarray with shape (64,64)
        """
        action_space = np.zeros(shape=(64, 64))
        moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
        for move in moves:
            action_space[move[0], move[1]] = 1
        return action_space

    def init_layer_board(self):
        """
        Initalize the numerical representation of the environment
        Returns:
        """
        self.layer_board = np.zeros(shape=(8, 8, 8))
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self.board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = mapper[piece.symbol()]
            self.layer_board[layer, row, col] = sign
        if self.board.turn:
                self.layer_board[6, :, :] = 1 / self.board.fullmove_number
        if self.board.can_claim_draw():
                self.layer_board[7, :, :] = 1
                
    def eval(self):
    	US = chess.BLACK
    	THEM = chess.WHITE
    	if(self.turn == chess.WHITE):
    		US = chess.WHITE
    		THEM = chess.BLACK
		
		moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]

    	safe = np.zeros(shape=(8, 8))
    	attackBy = np.zeros(shape=(2, 8, 8))
    	attackBy2 = np.zeros(shape=(2, 8, 8))
    	
    	for i in range(64):
    	
    		row = i // 8
            col = i % 8
    		
    		if len(self.board.attackers(THEM, i) > 1):
    			attackBy2[THEM][row][col] = True
    		else: 
    			attackBy2[THEM][row][col] = False
    		
    		b =	self.board.is_attacked_by(THEM, i)
    		piece = self.board.piece_at(i)
    		if US != color_at(i):
    			safe[row][col] = b
    		
    	pawns = pieces(chess.PAWN, US)
    	blocked = 0
    	for p in panwns:
    		if len(moves[p] == 0):
    			blocked += 1
    		

    def get_material_value(self):
        """
        Sums up the material balance using Reinfield values
        Returns: The material balance on the board
        """
        fen = self.board.fen()
        if(self.turn == chess.BLACK):
            pawns = -1 * fen.count("p")
            rooks = -5 * fen.count("r")
            minor = -3 * (fen.count("n") + fen.count("b"))
            queen = -9 * fen.count("q")
            king = -4 * fen.count("k")
            return pawns + rooks + minor + queen + king
        else:
            pawns = -1 * fen.count("P")
            rooks = -5 * fen.count("R")
            minor = -3 * (fen.count("N") + fen.count("B"))
            queen = -9 * fen.count("Q")
            king = -4 * fen.count("K")
            return pawns + rooks + minor + queen + king 
            
    def step(self, action):
        """
        Run a step
        Args:
            action: tuple of 2 integers
                Move from, Move to
        Returns:
            epsiode end: Boolean
                Whether the episode has ended
            reward: int
                Difference in material value after the move
        """
        self.turn = self.board.turn
        episode_end = False
        piece_balance_before = self.get_material_value()
        self.board.push(action)
        self.init_layer_board()
        piece_balance_after = self.get_material_value()
        capture_reward = piece_balance_after - piece_balance_before
        reward = 0 + capture_reward
        if self.board.is_game_over():
            reward = 0
            episode_end = True
        return episode_end, reward

