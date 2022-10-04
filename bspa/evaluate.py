import chess
import numpy as np
mapper = {}
mapper["PieceValue"] = 0
mapper["Pawn"] = 1
mapper["Knight"] = 2
mapper["Bishop"] = 3
mapper["Rook"] = 4
mapper["Queen"] = 5
mapper["King"] = 6



class evaluate(object):
    def __init__(self, theta):
        self.piece_value = {
            chess.PAWN: theta[mapper["PieceValue"]][0],
            chess.ROOK: theta[mapper["PieceValue"]][1],
            chess.KNIGHT: theta[mapper["PieceValue"]][2],
            chess.BISHOP: theta[mapper["PieceValue"]][3],
            chess.QUEEN: theta[mapper["PieceValue"]][4],
            chess.KING: theta[mapper["PieceValue"]][5]
        }
        
        self.pawnEvalWhite = theta[mapper["Pawn"]]
        self.pawnEvalBlack = np.flip(self.pawnEvalWhite)
        
        self.knightEval = theta[mapper["Knight"]]
        
        self.bishopEvalWhite = theta[mapper["Bishop"]]
        self.bishopEvalBlack = np.flip(self.bishopEvalWhite)
        
        self.rookEvalWhite = theta[mapper["Rook"]]
        self.rookEvalBlack = np.flip(self.rookEvalWhite)
        
        self.queenEval = theta[mapper["Queen"]]
        
        self.kingEvalWhite = theta[mapper["King"]]
        self.kingEvalBlack = np.flip(self.kingEvalWhite)
        

    def move_value(self, board: chess.Board, move: chess.Move) -> float:
        """
        How good is a move?
        A promotion is great.
        A weaker piece taking a stronger piece is good.
        A stronger piece taking a weaker piece is bad.
        Also consider the position change via piece-square table.
        """
        if move.promotion is not None:
            return -float("inf") if board.turn == chess.BLACK else float("inf")

        _piece = board.piece_at(move.from_square)
        if _piece:
            _from_value = self.evaluate_piece(_piece, move.from_square)
            _to_value = self.evaluate_piece(_piece, move.to_square)
            position_change = _to_value - _from_value
        else:
            raise Exception(f"A piece was expected at {move.from_square}")

        capture_value = 0.0
        if board.is_capture(move):
            capture_value = self.evaluate_capture(board, move)

        current_move_value = capture_value + position_change
        if board.turn == chess.BLACK:
            current_move_value = -current_move_value

        return current_move_value


    def evaluate_capture(self, board: chess.Board, move: chess.Move) -> float:
        """
        Given a capturing move, weight the trade being made.
        """
        if board.is_en_passant(move):
            return self.piece_value[chess.PAWN]
        _to = board.piece_at(move.to_square)
        _from = board.piece_at(move.from_square)
        if _to is None or _from is None:
            raise Exception(
                f"Pieces were expected at _both_ {move.to_square} and {move.from_square}"
            )
        return self.piece_value[_to.piece_type] - self.piece_value[_from.piece_type]


    def evaluate_piece(self, piece: chess.Piece, square: chess.Square) -> int:
        piece_type = piece.piece_type
        mapping = []
        if piece_type == chess.PAWN:
            mapping = self.pawnEvalWhite if piece.color == chess.WHITE else self.pawnEvalBlack
        if piece_type == chess.KNIGHT:
            mapping = self.knightEval
        if piece_type == chess.BISHOP:
            mapping = self.bishopEvalWhite if piece.color == chess.WHITE else self.bishopEvalBlack
        if piece_type == chess.ROOK:
            mapping = self.rookEvalWhite if piece.color == chess.WHITE else self.rookEvalBlack
        if piece_type == chess.QUEEN:
            mapping = self.queenEval
        if piece_type == chess.KING:
            mapping = self.kingEvalWhite if piece.color == chess.WHITE else self. kingEvalBlack

        return mapping[square]


    def evaluate_board(self, board: chess.Board) -> float:
        """
        Evaluates the full board and determines which player is in a most favorable position.
        The sign indicates the side:
            (+) for white
            (-) for black
        The magnitude, how big of an advantage that player has
        """
        total = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue

            value = self.piece_value[piece.piece_type] + self.evaluate_piece(piece, square)
            
            total += value if piece.color == chess.WHITE else -value

        return total

#     def eval(self, board):
#     	US = chess.BLACK
#     	THEM = chess.WHITE
#     	if(board.turn == chess.WHITE):
#     		US = chess.WHITE
#     		THEM = chess.BLACK
#     	moves = [[x.from_square, x.to_square] for x in board.generate_legal_moves()]
# 
#     	safe = np.zeros(shape=(2, 8, 8))
#     	attackBy = np.zeros(shape=(2, 8, 8))
#     	attackBy2 = np.zeros(shape=(2, 8, 8))
#     	blocked = np.zeros(shape=(8, 8))
#     	behind = np.zeros(shape=(8, 8))
#     	pawns = []
#     	
#     	for i in range(64):
#     		row = i // 8
#     		col = i % 8
#     		n_attack = len(board.attackers(THEM, i))
#     		if n_attack > 1:
#     			attackBy2[mapper[THEM]][row][col] = 1
#     		else: 
#     			attackBy2[mapper[THEM]][row][col] = 0
#     			
#     		if n_attack > 0:
#     		    attackBy[mapper[THEM]][row][col] = 1
#     		else: 
#     			attackBy[mapper[THEM]][row][col] = 0
#     			
#     		n_attack = len(board.attackers(US, i))
#     		if n_attack > 1:
#     			attackBy2[mapper[US]][row][col] = 1
#     		else: 
#     			attackBy2[mapper[US]][row][col] = 0
#     			
#     		if n_attack > 0:
#     		    attackBy[mapper[US]][row][col] = 1
#     		else: 
#     			attackBy[mapper[US]][row][col] = 0
#     		
#     		b =	board.is_attacked_by(THEM, i)
#     		piece = board.piece_at(i)
#     		if (US != board.color_at(i)) and not b:
#     			safe[mapper[US]][row][col] = 1
#     		else:
#     			safe[mapper[US]][row][col] = 0 
# 
#     		if US == board.color_at(i) and board.piece_type_at(i) == chess.PAWN:
#     			pawns.append(i)
# 
#     	movesfrom = [[x.from_square] for x in board.generate_legal_moves()]
#     	pawnmove = 0
#     	if US == chess.WHITE:
#     		pawnmove = 8
#     	else: 
#     		pawnmove = -8
# 
#     	for p in pawns:
#     		if  (not p in movesfrom) and board.piece_at(p + pawnmove) != None:
#     			blocked[p // 8][p % 8] = 1
#     		else:
#     			blocked[p // 8][p % 8] = 0
# 
#     		field = []
#     		for i in range(1, 4):
#     			field.append(p - pawnmove * i)
#     		for f in field:
#     			if f >= 0 and f <= 63:
#     				behind[f // 8][f % 8] = 1
#     	blocked = blocked.reshape((1, 8, 8))
#     	behind = behind.reshape((1, 8, 8))
#     	return np.vstack((safe, attackBy, attackBy2, blocked, behind)) 

