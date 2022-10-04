from typing import Dict, List, Any
import chess
import sys
import time
import evaluate


class engine(object):




    def __init__(self, evalu):
        self.evalu = evalu
        
        self.debug_info: Dict[str, Any] = {}
        self.MATE_SCORE     = 1000000000
        self.MATE_THRESHOLD =  999000000

    def next_move(self, depth: int, board: chess.Board, debug=True) -> chess.Move:
        """
        What is the next best move?
        """
        self.debug_info.clear()
        self.debug_info["nodes"] = 0
        t0 = time.time()

        move = self.minimax_root(depth, board)

        self.debug_info["time"] = time.time() - t0
        if debug == True:
            print(f"info {self.debug_info}")
        return move


    def get_ordered_moves(self, board: chess.Board) -> List[chess.Move]:
        """
        Get legal moves.
        Attempt to sort moves by best to worst.
        Use piece values (and positional gains/losses) to weight captures.
        """

        def orderer(move):
            return self.evalu.move_value(board, move)

        in_order = sorted(
            board.legal_moves, key=orderer, reverse=(board.turn == chess.WHITE)
        )
        return list(in_order)


    def minimax_root(self, depth: int, board: chess.Board) -> chess.Move:
        """
        What is the highest value move per our evaluation function?
        """
        # White always wants to maximize (and black to minimize)
        # the board score according to evaluate_board()
        maximize = board.turn == chess.WHITE
        best_move = -float("inf")
        if not maximize:
            best_move = float("inf")

        moves = self.get_ordered_moves(board)
        best_move_found = moves[0]

        for move in moves:
            board.push(move)
            # Checking if draw can be claimed at this level, because the threefold repetition check
            # can be expensive. This should help the bot avoid a draw if it's not favorable
            # https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.can_claim_draw
            if board.can_claim_draw():
                value = 0.0
            else:
                value = self.minimax(depth - 1, board, -float("inf"), float("inf"), not maximize)
            board.pop()
            if maximize and value >= best_move:
                best_move = value
                best_move_found = move
            elif not maximize and value <= best_move:
                best_move = value
                best_move_found = move

        return best_move_found


    def minimax(self,
        depth: int,
        board: chess.Board,
        alpha: float,
        beta: float,
        is_maximising_player: bool,
    ) -> float:
        """
        Core minimax logic.
        https://en.wikipedia.org/wiki/Minimax
        """
        self.debug_info["nodes"] += 1

        if board.is_variant_end():
            # The previous move resulted in checkmate
            if board.is_variant_win() and is_maximising_player:
                return self.MATE_SCORE
            if board.is_variant_loss() and is_maximising_player:
                return -self.MATE_SCORE
        # When the game is over and it's not a checkmate it's a draw
        # In this case, don't evaluate. Just return a neutral result: zero
        if board.is_variant_draw():
            return 0

        if depth == 0:
            return self.evalu.evaluate_board(board)

        if is_maximising_player:
            best_move = -float("inf")
            moves = self.get_ordered_moves(board)
            for move in moves:
                board.push(move)
                curr_move = self.minimax(depth - 1, board, alpha, beta, not is_maximising_player)
                # Each ply after a checkmate is slower, so they get ranked slightly less
                # We want the fastest mate!
                if curr_move > self.MATE_THRESHOLD:
                    curr_move -= 1
                elif curr_move < -self.MATE_THRESHOLD:
                    curr_move += 1
                best_move = max(
                    best_move,
                    curr_move,
                )
                board.pop()
                alpha = max(alpha, best_move)
                if beta <= alpha:
                    return best_move
            return best_move
        else:
            best_move = float("inf")
            moves = self.get_ordered_moves(board)
            for move in moves:
                board.push(move)
                curr_move = self.minimax(depth - 1, board, alpha, beta, not is_maximising_player)
                if curr_move > self.MATE_THRESHOLD:
                    curr_move -= 1
                elif curr_move < -self.MATE_THRESHOLD:
                    curr_move += 1
                best_move = min(
                    best_move,
                    curr_move,
                )
                board.pop()
                beta = min(beta, best_move)
                if beta <= alpha:
                    return best_move
            return best_move
