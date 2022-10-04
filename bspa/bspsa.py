import chess
import chess.pgn
import chess.variant
import numpy as np
from evaluate import evaluate
from evaluate import mapper
from engine import engine
import pandas as pd



def chess_match(enginePlus, engineMinus):
	board = chess.variant.AntichessBoard()
	while(not board.is_game_over()):
		move = enginePlus.next_move(2, board, True)
		board.push(move)
		print(board)
		if board.is_game_over():
			break
		move = engineMinus.next_move(2, board, True)
		board.push(move)
		print(board)
	if board.result() == "1-0":
		return 1
	elif board.result() == "0-1":
		return -1
	else:
		return 0
		
def bernuli_dist(n):
	bernuli = np.random.binomial(size = n, n = 1, p = 0.5)
	bernuli[bernuli == 0] = -1
	return(bernuli)
	
def thetaUpdate(theta, R_k, c_k, result_match, bernuli):
	if len(theta) != len(bernuli):
		raise Exception(f"Impossible to sum {theta} and {bernuli}")
	return theta + R_k * c_k * result_match * bernuli

alpha = 0.70
gamma = 0.12
a = 1.1
max_iter = 10000
A = 0.1 * max_iter
C = 0.1

thetaPieceValue = 5 + 2.5 * np.random.randn(6)
thetaPawn = np.random.randn(64)
thetaKnight =  np.random.randn(64)
thetaBishop = np.random.randn(64)
thetaRook =  np.random.randn(64)
thetaQueen =  np.random.randn(64)
thetaKing =  np.random.randn(64)

theta = np.array([thetaPieceValue, thetaPawn, thetaKnight, thetaBishop, thetaRook, thetaQueen, thetaKing])

for k in range(1, 2):
	c_k = C / (k ** gamma)
	a_k = a / ((k + A) ** alpha)
	R_k = a_k / c_k ** 2
	
	
	bernuliPieceValue = bernuli_dist(6)
	bernuliPawn = bernuli_dist(64)
	bernuliKnight = bernuli_dist(64)
	bernuliBishop = bernuli_dist(64)
	bernuliRook  = bernuli_dist(64)
	bernuliQueen = bernuli_dist(64) 
	bernuliKing =  bernuli_dist(64) 
	
	bernuli = np.array([bernuliPieceValue, bernuliPawn, bernuliKnight, bernuliBishop, bernuliRook, bernuliQueen, bernuliKing])
	
	
	evalPlus = evaluate(theta + bernuli * c_k)
	evalMinus = evaluate(theta - bernuli * c_k)

	enginePlus = engine(evalPlus)
	engineMinus = engine(evalMinus)

	result_match = chess_match(enginePlus, engineMinus)
	print(theta)
	theta = theta + R_k * c_k * result_match * bernuli
	print(theta)
	

df = pd.DataFrame(theta, index = [name for name in mapper])

df.to_csv("thetaResult.csv")
