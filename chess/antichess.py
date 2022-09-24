import copy
import chess
import chess.pgn
import chess.variant
import numpy as np
from Agent import Agent
from Enviroment import Env
import random
from collections import deque
import collections
from Critic import Critic
from itertools import islice
import io

def sample_memory(turncount, memory, sampling_probs):
    """
    Get a sample from memory for experience replay
    Args:
        turncount: int
        turncount limits the size of the minibatch
    Returns: tuple
        a mini-batch of experiences (list)
        indices of chosen experiences
    """
    minibatch = []
    indices = []

    memory = np.array(list(islice(memory, None, len(memory)-turncount)))

    probs = np.array(list(islice(sampling_probs, None, len(memory)-turncount)))
    probs = probs.astype('float64')  /  probs.sum().astype('float64')

    number_of_choices = min(1028, len(memory))
    resample_counts = np.random.multinomial(number_of_choices,
                                        probs)
                                        
    resample_index = 0
    for resample_count in resample_counts:
    	for _ in range(resample_count):
    		minibatch.append(memory[resample_index])
    		indices.append(resample_index)
    	resample_index += 1
    return minibatch, indices

def update_actorcritic(memory, turncount, sampling_probs, critic, actor, iter):
     """Actor critic"""
     if turncount < len(memory):
        minibatch, indices = sample_memory(turncount, memory, sampling_probs)
        if(len(minibatch) == 0):
            print("BAD BATCH")
            return sampling_probs
        # Update critic and find td errors for prioritized experience replay
        td_errors = critic.fit(minibatch, iter)

        # Get a Q value from the critic
        states = [x[0] for x in minibatch]
        actions = [x[1] for x in minibatch]
        Q_est = critic.get_action_values(np.stack(states, axis=0))
        action_spaces = [x[5] for x in minibatch]

        actor.policy_gradient_update(states, actions, Q_est, action_spaces)

    	# Update sampling probs
        for n, i in enumerate(indices):
            sampling_probs[i] = np.abs(td_errors[n])
        return deque(sampling_probs)
        

def make_move(env, actor):
	state = env.layer_board
	action_space = env.project_legal_moves()
	action_probs = actor.model.predict([np.expand_dims(state, axis=0), np.zeros((1, 1)), action_space.reshape(1, 4096)])
	action_probs = action_probs / action_probs.sum()
	for i in range(len(action_probs)):
		print(action_probs[i])
	move = np.random.choice(range(4096), p=np.squeeze(action_probs))
	move_from = move // 64
	move_to = move % 64
	moves = [x for x in env.board.generate_legal_moves() if x.from_square == move_from and x.to_square == move_to]
	assert len(moves) > 0  # should not be possible
	if len(moves) > 1:
		move = np.random.choice(moves)  # If there are multiple max-moves, pick a random one.
	elif len(moves) == 1:
		move = moves[0]
	do, reward = env.step(move)
	return state, move, do, reward, action_space.reshape(1, 4096)
	
def make_Emove(env, move):
	state = env.layer_board
	action_space = env.project_legal_moves()
	do, reward = env.step(move)
	return state, move, do, reward, action_space.reshape(1, 4096)
	

online_white = Agent("logs/fit/")
online_white.load("white")
critic_white = Critic()
critic_white.load("white")
online_black = Agent("logs2/fit/")
online_black.load("black")
critic_black = Critic()
critic_black.load("black")


number_of_steps = 10000

w_memory = deque(maxlen = 3000)
b_memory = deque(maxlen = 3000)
sampling_probs_w = deque(maxlen = 3000)
sampling_probs_b = deque(maxlen = 3000)



iter = 0
step = 0


w_state, w_move, do, w_reward, w_action_space = 0, 0, True, 0, 0
b_state, b_move, do, b_reward, b_action_space = 0, 0, True, 0, 0
turncount = 0


# pgn = open("lichess_db_antichess_rated_2022-07.pgn")
# data = io.StringIO(pgn.read())
# pgn.close()
# for i in range(5000, 10000):
#     turncount = 0
#     game = chess.pgn.read_game(data)
#     r_w = game.headers["WhiteElo"]
#     r_b = game.headers["BlackElo"]
#     turncount = 0
#     env = Env()
#     do = False
#     if int(r_b) > 2000:
#         for move in game.mainline_moves():
#             if env.board.turn == chess.WHITE:
#                 if turncount >= 1:
#                     if not do:
#                         b_memory.append((b_state, b_move, b_reward, env.layer_board, do, b_action_space))
#                         sampling_probs_b.append(1)
#                 w_state, w_move, do, w_reward, w_action_space = make_Emove(env, move)
#             else:
#                 b_state, b_move, do, b_reward, b_action_space = make_Emove(env, move)
#                 if not do:
#                     w_memory.append((w_state, w_move, w_reward, env.layer_board, do, w_action_space))
#                     sampling_probs_w.append(1)
#             iter += 1
#             turncount += 1
#     if int(r_b) > 2000 and i > 5300:
#         sampling_probs_w = update_actorcritic(w_memory, turncount, sampling_probs_w, critic_white, online_white, iter)
#         print("====================> ОБУЧЕНО БЕЛЫЕ")
#         sampling_probs_b = update_actorcritic(b_memory, turncount, sampling_probs_b, critic_black, online_black, iter)
#         print("====================> ОБУЧЕНО ЧЕРНЫЕ")
		
# pgn = open("lichess_db_antichess_rated_2022-08.pgn")
# data = io.StringIO(pgn.read())
# pgn.close()
# for i in range(5000, 10000):
#     turncount = 0
#     game = chess.pgn.read_game(data)
#     r_w = game.headers["WhiteElo"]
#     r_b = game.headers["BlackElo"]
#     turncount = 0
#     env = Env()
#     do = False
#     if int(r_b) > 2000:
#         for move in game.mainline_moves():
#             if env.board.turn == chess.WHITE:
#                 if turncount >= 1:
#                     if not do:
#                         b_memory.append((b_state, b_move, b_reward, env.layer_board, do, b_action_space))
#                         sampling_probs_b.append(1)
#                 w_state, w_move, do, w_reward, w_action_space = make_Emove(env, move)
#             else:
#                 b_state, b_move, do, b_reward, b_action_space = make_Emove(env, move)
#                 if not do:
#                     w_memory.append((w_state, w_move, w_reward, env.layer_board, do, w_action_space))
#                     sampling_probs_w.append(1)
#             iter += 1
#             turncount += 1
#     if int(r_b) > 2000 and i > 5300:
#         sampling_probs_w = update_actorcritic(w_memory, turncount, sampling_probs_w, critic_white, online_white, iter)
#         print("====================> ОБУЧЕНО БЕЛЫЕ")
#         sampling_probs_b = update_actorcritic(b_memory, turncount, sampling_probs_b, critic_black, online_black, iter)
#         print("====================> ОБУЧЕНО ЧЕРНЫЕ")
# 		
		

do = True
for step_first in range(number_of_steps):
	if(do):
		env = Env()
		do = False
		turncount = 0
	iter += 1
	w_state, w_move, do, w_reward,  w_action_space = make_move( env, online_white)
	if not do:
		if turncount >= 1:
			b_memory.append((b_state, b_move, b_reward, env.layer_board, do, b_action_space))
			sampling_probs_b.append(1)
		print(env.board)
		b_state, b_move, do, b_reward, b_action_space = make_move(env, online_black)
		if not do:
			w_memory.append((w_state, w_move, w_reward, env.layer_board, do, w_action_space))
			sampling_probs_w.append(1)
		turncount += 1
	if do and step_first > 100:
		sampling_probs_w = update_actorcritic(w_memory, turncount, sampling_probs_w, critic_white, online_white, iter)
		print("====================> ОБУЧЕНО БЕЛЫЕ")
		sampling_probs_b = update_actorcritic(b_memory, turncount, sampling_probs_b, critic_black, online_black, iter)
		print("====================> ОБУЧЕНО ЧЕРНЫЕ")

online_white.save('white')
critic_white.save('white')
online_black.save('black')
critic_black.save('black')
