import numpy as np
from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Conv2D, Dense, Reshape, Dot, Activation, Multiply, Flatten, Concatenate
import tensorflow as tf
from tensorflow.keras.models import Model
import copy
import random
import tensorflow.keras.backend as K
from tensorflow.python.framework.ops import disable_eager_execution
import datetime



def policy_gradient_loss(Returns):
    def modified_crossentropy(action, action_probs):
        cost = (K.categorical_crossentropy(action, action_probs, from_logits=False, axis=1) * Returns)
        return K.mean(cost)

    return modified_crossentropy


class Agent(object):

    def init_dqn(self):
        optimizer = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
        R = Input(shape=(1,), name='Rewards')
        legal_moves = Input(shape=(4096,), name='legal_move_mask')
        layer_state = Input(shape=(8, 8, 8), name='state')
        conv_xs = Conv2D(4, (1, 1), activation='relu')(layer_state)
        conv_s = Conv2D(8, (2, 2), strides=(1, 1), activation='relu')(layer_state)
        conv_m = Conv2D(12, (3, 3), strides=(2, 2), activation='relu')(layer_state)
        conv_l = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(layer_state)
        conv_xl = Conv2D(20, (8, 8), activation='relu')(layer_state)
        conv_rank = Conv2D(3, (1, 8), activation='relu')(layer_state)
        conv_file = Conv2D(3, (8, 1), activation='relu')(layer_state)

        f_xs = Flatten()(conv_xs)
        f_s = Flatten()(conv_s)
        f_m = Flatten()(conv_m)
        f_l = Flatten()(conv_l)
        f_xl = Flatten()(conv_xl)
        f_r = Flatten()(conv_rank)
        f_f = Flatten()(conv_file)

        dense1 = Concatenate(name='dense_bass')([f_xs, f_s, f_m, f_l, f_xl, f_r, f_f])
        dense2 = Dense(4096, activation='relu')(dense1)
		
        softmax_layer = Activation('softmax')(dense2)
        legal_softmax_layer = Multiply()([legal_moves, softmax_layer])  # Select legal moves
        self.model = Model(inputs=[layer_state, R, legal_moves], outputs=[legal_softmax_layer])
        self.model.compile(optimizer=optimizer, loss=policy_gradient_loss(R))
        print(self.model.summary())
        
    def __init__(self, lr=0.01, name = "logs/fit/"):
       disable_eager_execution()
       self.lr = lr
       self.long_term_mean = []
       self.weight_memory = []
       self.init_dqn()
       self.step = 0
       self.new_steps = 1000 # количество шагов, после которых веса online network  копируются в target network
       self.g = 0.99
    def predict(self, state):
    	with tf.device('/GPU:0'):
        	return self.model.predict(tf.expand_dims(state, axis = 0))
        
    def predict_array(self, state):
        print(len(state))
        return np.array([self.predict(x) for x in state])
            
            
    def policy_gradient_update(self, states, actions, rewards, action_spaces):
        """
        Update parameters with Monte Carlo Policy Gradient algorithm
        Args:
            states: (list of tuples) state sequence in episode
            actions: action sequence in episode
            rewards: rewards sequence in episode
        Returns:
        """
        n_steps = len(states)
        Returns = []
        targets = np.zeros((n_steps, 64, 64))
        for t in range(n_steps):
            action = actions[t]
            targets[t, action.from_square, action.to_square] = 1
            R = rewards[t, action.from_square * 64 + action.to_square]
            Returns.append(R)

        train_returns = np.stack(Returns, axis=0)
        targets = targets.reshape((n_steps, 4096))
        self.weight_memory.append(self.model.get_weights())
        with tf.device('/GPU:0'):
            self.model.fit(x=[np.stack(states, axis=0),
                          train_returns,
                          np.concatenate(action_spaces, axis=0)], y=[np.stack(targets, axis=0)])
        print("ОБУЧЕНО АГЕНТ")
        
    def save(self, path):
    	self.model.save_weights('saved/' + path + ".h5")
    def load(self, path):
    	self.model.load_weights('saved/' + path + ".h5")
    	