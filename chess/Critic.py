import numpy as np
from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Conv2D, Dense, Reshape, Dot, Activation, Multiply, Flatten, Concatenate
import tensorflow as tf
from keras.models import Model
import copy
import random
import keras.backend as K



class Critic(object):

    def init_dqn(self):
        optimizer = SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False)
        input_layer = Input(shape=(8, 8, 8), name='board_layer')
        inter_layer_1 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
        inter_layer_2 = Conv2D(1, (1, 1), data_format="channels_first")(input_layer)  # 1,8,8
        flat_1 = Reshape(target_shape=(1, 64))(inter_layer_1)
        flat_2 = Reshape(target_shape=(1, 64))(inter_layer_2)
        output_dot_layer = Dot(axes=1)([flat_1, flat_2])
        output_layer = Reshape(target_shape=(4096,))(output_dot_layer)
        self.model = Model(inputs=[input_layer], outputs=[output_layer])
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        print(self.model.summary())
        
    def __init__(self, lr=0.01):
       self.lr = lr
       self.init_dqn()
       self.step = 0
       self.target = copy.deepcopy(self.model)
       self.new_steps = 1000 # количество шагов, после которых веса online network  копируются в target network
       self.g = 0.99
    def predict(self, state):
    	with tf.device('/GPU:0'):
        	return self.model.predict(tf.expand_dims(state, axis = 0))
        
    def predict_array(self, state):
        print(len(state))
        return np.array([self.predict(x) for x in state])

    def fit(self, memory, step_first):
        self.step += 1
        td_errors = []
        state = np.array([x[0] for x in memory])
        action = np.array([x[1] for x in memory])
        rewards = np.array([x[2] for x in memory])
        next_state = np.array([x[3] for x in memory])
        replay_done = np.array([x[4] for x in memory], dtype=int)
        with tf.device('/GPU:0'):
        	target_for_action = rewards + (1-replay_done) * self.g * np.amax(self.target.predict(np.stack(next_state, axis=0)), axis=1)
        	y_target = self.model.predict(np.stack(state, axis=0))
        y_target = np.reshape(y_target, (len(memory), 64, 64))
        for i in range(len(action)):
            td_errors.append(y_target[i, action[i].from_square , action[i].to_square] - target_for_action[i])
            y_target[i, action[i].from_square , action[i].to_square] = target_for_action[i]
        y_target = np.reshape(y_target, (len(memory), 4096))
        with tf.device('/GPU:0'):
        	self.model.fit(np.stack(state, axis=0), y_target, epochs=self.step, verbose=1, initial_epoch=self.step-1)
        if step_first % self.new_steps == 0:
            self.target =  copy.deepcopy(self.model)
        print("ОБУЧЕНО КРИТИК")
        return td_errors
            
    def get_action_values(self, state):
        """
        Get action values of a state
        Args:
            state: np.ndarray with shape (8,8,8)
                layer_board representation
        Returns:
            action values
        """
        return self.model.predict(state) + np.random.randn() * 1e-9
        
    def save(self, path):
    	self.model.save_weights('saved/' + path + "Critic" + ".h5")
    def load(self, path):
    	self.model.load_weights('saved/' + path + "Critic" +  ".h5")
    	
