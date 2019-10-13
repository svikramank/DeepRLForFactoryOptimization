

import numpy as np
import tensorflow as tf
from keras.layers import Dense

from keras import optimizers
from keras.models import Model
from keras.layers import Input

class PolGrad:
    def __init__(self, action_space, state_size, gamma = 0.9,
                 epsilon = 1.0, epsilon_min = 0.00, epsilon_decay = 0.9999):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space
        self.action_size = len(action_space)
        self.state_size = state_size
        
        self.model = self.create_model()
        self.target_model = self.create_model()

    @staticmethod
    def custom_loss(y_pred, y_true, discounted_episode_rewards, allowed_actions):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards)
        return loss

    # Create the neural network model to train the q function
    def create_model(self):
        x = Input(shape=(self.state_size,), name='input')
        y_true = Input(shape=(self.action_size,), name='y_true')
        discounted_episode_rewards = Input(shape=(1,), name='rewards')
        allowed_actions = Input(shape=(self.action_size,), name='allowed_a')
        f = Dense(400, activation = 'sigmoid', kernel_initializer='glorot_uniform')(x)
        f = Dense(250, activation = 'sigmoid', kernel_initializer='glorot_uniform')(f)
        f = Dense(125, activation = 'sigmoid', kernel_initializer='glorot_uniform')(f)
        #logits = K.layers.Activation('linear')(f)
        y_pred = Dense(self.action_size, activation = 'softmax', kernel_initializer='glorot_uniform')(f)
        model = Model(inputs=[x, y_true, discounted_episode_rewards, allowed_actions], outputs = [y_pred])
        model.add_loss(self.custom_loss(y_pred, y_true, discounted_episode_rewards, allowed_actions))
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss = None, optimizer=adam, metrics=['mae'])
        return model


    # Action function to choose the best action given the q-function if not exploring based on epsilon
    def choose_action(self, state, allowed_actions):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        
        n = 0
        allowed_act_prob = np.zeros((1, self.action_size))
        for i in self.action_space:
            for j in allowed_actions:
                if i == j:
                    allowed_act_prob[0][n] = 1
            n+=1
            
        r = np.random.random()
        if r > self.epsilon:
            print(" ************* CHOOSING A PREDICTED ACTION *************")
            actions = np.ones((1, self.action_size))
            rewards = np.ones((1, 1))
            state = np.array(state).reshape(1, self.state_size)
            pred = self.model.predict([state, actions, rewards, allowed_act_prob])
            allowed_act_prob_aux = allowed_act_prob * pred
            if np.sum(allowed_act_prob_aux) != 0:
                allowed_act_prob = allowed_act_prob_aux
        else:
            print("******* CHOOSING A RANDOM ACTION *******")
        all_sum = np.sum(allowed_act_prob)
        multiply = 1/all_sum
        allowed_act_prob *= multiply
        # select action w.r.t the actions prob
        action = np.random.choice(range(allowed_act_prob.shape[1]), p=allowed_act_prob.ravel())
        return action

    # training our PG network
    def train_policy_gradient(self, states, actions, discounted_episode_rewards, allowed_actions):
        n = 0
        allowed_act_prob = np.zeros((states.shape[0], self.action_size))
        for i in self.action_space:
            for idx, val in enumerate(allowed_actions):
                for j in val:
                    if i == j:
                        allowed_act_prob[idx][n] = 1.0
            n+=1
        self.model.fit([states, actions, discounted_episode_rewards, allowed_act_prob])

    # Save our model
    def save_model(self, fn):
        self.model.save(fn)



        