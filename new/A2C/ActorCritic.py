from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random


########################################################################################################################################
#################################################################### CREATING A2C Class ################################################
########################################################################################################################################

# Advantage Actor-Critic agent 
class A2CAgent:
    def __init__(self, state_size, action_space, epsilon_decay=0.8):
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space)
        self.value_size = 1
        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_decay = epsilon_decay

        # Hyperparameters for Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # Create model for policy network 
        self.actor = self.build_actor()
        self.critic = self.build_critic()


    # Approximate policy and value using Neural Network 
    # actor: state is input and probability of each action is output of model 
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(400, input_dim= self.state_size, activation= 'relu', kernel_initializer='he_uniform'))
        actor.add(Dense(250, activation= 'relu', kernel_initializer='he_uniform'))
        actor.add(Dense(125, activation= 'relu', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation= 'softmax', kernel_initializer= 'he_uniform'))
        actor.summary()
        actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is the output of model 
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(400, input_dim= self.state_size, activation= 'relu', kernel_initializer='he_uniform'))
        critic.add(Dense(250, activation= 'relu', kernel_initializer='he_uniform'))
        critic.add(Dense(125, activation= 'relu', kernel_initializer='he_uniform'))
        critic.add(Dense(50, activation= 'relu', kernel_initializer= 'he_uniform'))
        critic.add(Dense(self.value_size, activation= 'linear', kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # Using the output of the policy network, pick action stochastically 
    def choose_action(self, state, allowed_actions):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        r = np.random.random()

        if r < self.epsilon:
            print("******* CHOOSING A RANDOM ACTION *******")
            return random.choice(allowed_actions)

        state = np.array(state).reshape(1, self.state_size)
        pred = self.actor.predict(state)
        pred = sum(pred.tolist(), [])
        temp = []
        for item in allowed_actions:
            temp.append(pred[self.action_space.index(item)])
        print(" ********************* CHOOSING A PREDICTED ACTION **********************")
        return allowed_actions[np.argmax(temp)]


    # Update the policy network every episode 
    def train_model(self, state, action, reward, next_state):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))
        state = np.array(state).reshape(1, self.state_size)
        next_state = np.array(state).reshape(1, self.state_size)
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        action_id = self.action_space.index(action)

        advantages[0][action_id] = reward + self.discount_factor * (next_value) - value
        target[0][0] = reward + self.discount_factor*next_value

        self.actor.fit(state, advantages, epochs=1)
        self.critic.fit(state, target, epochs=1)


    # Save the actor and critic models
    def save_model(self, fn1, fn2):
        self.actor.save(fn1)
        self.critic.save(fn2)

































