from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random


########################################################################################################################################
#################################################################### CREATING Deep Q-learning Class ####################################
########################################################################################################################################

class DQN:
    def __init__(self, state_space_dim, action_space, gamma=0.9, epsilon_decay=0.8, tau=0.125, learning_rate=0.005):
        self.state_space_dim = state_space_dim
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=2000)
        self.model = self.create_model()
        self.target_model = self.create_model()

    # Create the neural network model to train the q function
    def create_model(self):
        model = Sequential()
        model.add(Dense(400, input_dim= self.state_space_dim, activation='relu'))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(125, activation='relu'))
        model.add(Dense(len(self.action_space)))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    # Action function to choose the best action given the q-function if not exploring based on epsilon
    def choose_action(self, state, allowed_actions):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        r = np.random.random()
        if r < self.epsilon:
            print("******* CHOOSING A RANDOM ACTION *******")
            return random.choice(allowed_actions)
        # print(state)
        # print(len(state))
        state = np.array(state).reshape(1, self.state_space_dim)
        pred = self.model.predict(state)
        pred = sum(pred.tolist(), [])
        temp = []
        for item in allowed_actions:
            temp.append(pred[self.action_space.index(item)])
        print(" ********************* CHOOSING A PREDICTED ACTION **********************")
        return allowed_actions[np.argmax(temp)]

    # Create replay buffer memory to sample randomly
    def remember(self, state, action, reward, next_state, next_allowed_actions):
        self.memory.append([state, action, reward, next_state, next_allowed_actions])

    # Build the replay buffer
    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, new_allowed_actions = sample
            state = np.array(state).reshape(1, self.state_space_dim)
            target = self.target_model.predict(state)
            action_id = self.action_space.index(action)
            # if done:
            #     target[0][action_id] = reward
            # else:
                # take max only from next_allowed_actions
            new_state = np.array(new_state).reshape(1,self.state_space_dim)
            next_pred = self.target_model.predict(new_state)[0]
            next_pred = next_pred.tolist()
            t = []
            print("new_allowed_actions:", new_allowed_actions)
            for it in new_allowed_actions:
                t.append(next_pred[self.action_space.index(it)])
            Q_future = max(t)
            target[0][action_id] = reward + self.gamma * Q_future
            self.model.fit(state, target, epochs=1, verbose=1)


    # Update our target network
    def train_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    # Save our model
    def save_model(self, fn):
        self.model.save(fn)