import random
import os
import gym
import numpy as np
import cv2
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K

# # Installing Atari on Windows
# # https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows

# print(K.image_data_format())

EPISODES = 201
FRAME_NUM = 4
ROWS = 80
COLS = 60
STATE_STACK = deque([np.zeros((ROWS,COLS), dtype=np.int) for i in range(FRAME_NUM)],
                    maxlen=FRAME_NUM)

# predictions = []

class Agent:
    def __init__(self, action_size, model_name):
        self.action_size = action_size
        self.model_name = model_name
        self.memory = deque(maxlen=30000)
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001

        self.model = self._build_model(self.model_name)
        self.random_acts = 0

    def _build_model(self, model_name):

        if os.path.exists("{}.h5".format(model_name)):
            model = load_model("{}.h5".format(model_name))
            print("{} Model Loaded".format(model_name))
            return model
        else:
            model = Sequential()
            model.add(Conv2D(32, 8, padding='same',
                             activation='relu',
                             data_format='channels_last',
                             input_shape=(ROWS, COLS, FRAME_NUM)))

            model.add(Conv2D(64, 4, padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3)))

            model.add(Conv2D(64, 3, padding='same', activation='relu'))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.7))

            model.add(Dense(self.action_size))
            model.compile(loss='mse', optimizer=Adam(lr=0.003))

            print(model.summary())

            return model

    def save_model(self):
        self.model.save('{}.h5'.format(self.model_name))
        print("{} saved".format(self.model_name))

    def training_data(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            self.random_acts += 1
            return random.randrange(self.action_size)
        else:
            action = self.model.predict(state)[0]
            # predictions.append(action)
            return np.argmax(action)

    def state_preprocess(self, state):
        img = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.resize(img, (COLS, ROWS))
        return img_gray

    def create_stack(self, state):
        state = self.state_preprocess(state)
        STATE_STACK.append(state)
        state = np.stack(STATE_STACK, axis=2)
        return np.reshape(state, [1, ROWS, COLS, FRAME_NUM])

    def train_network(self, batch_size):
        x_batch, y_batch = [], []

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:

            y_target = self.model.predict(state)
            if done:
                y_target[0][action] = reward
            else:
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])

            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=batch_size,
                       epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    action_size = env.action_space.n
    agent = Agent(action_size, "Breakout_1")

    batch_size = 256
    scores = []

    for e in range(EPISODES):
        observation = env.reset()

        state = agent.create_stack(observation)

        agent.random_acts = 0

        done = False
        total_reward = 0

        while not done:
        # for i in range(500):
            # env.render()

            action = agent.act(state)

            next_frame, reward, done, info = env.step(action)
            cv2.imshow('DQN', next_frame)
            # reward = reward if not done else -10

            total_reward += reward
            scores.append(total_reward)

            if not done:
                next_state = agent.create_stack(next_frame)
                agent.training_data(state, action, reward, next_state, done)
                state = next_state

            else:
                next_state = np.zeros((210, 160, 3), dtype=np.uint8)
                next_state = agent.create_stack(next_state)
                agent.training_data(state, action, reward, next_state, done)

                print("Episode: {}, Score: {}, Epsilon: {:.3}"
                          .format(e+1, total_reward, agent.epsilon))
                print("Random Acts: ", agent.random_acts)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if len(agent.memory) > batch_size:
            agent.train_network(batch_size)

        if ( e %100 == 0):
            print("Average Score of {} Episodes: ".format(e) ,np.mean(scores))
            # agent.save_model()
            # print(np.sum(predictions, axis=0))

cv2.destroyAllWindows()