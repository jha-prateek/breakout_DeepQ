import time
import os
import gym
import numpy as np
import cv2
from collections import deque
from keras.models import load_model

model_name = "Breakout_1"
rows = 46
cols = 30

STATE_STACK = deque(deque([np.zeros((rows,cols), dtype=np.int) for i in range(4)],
                          maxlen=4),maxlen=4)

def state_preprocess(state):
    img = cv2.cvtColor(np.array(state), cv2.COLOR_RGB2GRAY)
    img_gray = cv2.resize(img, (cols, rows))
    return img_gray

model = 0

if os.path.exists("{}.h5".format(model_name)):
    model = load_model("{}.h5".format(model_name))
    print("{} Model Loaded".format(model_name))
else:
    print("No {} Found!")

env = gym.make('Breakout-v0')

for e in range(5):
    state = env.reset()

    state = state_preprocess(state)
    STATE_STACK.append(state)
    state = np.stack(STATE_STACK, axis=2)
    state = np.reshape(state, [1, rows, cols, 4])

    for i in range(100):
        env.render()

        action = model.predict(state)

        print("Action: ", np.argmax(action[0]))

        next_state, reward, done, _ = env.step(np.argmax(action[0]))

        next_state = state_preprocess(next_state)
        STATE_STACK.append(next_state)

        next_state = np.stack(STATE_STACK, axis=2)
        next_state = np.reshape(next_state, [1, rows, cols, 4])

        state = next_state

        time.sleep(0.1)
        if done:
            break
