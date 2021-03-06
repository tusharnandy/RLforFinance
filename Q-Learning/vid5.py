from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2d, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
from keras.callbacks import TensorBoard
import time
import numpy as np

replay_memory_size = 50_000
model_name = "256x2"

class DQNAgent:

    def __init__(self):
        # main model gets trained every step
        self.model = self.create_model()

        # Target model this what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weigths())
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.tensorboard = ModidfiedTensorBoard(log_dir=f"logs/{model_name}-{int(time.time())}")
        self.target_update_counter = 0


    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape=env.observation_space_values))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3,3), input_shape=env.observation_space_values))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(64))

        model.add(Dense(env.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, terminal_state, step):
        self.model_predict(np.array(state).shape(-1, *state.shape)/255)[0]
