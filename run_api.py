import gym
import random
import numpy as np
import tensorflow as tf
from keras import layers
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model

from collections import deque
from keras.optimizers import RMSprop
from keras import backend as K
from datetime import datetime
import os.path
import time
from keras.models import load_model
from keras.models import clone_model
import json


class RL_Atari_Breakout:

    def __init__(self, filepath=None, ATARI_SHAPE=None, ACTION_SIZE=None):
        if filepath is None:
            filepath = 'config.json'
        self.FLAGS = self._fill_up_flags(filepath)

        # 210*160*3(color) --> 84*84(mono)
        # float --> integer (to reduce the size of replay memory)

        self.ATARI_SHAPE = (84, 84, 4) if ATARI_SHAPE is None else ATARI_SHAPE  # input image size to model
        self.ACTION_SIZE = 3 if ACTION_SIZE is None else ACTION_SIZE

    def _fill_up_flags(self, filepath):
        with open(filepath, 'r') as f:
            temp_dict = json.load(f)
        for key, data in temp_dict.items():
            if data['type'] == 'string':
                tf.app.flags.DEFINE_string(*data['vars'])
            if data['type'] == 'integer':
                tf.app.flags.DEFINE_integer(*data['vars'])
            if data['type'] == 'float':
                tf.app.flags.DEFINE_float(*data['vars'])
            if data['type'] == 'boolean':
                tf.app.flags.DEFINE_boolean(*data['vars'])
            print("{} :  {},{},{} ... done".format(data['type'], *data['vars']))
        return tf.app.flags.FLAGS

    def _atari_model(self):
        def huber_loss(y, q_value):
            error = K.abs(y - q_value)
            quadratic_part = K.clip(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
            return loss

        # With the functional API we need to define the inputs.
        frames_input = layers.Input(self.ATARI_SHAPE, name='frames')
        actions_input = layers.Input((self.ACTION_SIZE,), name='action_mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

        # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
        conv_1 = layers.convolutional.Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu'
        )(normalized)
        # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
        conv_2 = layers.convolutional.Conv2D(
            32, (4, 4), strides=(2, 2), activation='relu'
        )(conv_1)
        # Flattening the second convolutional layer.
        conv_flattened = layers.core.Flatten()(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = layers.Dense(256, activation='relu')(conv_flattened)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = layers.Dense(self.ACTION_SIZE)(hidden)
        # Finally, we multiply the output by the mask!
        filtered_output = layers.Multiply(name='QValue')([output, actions_input])

        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        model.summary()
        optimizer = RMSprop(lr=self.FLAGS.learning_rate, rho=0.95, epsilon=0.01)
        # model.compile(optimizer, loss='mse')
        # to changed model weights more slowly, uses MSE for low values and MAE(Mean Absolute Error) for large values
        model.compile(optimizer, loss=huber_loss)
        return model

    def _train_initialization(self, replacement_model=None):
        # deque: Once a bounded length deque is full, when new items are added,
        # a corresponding number of items are discarded from the opposite end

        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        training_variables = {
            'memory': deque(maxlen=self.FLAGS.replay_memory),
            'episode_number': 0,
            'epsilon': self.FLAGS.init_epsilon,
            'epsilon_decay': (self.FLAGS.init_epsilon - self.FLAGS.final_epsilon) / self.FLAGS.epsilon_step_num,
            'global_step': 0,
            'now': now,
            'log_dir': '{}/run-{}-log'.format(self.FLAGS.train_dir, now)
        }

        if self.FLAGS.resume:
            model = load_model(self.FLAGS.restore_file_path)
            # Assume when we restore the model, the epsilon has already decreased to the final value
            training_variables['epsilon'] = self.FLAGS.final_epsilon
        else:
            model = self._atari_model() if replacement_model is None else replacement_model()

        return model, training_variables

    def _frame_pre_processing(self, observed_frame):
        return np.uint8(resize(rgb2gray(observed_frame), self.ATARI_SHAPE[:2], mode='constant') * 255)

    def _episode_init(self, episode_variables):
        episode_variables['done'] = False
        episode_variables['dead'] = False
        # 1 episode = 5 lives

        episode_variables['step'] = 0
        episode_variables['score'] = 0
        episode_variables['start_life'] = 5
        episode_variables['loss'] = 0.0
        self.env.reset()

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, self.FLAGS.no_op_steps)):
            observed_frame, _, _, _ = self.env.step(1)
        episode_variables['observed_frame'] = observed_frame

        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        state = self._frame_pre_processing(observed_frame)
        episode_variables['state'] = state
        history = np.stack((state, state, state, state), axis=2)
        episode_variables['history'] = np.reshape([history], (1, *self.ATARI_SHAPE))

    def _episode_get_action(self, training_variables, episode_variables, model_target):
        if np.random.rand() <= training_variables['epsilon'] \
                or training_variables['global_step'] <= self.FLAGS.observe_step_num:
            return random.randrange(self.ACTION_SIZE)
        else:
            q_value = model_target.predict([episode_variables['history'],
                                            np.ones(self.ACTION_SIZE).reshape(1, self.ACTION_SIZE)])
            return np.argmax(q_value[0])

    def _episode_evaluate_action(self, action):
        real_action = action + 1
        state = self.env.unwrapped.clone_full_state()
        reaction = self._evaluate(action)
        self.env.unwrapped.restore_full_state(state)
        return reaction

    def _update_epsilon(self, training_variables):
        if training_variables['epsilon'] > self.FLAGS.final_epsilon \
                and training_variables['global_step'] > self.FLAGS.observe_step_num:
            training_variables['epsilon'] -= training_variables['epsilon_decay']

    def _update_episode_variables(self, episode_variables, reaction_variables):
        # pre-process the observation --> history
        next_state = self._frame_pre_processing(reaction_variables['observed_frame'])
        next_state = np.reshape([next_state], (1, *self.ATARI_SHAPE[:2], 1))
        next_history = np.append(next_state, episode_variables['history'][:, :, :, :3], axis=3)

        # if the agent missed ball, agent is dead --> episode is not over
        if episode_variables['start_life'] > reaction_variables['info']['ale.lives']:
            episode_variables['dead'] = True
            episode_variables['start_life'] = reaction_variables['info']['ale.lives']

        return next_state, next_history

    def _evaluate(self, action):
        real_action = action + 1
        observed_frame, reward, done, info = self.env.step(real_action)
        return {'observed_frame': observed_frame, 'reward': reward, 'done': done, 'info': info, 'action': action}

    def _store_memory(self, training_variables, episode_variables, reaction_variables, next_history):
        training_variables['memory'].append((episode_variables['history'],
                                             reaction_variables['action'],
                                             reaction_variables['reward'],
                                             next_history,
                                             episode_variables['dead']))

    def _train_memory_batch(self, memory, model, log_dir):
        def get_one_hot(targets, nb_classes):
            return np.eye(nb_classes)[np.array(targets).reshape(-1)]

            mini_batch = random.sample(memory, self.FLAGS.batch_size)
            history = np.zeros((self.FLAGS.batch_size, self.ATARI_SHAPE[0],
                                self.ATARI_SHAPE[1], self.ATARI_SHAPE[2]))
            next_history = np.zeros((self.FLAGS.batch_size, self.ATARI_SHAPE[0],
                                     self.ATARI_SHAPE[1], self.ATARI_SHAPE[2]))
            target = np.zeros((self.FLAGS.batch_size,))
            action, reward, dead = [], [], []

            for idx, val in enumerate(mini_batch):
                history[idx] = val[0]
                next_history[idx] = val[3]
                action.append(val[1])
                reward.append(val[2])
                dead.append(val[4])

            actions_mask = np.ones((self.FLAGS.batch_size, self.ACTION_SIZE))
            next_Q_values = model.predict([next_history, actions_mask])

            # like Q Learning, get maximum Q value at s'
            # But from target model
            for i in range(self.FLAGS.batch_size):
                if dead[i]:
                    target[i] = -1
                    # target[i] = reward[i]
                else:
                    target[i] = reward[i] + self.FLAGS.gamma * np.amax(next_Q_values[i])

            action_one_hot = get_one_hot(action, self.ACTION_SIZE)
            target_one_hot = action_one_hot * target[:, None]

            h = model.fit(
                [history, action_one_hot], target_one_hot, epochs=1,
                batch_size=self.FLAGS.batch_size, verbose=0)

            return h

    def _batch_training(self, training_variables, episode_variables, reaction_variables, model_target, model):
        # check if the memory is ready for training
        if training_variables['global_step'] > self.FLAGS.observe_step_num:
            h0 = self._train_memory_batch(training_variables['memory'], model, training_variables['log_dir'])
            if h0 is None:
                temp = 0.0
            else:
                temp=h0.history['loss'][0]
            episode_variables['loss'] += temp
            # if loss > 100.0:
            #    print(loss)
            if training_variables['global_step'] % self.FLAGS.refresh_target_model_num == 0:  # update the target model
                model_target.set_weights(model.get_weights())

        episode_variables['score'] += reaction_variables['reward']

    def _check_exit_condition(self, training_variables, episode_variables, next_history):
        # If agent is dead, set the flag back to false, but keep the history unchanged,
        # to avoid to see the ball up in the sky

        if episode_variables['dead']:
            episode_variables['dead'] = False
        else:
            episode_variables['history'] = next_history
        # print("step: ", global_step)
        training_variables['global_step'] += 1
        episode_variables['step'] += 1

    def _print_summary_of_episode(self, training_variables, episode_variables):
        if training_variables['global_step'] <= self.FLAGS.observe_step_num:
            state = "observe"
        elif self.FLAGS.observe_step_num < training_variables['global_step'] \
                <= self.FLAGS.observe_step_num + self.FLAGS.epsilon_step_num:
            state = "explore"
        else:
            state = "train"
        print('state: {}, episode: {}, score: {}, global_step: {}, avg loss: {}, step: {}, memory length: {}'
              .format(state,
                      training_variables['episode_number'],
                      episode_variables['score'],
                      training_variables['global_step'],
                      episode_variables['loss'] / float(episode_variables['step']),
                      episode_variables['step'],
                      len(training_variables['memory'])))

    def _save_episode(self, training_variables, episode_variables, model, file_writer):
        if training_variables['episode_number'] % 100 == 0 \
                or (training_variables['episode_number'] + 1) == self.FLAGS.num_episode:
            now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            file_name = "breakout_model_{}.h5".format(now)
            model_path = os.path.join(self.FLAGS.train_dir, file_name)
            model.save(model_path)

        # Add user custom data to TensorBoard
        loss_summary = tf.Summary(
            value=[tf.Summary.Value(tag="loss",
                                    simple_value=episode_variables['loss'] /
                                                 float(episode_variables['step']))])
        file_writer.add_summary(loss_summary, global_step=training_variables['episode_number'])

        score_summary = tf.Summary(
            value=[tf.Summary.Value(tag="score", simple_value=episode_variables['score'])])
        file_writer.add_summary(score_summary, global_step=training_variables['episode_number'])

    def train(self, replacement_model=None):
        self.env = gym.make('BreakoutDeterministic-v4')
        model, training_variables = self._train_initialization(replacement_model)

        file_writer = tf.summary.FileWriter(training_variables['log_dir'], tf.get_default_graph())

        model_target = clone_model(model)
        model_target.set_weights(model.get_weights())

        while training_variables['episode_number'] < self.FLAGS.num_episode:

            # initialization
            episode_variables = {}
            self._episode_init(episode_variables)

            # game on!
            while not episode_variables['done']:
                if self.FLAGS.render:
                    self.env.render()
                    time.sleep(0.01)

                # STEP 1: GET ACTION FOR CURRENT STATE
                # get action for the current history and go one step in environment
                action = self._episode_get_action(training_variables, episode_variables, model_target)

                # STEP 2: UPDATE EPSILON
                # scale down epsilon, the epsilon only begin to decrease after observed steps
                self._update_epsilon(training_variables)

                # STEP 3: EVALUATE STEP
                reaction_variables = self._evaluate(action)
                episode_variables['done'] = reaction_variables['done']

                # STEP 4: UPDATE VARIABLES & STORE MEMORY
                next_state, next_history = self._update_episode_variables(episode_variables, reaction_variables)
                self._store_memory(training_variables, episode_variables, reaction_variables, next_history)

                # STEP 5: BATCH TRAIN IF READY
                self._batch_training(training_variables, episode_variables, reaction_variables, model_target, model)

                # STEP 6: CHECK EXIT CONDITION
                self._check_exit_condition(training_variables, episode_variables, next_history)

                if episode_variables['done']:
                    self._print_summary_of_episode(training_variables, episode_variables)
                    self._save_episode(training_variables, episode_variables, model, file_writer)
                    training_variables['episode_number'] += 1

        file_writer.close()


if __name__ == '__main__':
    my_rl_model = RL_Atari_Breakout()
    my_rl_model.train()
