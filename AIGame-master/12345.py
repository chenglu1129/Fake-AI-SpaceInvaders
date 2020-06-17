from tensorflow.keras import layers, models, regularizers, optimizers, Sequential

import tensorflow as tf
import numpy as np
import gym
import os

def regularized_padded_conv2d(*args, **kwargs):

    return layers.Conv2D(*args, **kwargs, padding='same',
                         kernel_regularizer=regularizers.l2(5e-5),
                         bias_regularizer=regularizers.l2(5e-5),
                         kernel_initializer='glorot_normal')

class _CNN_Model(models.Model):
    def __init__(self, action_space_n):
        super(_CNN_Model, self).__init__()
        self.Conv_1 = regularized_padded_conv2d(filters=36, kernel_size=(3, 3), strides=1)
        self.pool_1 = layers.MaxPooling2D(pool_size=(3, 3), strides=3)

        self.Conv_2 = regularized_padded_conv2d(filters=36, kernel_size=(3, 3), strides=1)
        self.pool_2 = layers.MaxPooling2D(pool_size=(3, 3), strides=3)

        self.Conv_3 = regularized_padded_conv2d(filters=36, kernel_size=(3, 3), strides=1)
        self.pool_3 = layers.MaxPooling2D(pool_size=(3, 3), strides=3)

        self.flatten = layers.Flatten()
        self.fc_1 = layers.Dense(units=1024, activation='relu')
        self.fc_2 = layers.Dense(units=256, activation='relu')

        self.out = layers.Dense(units=action_space_n)

    def call(self, inputs, training=None, mask=None):

        x = self.Conv_1(inputs)
        x = self.pool_1(x)

        x = self.Conv_2(x)
        x = self.pool_2(x)

        x = self.Conv_3(x)
        x = self.pool_3(x)

        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.out(x)
        x = tf.nn.softmax(x)
        return x

class Agent(object):
    def __init__(self, env, state_space):
        self._env = env
        self.state_space = state_space

        self.model = _CNN_Model(action_space_n=env.action_space.n)

        self.model.compile(loss=self.loss, optimizer=optimizers.Adam())

        self.gamma = 0.9

    def loss(self, state):
        probs = self.model.predict(np.array([state]))
        _acts = np.array([[1 if np.random.choice(self._env.action_space.n, p=prob) == i
                               else 0
                           for i in range(self._env.action_space.n)]
                                for prob in probs])
        return - tf.reduce_sum(np.log(probs * _acts))

    def choose_action(self, state):
        prob = self.model.predict(np.array([state], dtype=np.float)/255.0)[0]
        return np.random.choice(self._env.action_space.n, p=prob)

    def discount_rewards(self, rewards):
        prior = 0
        out = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            prior = prior * self.gamma + rewards[i]
            out[i] = prior
        return out / np.std(out - np.mean(out))

    def train(self, records):
        s_batch = np.array([record[0] for record in records], dtype=np.float)/255.0
        r_batch = self.discount_rewards([record[1] for record in records])

        self.model.fit(s_batch, sample_weight=r_batch, verbose=1)

if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    env = gym.make('SpaceInvaders-v0')
    agent = Agent(env, (None, 210, 160, 3))
    episodes = 1000
    score_list = []
    for e in range(episodes):
        s = env.reset()
        score = 0
        replay_records = []
        while True:
            env.render()
            a = agent.choose_action(s)

            next_s, r, done, _ = env.step(a)
            r = -100 if done else r
            replay_records.append((s, r))

            score += r
            s = next_s
            if done:
                agent.train(replay_records)
                score_list.append(score)
                print('episode:', e, 'score:', score, 'max:', max(score_list))
                break
    env.close()
