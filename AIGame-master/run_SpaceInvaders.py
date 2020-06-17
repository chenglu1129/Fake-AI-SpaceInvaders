"""
Deep Q network,

"""
import os
# os.environ['KERAS_BACKEND']='theano'

import gym
from RL_brain_CNN import DeepQNetwork
import cv2
import numpy as np
import RL_brain_CNN
import time
from receiveThread import myThread
#from keras import backend
#from plot import plt, plot_reward
import matplotlib.pyplot as plt

# 降低tensorflow警告等级
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

env = gym.make('SpaceInvaders-v0')
env = env.unwrapped

print(env.action_space)
# print(env.observation_space)
print(env.observation_space.shape)
print(env.observation_space.high)
print(env.observation_space.low)
print(env.reward_range)

inputImageSize = (210, 160, 1)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  observation_shape=inputImageSize,
                  learning_rate=0.01, epsilon_max=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.05,
                  output_graph=True)

total_steps = 100


thread1 = myThread(1, "Thread-1", 1)
thread1.start()
total_reward_list = []
for i_episode in range(10):

    observation = env.reset()
    # 使用opencv做灰度化处理
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(observation, (inputImageSize[1], inputImageSize[0]))
    total_reward = 0
    while True:
        env.render()
        observation_, reward, done, info = env.step(env.action_space.sample())
        #print(env.action_space.sample())
        observation_, reward, done, info = env.step(4)  # 4是发送子弹 2、3分别是左右
        #if reward > 0:
             # print(reward)
        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        # 给reward做归一化处理
        reward = reward / 200
        # 使用 opencv 做灰度化处理
        observation_ = cv2.cvtColor(observation_, cv2.COLOR_BGR2GRAY)
        observation_ = cv2.resize(observation_, (inputImageSize[1], inputImageSize[0]))
        cv2.imshow('obe', observation_)

        RL.store_transition(observation, action, reward, observation_)

        total_reward += reward
        if total_steps > 1000 and total_steps % 2 == 0 and thread1.learn_flag == 1:
            t0 = time.time()
            RL.learn()
            t1 = time.time()
            if total_steps < 1010:
                print("学习一次时间：", t1 - t0)

        if done:
            total_reward_list.append(total_reward)
            print('episode: ', i_episode,
                  'total_reward: ', round(total_reward, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            # plot_reward()
            print('total reward list:', total_reward_list)
            print('max score:', max(total_reward_list))
            print('min score:', min(total_reward_list))
            print('average score:', np.mean(total_reward_list))
            break

        observation = observation_
        total_steps += 1

# RL.plot_cost()       #误差
# RL.plot_reward(total_reward_list)      #奖励
plt.plot(np.arange(len(total_reward_list)), total_reward_list)
plt.ylabel('Reward')
plt.xlabel('training episode')
plt.show()

