import time
import tensorflow as tf
from typing import Tuple

import gym
import numpy as np
from gym.spaces import Box

from Executor.utils import configure_training, process_done, is_goal_reached, TIME_BTWN_FRAMES, R_DIM, \
    STATE_REPRESENTATION, S_DIM, N_FRAMES


class ExecutorEnvironment(object):
    def __init__(self, env_name: str = '', gym_env=None, render: bool = False, wait=False, compressor=None):
        self.env_name = env_name
        self.compressor = compressor
        self.render = render
        self.env = gym.make(env_name) if gym_env is None else gym_env
        self.s = self.env.reset()
        if STATE_REPRESENTATION:
            self.s_dim = R_DIM
        else:
            self.s_dim = self.env.observation_space.shape[0]

        if isinstance(self.env.action_space, Box):
            self.a_dim = self.env.action_space.shape[0]
            self.a_func = lambda x: x
        else:
            self.a_dim = self.env.action_space.n
            self.a_func = self.sample_a()
        self.goal_space, self.high, self.low, self.noise, self.max_steps, self.s_func, self.s_post, self.a_repeat = configure_training(env_name)
        self.step_counter = -1
        self.wait = wait
        self.S_storage = [np.zeros_like(self.s) for _ in range(N_FRAMES)]

    def sample_a(self):
        sess = tf.Session()
        Probs = tf.placeholder(tf.float32, [self.a_dim])
        dist = tf.distributions.Categorical(probs=tf.nn.softmax(Probs))
        a = dist.sample(1)[0]

        def sample(p):
            return sess.run(a, feed_dict={Probs: p})
        return sample

    def reset(self):
        del self.S_storage
        self.step_counter = -1
        self.s = self.s_func(self.env.reset())
        self.S_storage = [np.zeros_like(self.s) for _ in range(N_FRAMES)]
        return self.s_post(self.S_storage, self).flatten()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        self.step_counter += 1
        # print(action)
        a = self.a_func(action)
        for _ in range(self.a_repeat):
            self.s, r, done, _ = self.env.step(a)
        self.s = self.s_func(self.s)
        if self.render:
            self.env.render()
            if self.wait:
                time.sleep(TIME_BTWN_FRAMES)
        self.S_storage.__delitem__(0)
        self.S_storage.append(self.s)
        return self.s_post(self.S_storage, self).flatten(), r, process_done(done, self)

    def sample_goal(self):
        goal = np.zeros(S_DIM)

        if self.env_name == 'Pendulum-v0':
            theta = np.random.uniform(-np.pi, np.pi)
            goal[0] = np.cos(theta)
            goal[1] = np.sin(theta)
            goal[-1] = np.random.uniform(self.goal_space[-1][0], self.goal_space[-1][1])
        elif self.env_name == 'Acrobot-v1':
            theta1 = np.random.uniform(-np.pi, np.pi)
            theta2 = np.random.uniform(-np.pi, np.pi)
            goal[0] = np.cos(theta1)
            goal[1] = np.sin(theta1)
            goal[2] = np.cos(theta2)
            goal[3] = np.sin(theta2)
            goal[4] = np.random.uniform(self.goal_space[4][0], self.goal_space[4][1])
            goal[5] = np.random.uniform(self.goal_space[5][0], self.goal_space[5][1])
        else:
            goal = np.zeros(self.goal_space.shape[0])
            for t in range(self.goal_space.shape[0]):
                goal[t] = np.random.uniform(self.goal_space[t][0], self.goal_space[t][1])
        return goal

    def check_goal(self, s, g):
        if STATE_REPRESENTATION:
            return is_goal_reached(self.compressor.compress_state(s), g)
        return is_goal_reached(s, g)

