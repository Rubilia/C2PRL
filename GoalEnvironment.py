from typing import Tuple

import gym
import numpy as np
import tensorflow as tf

from Executor.agent import Agent
from Executor.environment import ExecutorEnvironment
from Executor.state_representer import VariantionalAutoEncoder
from Executor.utils import STATE_REPRESENTATION


class GoalEnvironment(object):
    def __init__(self, env_name: str, render=False, restore=True, is_ac=True):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.s = self.env.reset()
        # Executor config
        sess = tf.Session()
        if STATE_REPRESENTATION:
            self.compressor = VariantionalAutoEncoder(sess)
        else: self.compressor = None
        self.executor_env = ExecutorEnvironment(env_name, gym_env=self.env, render=render, wait=render, compressor=self.compressor)
        self.agent = Agent(sess=sess, env=self.executor_env, compressor=self.compressor, is_AC=is_ac)
        sess.run(tf.global_variables_initializer())
        if restore:
            try:
                self.agent.restore('data\\models\\pretrain\\')
                print('Executor restored')
                if STATE_REPRESENTATION:
                    self.compressor.restore('data\\models\\pretrain\\')
                    self.compressor.clear_data()
                    print('Compressor restored')
            except Exception:
                print('Restoration of executor has failed')
        self.LOW = np.array([self.executor_env.goal_space[x][0] for x in range(self.executor_env.goal_space.shape[0])])
        self.HIGH = np.array([self.executor_env.goal_space[x][1] for x in range(len(self.LOW))])

    def reset(self):
        return self.executor_env.reset()

    def step(self, goal) -> Tuple[np.ndarray, float, bool, int]:
        self.agent.test = True
        g = goal.flatten()
        if self.env_name == 'Pendulum-v0' and not STATE_REPRESENTATION:
            g = np.array([np.cos(g[0]), np.sin(g[0]), g[1]])
        elif self.env_name == 'Acrobot-v1':
            G = np.zeros(6)
            G[0] = np.cos(g[0])
            G[1] = np.sin(g[0])
            G[2] = np.cos(g[1])
            G[3] = np.sin(g[1])
            G[4] = g[2]
            G[5] = g[3]
            g = G
        self.s, r, done, steps = self.agent.reach_goal(self.executor_env.s_post(self.executor_env.S_storage, self.executor_env), g, compressor=self.compressor)
        if STATE_REPRESENTATION:
            return self.compressor.compress_state(self.s), r, done, steps
        else:
            return self.s, r, done, steps

    def set_render(self, render: bool):
        self.executor_env.render = render
