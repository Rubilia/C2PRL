import time
from typing import Tuple

import numpy as np
import tensorflow as tf

from Executor.actor import Actor
from Executor.critic import Critic
from Executor.dqn import DQN
from Executor.environment import ExecutorEnvironment
from Executor.experience_buffer import ReplayBuffer
from Executor.state_representer import VariantionalAutoEncoder
from Executor.utils import SHOW_GOALS, RANDOM_ACTIONS_PERCENT, GOALS_TO_REPLAY, STATE_REPRESENTATION, R_DIM


class Agent(object):
    def __init__(self, sess, env: ExecutorEnvironment, compressor=None, batch_size=1024,
                 buffer_size=10 ** 6, scope='', GLOBAL=False, buffer=None, AC=None, device='gpu:0', is_AC=True):
        self.scope = scope
        self.test = False
        self.env: ExecutorEnvironment = env
        self.batch_size = batch_size
        self.GLOBAL = GLOBAL
        self.is_AC = is_AC

        self.goal = None
        self.state = None

        self.num_updates = 16

        if is_AC:
            self.actor: Actor = Actor(sess, env, compressor=compressor, batch_size=self.batch_size,
                                      scope=scope + 'actor_executor', device=device) if AC is None else AC[0]
            self.critic: Critic = Critic(sess, env, compressor=compressor, scope=scope + 'critic_executor',
                                         device=device) if AC is None else AC[1]
        else:
            self.Q_net = DQN(sess, self.env.a_dim, self.env.s_dim, compressor=compressor)

        self.buffer = ReplayBuffer(buffer_size, batch_size=self.batch_size, s_dim=env.s_dim,
                                   a_dim=env.a_dim) if buffer is None else buffer
        # self.dher_buffer = DHERReplay(size=10**6)
        self.goal_ep_data = []

    def train(self, compressor: VariantionalAutoEncoder=None, wait=0) -> Tuple[int, str]:
        advanced_info = ''
        # if not self.test:
        #     self.dher_buffer.start_episode()

        self.state = self.env.reset()
        self.goal = self.env.sample_goal()
        if SHOW_GOALS:
            advanced_info = 'Goal to reach: ' + str(self.goal)
        step_counter = 0
        goal_counter = 0
        done = False

        while not done:
            action = self.choose_action().flatten()
            state_, _, done = self.env.step(action)
            if wait > 0:
                time.sleep(wait)

            # Add data to learn state representation
            if STATE_REPRESENTATION and compressor is not None and compressor.add_new_data:
                compressor.add_data(self.state)

            if STATE_REPRESENTATION:
                is_goal_reached = self.env.check_goal(compressor.compress_state(np.copy(state_)), np.copy(self.goal))
            else:
                is_goal_reached = self.env.check_goal(state_, self.goal)

            step_counter += 1

            if not self.test:
                self.update_HER_replay(action, state_, is_goal_reached, compressor)
                # self.dher_buffer.add(self.state, action, is_goal_reached-1, state_, self.goal, done)

            self.state = state_

            if is_goal_reached:
                goal_counter += 1
                self.goal = self.env.sample_goal()
                if SHOW_GOALS:
                    advanced_info += '\nGoal reached at %d step\nAchieved state: ' % step_counter + str(state_) + '\n'
                    advanced_info += 'Next goal to reach: ' + str(self.goal) + '\n'

            if step_counter >= self.env.max_steps or done:
                break

        if done and not step_counter == self.env.max_steps:
            advanced_info += 'Environment is done in %d steps' % step_counter

        self.utilize_episode_data(compressor=compressor)
        if self.GLOBAL and not self.test:
            self.learn(self.num_updates)

        return goal_counter, advanced_info

    def set_params(self, global_agent):
        if self.is_AC:
            self.actor.update_params(global_agent.actor.get_params())
            self.critic.update_params(global_agent.critic.get_params())
        else:
            self.Q_net.update_params(global_agent.Q_net.get_params())

    def reach_goal(self, state: np.ndarray, goal: np.ndarray, compressor, limit: int = 50) -> Tuple[np.ndarray, float, bool, int]:
        self.state = state
        self.goal = goal
        done = False
        is_reached = False
        step_counter = 0
        total_r = 0
        while not (done or is_reached) and step_counter < limit:
            action = self.choose_action().flatten()
            state_, r, done = self.env.step(action)
            total_r += float(r)
            if STATE_REPRESENTATION:
                is_reached = self.env.check_goal(state_, self.goal)
            else:
                is_reached = self.env.check_goal(state_, self.goal)
            step_counter += 1
            self.state = state_
        return self.state, total_r, done, step_counter

    def choose_action(self):
        if self.test:
            # Standard action
            if self.is_AC:
                return self.actor.sample_action(np.reshape(self.state, (1, R_DIM)),
                                                np.reshape(self.goal, (1, R_DIM)))
            else:
                return self.Q_net.choose_action(np.reshape(self.state, (1, R_DIM)),
                                                np.reshape(self.goal, (1, R_DIM)))

        if np.random.random_sample() > RANDOM_ACTIONS_PERCENT:
            # Action with noise
            if self.is_AC:
                a = self.actor.sample_action(np.reshape(self.state, (1, R_DIM)),
                                                 np.reshape(self.goal, (1, R_DIM)))
                a += np.random.normal(np.zeros(self.env.a_dim), self.env.noise)
                return a.clip(self.env.low, self.env.high)
            else:
                return (self.Q_net.choose_action(np.reshape(self.state, (1, len(self.state))),
                                          np.reshape(self.goal, (1, len(self.goal)))) +
                 np.random.normal(np.zeros(self.env.a_dim), self.env.noise)).clip(self.env.low, self.env.high)
        # Random action
        return np.random.uniform(self.env.low, self.env.high)

    def update_HER_replay(self, action: np.ndarray, state_: np.ndarray, is_goal_reached, compressor=None):
        if is_goal_reached:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        self.buffer.add(np.copy([self.state, action, reward, state_, self.goal, done]))
        self.goal_ep_data.append(np.copy([self.state, action, None, state_, None, None, state_]))

    def utilize_episode_data(self, compressor: VariantionalAutoEncoder = None):
        if len(self.goal_ep_data) < GOALS_TO_REPLAY: return

        samples = np.random.randint(len(self.goal_ep_data), size=GOALS_TO_REPLAY)
        samples[-1] = len(self.goal_ep_data) - 1
        samples = np.sort(samples)

        for i in range(len(samples)):
            new_goal = self.goal_ep_data[int(samples[i])][6]
            for index in range(len(self.goal_ep_data)):
                data = np.copy(self.goal_ep_data[index])


                if STATE_REPRESENTATION:
                    data[4] = compressor.compress_state(new_goal).flatten()
                    data[2] = self.env.check_goal(compressor.compress_state(np.copy(new_goal)), compressor.compress_state(np.copy(data[6])) - 1.)
                else:
                    data[2] = self.env.check_goal(new_goal, data[6]) - 1.
                    data[4] = new_goal
                if data[2] == 0.:
                    data[5] = True
                else:
                    data[5] = False
                self.buffer.add(data[:-1])

        # trajectories = self.dher_buffer.sample(DHER_BATCH)
        #
        # for item in trajectories:
        #     self.buffer.add(np.copy(item))
        self.goal_ep_data = []

    def learn(self, num_updates):
        if self.buffer.size() < self.batch_size: return

        for _ in range(num_updates):
                s, a, r, s_, g, done = self.buffer.get_batch()

                if self.is_AC:
                    self.critic.update(s, a, r, s_, g,
                                       self.actor.sample_action(s_, g), done)
                    dq_da = self.critic.get_dq_da(s, g, self.actor.sample_action(s, g))
                    self.actor.update(s, g, dq_da)
                else:
                    self.Q_net.learn(s, r, s_, g, done)
                del s
                del a
                del r
                del s_
                del g
                del done

    def save(self, path: str):
        if self.is_AC:
            self.actor.save(path)
            self.critic.save(path)
        else:
            self.Q_net.save(path)

    def restore(self, path: str):
        if self.is_AC:
            self.actor.restore(path)
            self.critic.restore(path)
        else:
            self.Q_net.restore(path)
