import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from Executor.utils import DEFAULT_DEVICE, get_executor_config, STATE_REPRESENTATION

np.random.seed(1)


# Deep Q Network off-policy
class DQN:
    def __init__(self, sess, a_dim, s_dim, lr=1e-3, y=0.99, replace_target_freq=1000, scope='q_learner', compressor=None):
        self.sess = sess
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.lr = lr
        self.y = y
        self.replace_target_freq = replace_target_freq
        self.scope = scope
        self.compressor = compressor

        self.learn_step_counter = 0
        self._build_net()

        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + 'target_net')
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + 'eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self):
        # Input placeholders
        self.s = tf.placeholder(tf.float32, [None, self.s_dim], name='s')  # input State
        self.g = tf.placeholder(tf.float32, [None, self.s_dim], name='g')  # goal
        self.td_target = tf.placeholder(tf.float32, [None, self.a_dim], name='td_target')  # TD target
        
        self.input = tf.concat([self.s, self.g], 1)

        self.q = self.build_net(self.input, self.scope + 'eval_net')

        self.q_target = self.build_net(self.input, self.scope + 'target_net')

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.td_target, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    
    def build_net(self, features, scope: str):
        neurons = get_executor_config('q_network')
        neurons = [self.s_dim * 2] + neurons + [self.a_dim]
        output = features
        with tf.variable_scope(scope):
            for id, n in enumerate(neurons[1:-1]):
                weight_init = tf.random_uniform_initializer(minval=-1 / neurons[id] ** 0.5,
                                                            maxval=1 / neurons[id] ** 0.5)
                bias_init = tf.random_uniform_initializer(minval=-1 / neurons[id] ** 0.5,
                                                          maxval=1 / neurons[id] ** 0.5)
                output = Dense(n, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init)(output)
            output = Dense(neurons[-1], activation=None,
                           kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                           bias_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(output)
        return output

    def choose_action(self, state, goal):
        return self.sess.run(self.q, feed_dict={self.s: state.reshape((1, self.s_dim)), self.g: goal.reshape((1, self.s_dim))})

    def learn(self, s, r, s_, g, done: np.ndarray):
        if self.learn_step_counter % self.replace_target_freq == 0:
            # replace target params whet its time
            self.sess.run(self.target_replace_op)
            print('target replaced')

        Q_ = self.sess.run(self.q_target, feed_dict={self.s: s_, self.g: g})
        td_target = Q_
        x = r + self.y * np.max(Q_, axis=1) * (np.ones(done.shape) - done)
        for i, item in enumerate(x):
            j = np.argmax(Q_[i])
            td_target.itemset((i, j), x[i])

        _, loss = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                    self.s: s, self.g: g, self.td_target: td_target
            })

        self.learn_step_counter += 1
        return loss

    def get_params(self):
        # return current params of the network
        return self.sess.run(self.e_params)

    def get_target_params(self):
        # return target params of the network
        return self.sess.run(self.t_params)

    def save(self, path):
        weights: np.ndarray = self.get_params()
        np.save(path + 'q_net', weights, allow_pickle=True)

    def update_params(self, params):
        ops = []
        for param1, param2 in zip(params, self.t_params):
            ops.append(tf.assign(param2, param1))

        with tf.device(DEFAULT_DEVICE):
            self.sess.run(ops)

    def update_target(self, params):
        ops = []
        for param1, param2 in zip(params, self.e_params):
            ops.append(tf.assign(param2, param1))

        with tf.device(DEFAULT_DEVICE):
            self.sess.run(ops)

    def restore(self, path):
        weights = np.load(path + 'q_net.npy', allow_pickle=True)
        self.update_params(weights)
        self.sess.run(self.target_replace_op)

