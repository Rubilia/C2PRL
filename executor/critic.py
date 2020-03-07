import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from Executor.utils import get_executor_config, DEFAULT_DEVICE


class Critic(object):

    def __init__(self, sess, env, compressor=None, lr=1e-3, y=0.99, scope='critic_executor', device='cpu:0'):
        # params
        self.sess = sess
        self.scope = scope
        self.y = y
        self.env = env
        self.device = device
        self.compressor = compressor

        # data placeholders
        self.state = tf.placeholder(tf.float32, shape=(None, self.env.s_dim))
        self.goal = tf.placeholder(tf.float32, shape=(None, self.env.s_dim))
        self.action = tf.placeholder(tf.float32, shape=(None, self.env.a_dim))
        self.input = tf.concat([self.state, self.goal, self.action], axis=1)

        # Create network
        self.output = self.build_critic(self.input, scope=self.scope)
        self.weights = [v for v in tf.trainable_variables() if self.scope in v.op.name]

        # learning
        self.v = tf.placeholder(tf.float32, shape=(None, 1))
        self.loss = tf.reduce_mean(tf.square(self.v - self.output))
        self.train = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.dq_da = tf.gradients(self.output, self.action)
        self.saver = tf.train.Saver(max_to_keep=10**6)

    def get_dq_da(self, state, goal, action):
        return self.sess.run(self.dq_da, feed_dict={self.state: state, self.goal: goal, self.action: action})[0]

    def update(self, s, a, r, s_, g, a_, done) -> float:
        V = self.sess.run(self.output, feed_dict={self.state: s_, self.goal: g, self.action: a_})
        for i in range(len(V)):
            if done[i]:
                V[i] = r[i]
            else:
                V[i] = r[i] + self.y * V[i][0]
        loss, _ = self.sess.run([self.loss, self.train], feed_dict={self.state: s, self.goal: g,
                                                                    self.action: a, self.v: V})
        return loss

    def build_critic(self, features, scope):
        neurons = get_executor_config('critic_network')
        neurons = neurons + [1]
        output = features
        with tf.variable_scope(scope):
            for id, n in enumerate(neurons[:-1]):
                weight_init = tf.random_uniform_initializer(minval=-1 / neurons[id] ** 0.5,
                                                            maxval=1 / neurons[id] ** 0.5)
                bias_init = tf.random_uniform_initializer(minval=-1 / neurons[id] ** 0.5,
                                                          maxval=1 / neurons[id] ** 0.5)
                output = Dense(n, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init)(output)
            output = Dense(neurons[-1], activation=None,
                           kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                           bias_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(
                output)
        return output

    def update_params(self, params):
        ops = []
        for param1, param2 in zip(params, self.weights):
            ops.append(tf.assign(param2, param1))

        with tf.device(DEFAULT_DEVICE):
            self.sess.run(ops)

    def get_params(self):
        return self.sess.run(self.weights)

    def save(self, path):
        path += 'critic'
        weights: np.ndarray = self.get_params()
        np.save(path, weights, allow_pickle=True)

    def restore(self, path):
        path += 'critic.npy'
        weights = np.load(path, allow_pickle=True)
        self.update_params(weights)
