import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tempfile import TemporaryFile
from Executor.environment import ExecutorEnvironment
from Executor.state_representer import VariantionalAutoEncoder
from Executor.utils import get_executor_config, STATE_REPRESENTATION, DEFAULT_DEVICE


class Actor(object):

    def __init__(self, sess, env, batch_size, compressor=None, lr=5e-4, scope='actor_executor', device='cpu:0'):


        self.layers = []



        self.sess = sess
        self.scope = scope
        self.env: ExecutorEnvironment = env
        self.device = device
        self.compressor: VariantionalAutoEncoder = compressor

        # placeholders for evaluation
        self.state = tf.placeholder(tf.float32, shape=(None, self.env.s_dim))
        self.goal = tf.placeholder(tf.float32, shape=(None, self.env.s_dim))
        self.input = tf.concat([self.state, self.goal], axis=1)

        # Create network
        self.output = self.build_actor(self.input, scope=self.scope)
        self.weights = list(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))

        # learning tools
        self.dq_da = tf.placeholder(tf.float32, shape=(None, self.env.a_dim))
        self.policy_gradient = list(map(lambda x: tf.div(x, batch_size), tf.gradients(self.output, self.weights, -self.dq_da)))
        self.train = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.policy_gradient, self.weights))

    def sample_action(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        A = self.sess.run(self.output, feed_dict={self.state: state, self.goal: goal})
        return A

    def update(self, state: np.ndarray, goal: np.ndarray, dq_da: np.ndarray):
        self.sess.run([self.train], feed_dict={self.state: state, self.goal: goal, self.dq_da: dq_da})

    def build_actor(self, features, scope: str):
        neurons = get_executor_config('actor_network')
        neurons = neurons + [self.env.a_dim]
        output = features

        self.layers.append(output)

        with tf.variable_scope(scope):
            for id, n in enumerate(neurons[:-1]):
                weight_init = tf.random_uniform_initializer(minval=-1 / neurons[id] ** 0.5,
                                                            maxval=1 / neurons[id] ** 0.5)
                bias_init = tf.random_uniform_initializer(minval=-1 / neurons[id] ** 0.5,
                                                          maxval=1 / neurons[id] ** 0.5)
                output = Dense(n, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init)(output)
                self.layers.append(output)

            output = Dense(neurons[-1], activation='linear',
                           kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                           bias_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(
                output)
            self.layers.append(output)

        return output

    def update_params(self, params):
        ops = []
        for param1, param2 in zip(params, self.weights):
            ops.append(tf.assign(param2, param1))

        with tf.device(DEFAULT_DEVICE):
            self.sess.run(ops)

    def get_params(self):
        return self.sess.run(self.weights)

    def save(self, path: str):
        path += 'actor'
        weights: np.ndarray = self.get_params()
        np.save(path, weights, allow_pickle=True)

    def restore(self, path: str):
        path += 'actor.npy'
        weights = np.load(path, allow_pickle=True)
        self.update_params(weights)
